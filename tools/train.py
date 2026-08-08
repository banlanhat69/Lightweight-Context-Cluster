from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.amp import GradScaler

from lightweight_hbcc.augment import MixupCutmix
from lightweight_hbcc.config import apply_overrides, load_config, save_config
from lightweight_hbcc.data import (
    build_loaders,
    num_classes_for_dataset,
    validate_no_augmentation_config,
)
from lightweight_hbcc.engine import (
    evaluate,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
    write_metrics,
)
from lightweight_hbcc.losses import DistillationLoss
from lightweight_hbcc.models import build_model


_REMOVED_BATCH_AUGMENTATION_KEYS = {
    "mixup_alpha",
    "cutmix_alpha",
    "cutmix_prob",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HBCC/CoC/baseline image classification models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="runs")
    parser.add_argument(
        "--resume",
        help="Resume a run from latest.pth (best.pth is accepted but not recommended).",
    )
    parser.add_argument("--teacher-config")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--limit-test-batches", type=int)
    parser.add_argument(
        "--run-until-epoch",
        type=int,
        help=(
            "Stop after this many total epochs while preserving the scheduler's full "
            "train.epochs horizon. Useful for splitting one run across Kaggle sessions."
        ),
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip final evaluation on the held-out test split.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars for notebook/log runs.")
    parser.add_argument("--print-every", type=int, default=1, help="Print an epoch summary every N epochs. Use 0 to print only the final epoch.")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def resolve_seed_config(cfg: dict[str, Any]) -> tuple[int, int, bool]:
    """Resolve run and loader seeds, with data.loader_seed as the only explicit override."""

    train_cfg = cfg.setdefault("train", {})
    data_cfg = cfg.setdefault("data", {})
    seed = int(train_cfg.get("seed", 42))
    deterministic = bool(train_cfg.get("deterministic", False))
    loader_seed = int(data_cfg.get("loader_seed", seed))
    train_cfg["seed"] = seed
    data_cfg["loader_seed"] = loader_seed
    return seed, loader_seed, deterministic


def _resume_comparable_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return trajectory-defining settings while ignoring machine-local paths."""

    comparable = {
        section: dict(cfg.get(section, {}))
        for section in ("protocol", "data", "train", "model")
    }
    for key in (
        "root",
        "download",
        "workers",
        "persistent_workers",
        "pin_memory",
    ):
        comparable["data"].pop(key, None)
    comparable["train"].pop("skip_test", None)
    return comparable


def validate_resume_checkpoint(
    cfg: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_path: str | Path,
) -> None:
    """Reject accidental cross-model or cross-recipe resumes."""

    if "epoch" not in checkpoint:
        raise ValueError(f"Resume checkpoint has no completed epoch: {checkpoint_path}")
    checkpoint_cfg = checkpoint.get("config")
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError(
            "Resume checkpoint must contain its full config. Use a latest.pth/best.pth "
            "created by this project."
        )
    current = _resume_comparable_config(cfg)
    saved = _resume_comparable_config(checkpoint_cfg)
    mismatched = [
        section for section in current if current[section] != saved.get(section)
    ]
    if int(checkpoint.get("format_version", 1)) >= 2:
        current_data = cfg.get("data", {})
        saved_data = checkpoint_cfg.get("data", {})
        for key in ("workers", "persistent_workers"):
            if (
                current_data.get(key) != saved_data.get(key)
                and "data" not in mismatched
            ):
                mismatched.append("data")
    if mismatched:
        raise ValueError(
            "Resume config does not match the checkpoint in trajectory-defining "
            f"sections: {', '.join(mismatched)}. Keep the model, recipe, batch size, "
            "seed, and total epoch budget unchanged when resuming."
        )


def capture_rng_state(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader | None,
) -> dict[str, Any]:
    """Capture process and DataLoader-generator RNG at an epoch boundary."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "train_loader": train_loader.generator.get_state(),
        "val_loader": val_loader.generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if test_loader is not None:
        state["test_loader"] = test_loader.generator.get_state()
    return state


def restore_rng_state(
    state: dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader | None,
) -> None:
    """Restore RNG captured by :func:`capture_rng_state`."""

    required = ("python", "numpy", "torch", "train_loader", "val_loader")
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(f"Checkpoint RNG state is incomplete: {', '.join(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    train_loader.generator.set_state(state["train_loader"].cpu())
    val_loader.generator.set_state(state["val_loader"].cpu())
    if test_loader is not None and "test_loader" in state:
        test_loader.generator.set_state(state["test_loader"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def prepare_metrics_for_resume(
    metrics_path: Path,
    test_metrics_path: Path,
    start_epoch: int,
    checkpoint_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restore/trim history and discard a stale final-test record."""

    existing: list[dict[str, Any]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
    candidates = []
    for source in (existing, checkpoint_history):
        candidates.append(
            [
                record
                for record in source
                if record.get("phase") != "test"
                and int(record.get("epoch", -1)) < start_epoch
            ]
        )
    history = max(candidates, key=len)
    with metrics_path.open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if test_metrics_path.exists():
        test_metrics_path.unlink()
    return history


def cpu_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone a portable CPU copy of model weights for best-checkpoint recovery."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def read_checkpoint_file(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a mapping: {path}")
    return checkpoint


def weight_decay_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Exclude biases, norm scales, and scalar gates from weight decay."""

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def validate_training_mode(
    cfg: dict[str, Any],
    teacher_config: str | None,
    teacher_checkpoint: str | None,
) -> bool:
    """Validate CE/KD mode and guard augmentation-incompatible settings."""

    train_cfg = cfg.setdefault("train", {})
    legacy_keys = sorted(_REMOVED_BATCH_AUGMENTATION_KEYS.intersection(train_cfg))
    augmentation = str(cfg.get("protocol", {}).get("augmentation", "none")).lower()
    if legacy_keys and augmentation == "none":
        joined = ", ".join(legacy_keys)
        raise ValueError(
            "Batch augmentation settings require a non-none protocol.augmentation. "
            f"Remove these settings or select an augmentation recipe: {joined}"
        )

    if bool(teacher_config) != bool(teacher_checkpoint):
        raise ValueError("--teacher-config and --teacher-checkpoint must be provided together.")

    has_teacher = bool(teacher_config and teacher_checkpoint)
    kd_alpha = float(train_cfg.get("kd_alpha", 0.0))
    session = str(cfg.get("protocol", {}).get("session", "")).lower()
    model_name = str(cfg.get("model", {}).get("name", ""))
    if has_teacher:
        kd_method = str(train_cfg.get("kd_method", "reverse_kd")).lower()
        if kd_method not in {"standard", "reverse_kd", "dkd", "dkd_at"}:
            raise ValueError(
                "HBCC distillation method must be 'standard', 'reverse_kd', "
                "'dkd', or 'dkd_at'."
            )
        if model_name != "hbcc":
            raise ValueError("Knowledge distillation is supported only for HBCC students.")
        if kd_method in {"standard", "reverse_kd"} and kd_alpha <= 0.0:
            raise ValueError(f"{kd_method} requires train.kd_alpha > 0.")
        if kd_method in {"dkd", "dkd_at"}:
            tckd_weight = float(train_cfg.get("dkd_tckd_weight", 1.0))
            nckd_weight = float(train_cfg.get("dkd_nckd_weight", 4.0))
            dkd_scale = float(train_cfg.get("dkd_scale", 1.0))
            if tckd_weight < 0.0 or nckd_weight < 0.0:
                raise ValueError("DKD weights must be non-negative.")
            if tckd_weight == 0.0 and nckd_weight == 0.0:
                raise ValueError("DKD requires at least one positive weight.")
            if dkd_scale <= 0.0:
                raise ValueError("DKD requires train.dkd_scale > 0.")
        if kd_method == "dkd_at":
            feature_method = str(
                train_cfg.get("feature_kd_method", "attention")
            ).lower()
            feature_weight = float(train_cfg.get("feature_kd_weight", 0.25))
            feature_stages = [
                int(stage) for stage in train_cfg.get("feature_kd_stages", [2, 3, 4])
            ]
            feature_warmup = int(train_cfg.get("feature_kd_warmup_epochs", 20))
            if feature_method != "attention":
                raise ValueError("dkd_at requires train.feature_kd_method=attention.")
            if feature_weight <= 0.0:
                raise ValueError("dkd_at requires train.feature_kd_weight > 0.")
            if not feature_stages or len(feature_stages) != len(set(feature_stages)):
                raise ValueError("feature_kd_stages must be non-empty and unique.")
            if any(stage < 1 or stage > 4 for stage in feature_stages):
                raise ValueError("feature_kd_stages values must be in [1, 4].")
            if feature_warmup < 0:
                raise ValueError("feature_kd_warmup_epochs must be non-negative.")
        if session and session != "kd":
            raise ValueError("A teacher checkpoint is valid only for protocol.session=kd.")
    else:
        kd_method = str(train_cfg.get("kd_method", "none")).lower()
        if kd_method not in {"none", "ce"}:
            raise ValueError("A distillation method requires a teacher checkpoint.")
        if kd_alpha != 0.0:
            raise ValueError("train.kd_alpha must be 0 when no ResNet-18 teacher is provided.")
        if session == "kd":
            raise ValueError("protocol.session=kd requires a ResNet-18 teacher checkpoint.")
        kd_method = "none"
    train_cfg["kd_method"] = kd_method
    return has_teacher


def validate_teacher_artifacts(
    student_cfg: dict[str, Any],
    teacher_cfg: dict[str, Any],
    checkpoint_cfg: dict[str, Any] | None,
) -> None:
    """Ensure KD cannot accidentally use an augmented or mismatched teacher."""

    if checkpoint_cfg is None:
        raise ValueError("The ResNet-18 teacher checkpoint must contain its training config.")
    validate_no_augmentation_config(teacher_cfg.get("data", {}))
    validate_no_augmentation_config(checkpoint_cfg.get("data", {}))

    errors: list[str] = []
    if teacher_cfg.get("model", {}).get("name") != "resnet18_cifar":
        errors.append("teacher config model must be resnet18_cifar")
    if teacher_cfg.get("protocol", {}).get("session") != "baseline":
        errors.append("teacher config must come from the baseline session")
    if teacher_cfg.get("protocol", {}).get("augmentation") != "none":
        errors.append("teacher config must declare augmentation=none")
    teacher_train = teacher_cfg.get("train", {})
    if float(teacher_train.get("kd_alpha", 0.0)) != 0.0:
        errors.append("teacher must be trained with CE (kd_alpha=0 or omitted)")
    if str(teacher_train.get("kd_method", "none")).lower() not in {"none", "ce"}:
        errors.append("teacher config must declare CE-only training")

    student_dataset = student_cfg.get("protocol", {}).get("dataset")
    teacher_dataset = teacher_cfg.get("protocol", {}).get("dataset")
    if student_dataset != teacher_dataset:
        errors.append(
            f"teacher dataset {teacher_dataset!r} does not match student dataset {student_dataset!r}"
        )
    student_seed = int(student_cfg.get("train", {}).get("seed", -1))
    teacher_seed = int(teacher_cfg.get("train", {}).get("seed", -2))
    if student_seed != teacher_seed:
        errors.append(
            f"teacher seed {teacher_seed} does not match student seed {student_seed}"
        )
    for section in ("protocol", "data", "train", "model"):
        if checkpoint_cfg.get(section) != teacher_cfg.get(section):
            errors.append(f"teacher checkpoint {section} metadata does not match teacher config")
    if errors:
        raise ValueError("Invalid ResNet-18 teacher:\n- " + "\n- ".join(errors))


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    dataset_name = cfg.get("data", {}).get("name", "cifar10")
    cfg.setdefault("model", {})["num_classes"] = cfg["model"].get("num_classes", num_classes_for_dataset(dataset_name))
    seed, loader_seed, deterministic = resolve_seed_config(cfg)
    train_cfg = cfg["train"]
    epochs = int(train_cfg.get("epochs", 200))
    run_until_epoch = (
        int(args.run_until_epoch) if args.run_until_epoch is not None else epochs
    )
    if not 1 <= run_until_epoch <= epochs:
        raise ValueError(
            f"--run-until-epoch must be in [1, {epochs}], got {run_until_epoch}."
        )
    has_teacher = validate_training_mode(cfg, args.teacher_config, args.teacher_checkpoint)
    kd_method = str(train_cfg.get("kd_method", "none"))
    kl_direction = {
        "standard": "teacher||student",
        "reverse_kd": "student||teacher",
        "dkd": "decoupled_teacher||student",
        "dkd_at": "decoupled_teacher||student+multi_stage_attention",
    }.get(kd_method)
    cfg["distillation"] = {
        "enabled": has_teacher,
        "method": kd_method,
        "teacher_model": "resnet18_cifar" if has_teacher else None,
        "teacher_config": str(args.teacher_config) if has_teacher else None,
        "teacher_checkpoint": str(args.teacher_checkpoint) if has_teacher else None,
        "alpha": (
            float(train_cfg.get("kd_alpha", 0.0))
            if kd_method in {"standard", "reverse_kd"}
            else None
        ),
        "temperature": float(train_cfg.get("kd_temperature", 4.0)),
        "kl_direction": kl_direction if has_teacher else None,
        "dkd_tckd_weight": (
            float(train_cfg.get("dkd_tckd_weight", 1.0))
            if kd_method in {"dkd", "dkd_at"}
            else None
        ),
        "dkd_nckd_weight": (
            float(train_cfg.get("dkd_nckd_weight", 4.0))
            if kd_method in {"dkd", "dkd_at"}
            else None
        ),
        "dkd_scale": (
            float(train_cfg.get("dkd_scale", 1.0))
            if kd_method in {"dkd", "dkd_at"}
            else None
        ),
        "dkd_warmup_epochs": (
            int(train_cfg.get("dkd_warmup_epochs", 20))
            if kd_method in {"dkd", "dkd_at"}
            else None
        ),
        "feature_kd_method": "attention" if kd_method == "dkd_at" else None,
        "feature_kd_weight": (
            float(train_cfg.get("feature_kd_weight", 0.25))
            if kd_method == "dkd_at"
            else None
        ),
        "feature_kd_stages": (
            [int(stage) for stage in train_cfg.get("feature_kd_stages", [2, 3, 4])]
            if kd_method == "dkd_at"
            else None
        ),
        "feature_kd_warmup_epochs": (
            int(train_cfg.get("feature_kd_warmup_epochs", 20))
            if kd_method == "dkd_at"
            else None
        ),
    }
    seed_everything(seed, deterministic=deterministic)
    skip_test = args.skip_test or bool(train_cfg.get("skip_test", False))
    device = resolve_device(args.device)
    teacher = None
    if has_teacher:
        assert args.teacher_config is not None
        assert args.teacher_checkpoint is not None
        print("setup: validating frozen ResNet-18 teacher...", flush=True)
        teacher_cfg = load_config(args.teacher_config)
        teacher_cfg.setdefault("model", {})["num_classes"] = cfg["model"]["num_classes"]
        teacher = build_model(teacher_cfg).to(device)
        teacher_checkpoint = load_checkpoint(
            teacher,
            args.teacher_checkpoint,
            device,
            strict=True,
        )
        embedded_config = teacher_checkpoint.get("config")
        validate_teacher_artifacts(
            cfg,
            teacher_cfg,
            embedded_config if isinstance(embedded_config, dict) else None,
        )
        teacher.eval()
        teacher.requires_grad_(False)
        print(
            f"setup: frozen ResNet-18 teacher loaded from {args.teacher_checkpoint}",
            flush=True,
        )
        # Teacher construction consumes RNG state. Restore the run seed so the
        # HBCC student starts from the same initialization as its CE counterpart.
        seed_everything(seed, deterministic=deterministic)

    output_dir = Path(args.output) / cfg.get("experiment", {}).get("name", Path(args.config).stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")
    metrics_path = output_dir / "metrics.jsonl"
    test_metrics_path = output_dir / "test_metrics.json"
    if not args.resume:
        for stale_path in (metrics_path, test_metrics_path):
            if stale_path.exists():
                stale_path.unlink()

    setup_start = time.perf_counter()
    print(
        f"setup: dataset={dataset_name} output={output_dir} device={device} "
        f"seed={seed} loader_seed={loader_seed} deterministic={deterministic}",
        flush=True,
    )
    print("setup: building data loaders...", flush=True)
    train_loader, val_loader, test_loader = build_loaders(
        cfg.get("data", {}),
        include_test=not skip_test and run_until_epoch == epochs,
    )
    print(
        "setup: data ready "
        f"train_samples={len(train_loader.dataset)} train_batches={len(train_loader)} "
        f"val_samples={len(val_loader.dataset)} val_batches={len(val_loader)} "
        f"test_enabled={test_loader is not None} elapsed={time.perf_counter() - setup_start:.1f}s",
        flush=True,
    )
    print("setup: building model...", flush=True)
    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"setup: model ready params={param_count} trainable_params={trainable_param_count} "
        f"elapsed={time.perf_counter() - setup_start:.1f}s",
        flush=True,
    )
    if cfg["model"].get("name") in {
        "hbcc_food101_best",
        "hbcc_food101_fair",
        "hbcc_food101_best100",
    } and not (
        2_400_000 <= param_count <= 2_700_000
    ):
        raise RuntimeError(
            "Canonical HBCC-Food101 must remain near 2.5M parameters; "
            f"got {param_count:,}."
        )
    image_size = int(cfg.get("data", {}).get("image_size", 32))
    expected_classes = int(cfg["model"]["num_classes"])
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        sample = torch.zeros(1, 3, image_size, image_size, device=device)
        sample_output = model(sample)
        if tuple(sample_output.shape) != (1, expected_classes):
            raise RuntimeError(
                f"Model preflight expected logits {(1, expected_classes)}, "
                f"got {tuple(sample_output.shape)}."
            )
        if cfg["model"].get("name") in {
            "hbcc_food101_best",
            "hbcc_food101_fair",
            "hbcc_food101_best100",
        }:
            feature_shapes = [
                tuple(feature.shape[-2:])
                for feature in model.forward_intermediates(sample)
            ]
            expected_shapes = [(56, 56), (28, 28), (14, 14), (7, 7)]
            if feature_shapes != expected_shapes:
                raise RuntimeError(
                    f"HBCC Food-101 feature shapes must be {expected_shapes}, "
                    f"got {feature_shapes}."
                )
            print(f"setup: HBCC feature shapes={feature_shapes}", flush=True)
    model.train(was_training)
    resume_checkpoint: dict[str, Any] | None = None
    resume_path: Path | None = None
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"setup: validating resume checkpoint {resume_path}", flush=True)
        resume_checkpoint = read_checkpoint_file(resume_path)
        resume_model_state = resume_checkpoint.get("model", resume_checkpoint)
        model.load_state_dict(resume_model_state, strict=True)
        validate_resume_checkpoint(cfg, resume_checkpoint, resume_path)

    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    parameter_groups = weight_decay_parameter_groups(
        model,
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=float(train_cfg.get("lr", 1e-3)),
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=float(train_cfg.get("lr", 0.1)),
            momentum=float(train_cfg.get("momentum", 0.9)),
            nesterov=bool(train_cfg.get("nesterov", True)),
        )
    else:
        raise ValueError("train.optimizer must be 'adamw' or 'sgd'.")
    print(f"setup: optimizer={optimizer_name}", flush=True)
    setup_metadata = {
        "model": cfg["model"].get("name"),
        "optimizer": optimizer_name,
        "parameter_count": param_count,
        "trainable_parameter_count": trainable_param_count,
        "epoch_budget": int(train_cfg.get("epochs", 200)),
        "run_until_epoch": run_until_epoch,
        "pretrained": False,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "resume_completed_epochs": (
            int(resume_checkpoint["epoch"]) + 1
            if resume_checkpoint is not None
            else 0
        ),
        "checkpoint_format_version": (
            int(resume_checkpoint.get("format_version", 1))
            if resume_checkpoint is not None
            else None
        ),
    }
    with (output_dir / "setup.json").open("w", encoding="utf-8") as handle:
        json.dump(setup_metadata, handle, indent=2)
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(train_cfg.get("warmup_start_factor", 0.01)),
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=float(train_cfg.get("min_lr", 0.0)),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=float(train_cfg.get("min_lr", 0.0)),
        )
    criterion = DistillationLoss(
        method=str(train_cfg.get("kd_method", "none")),
        temperature=float(train_cfg.get("kd_temperature", 4.0)),
        alpha=float(train_cfg.get("kd_alpha", 0.0)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        dkd_tckd_weight=float(train_cfg.get("dkd_tckd_weight", 1.0)),
        dkd_nckd_weight=float(train_cfg.get("dkd_nckd_weight", 4.0)),
        dkd_scale=float(train_cfg.get("dkd_scale", 1.0)),
        dkd_warmup_epochs=int(train_cfg.get("dkd_warmup_epochs", 20)),
        feature_kd_weight=float(train_cfg.get("feature_kd_weight", 0.25)),
        feature_kd_stages=[
            int(stage) for stage in train_cfg.get("feature_kd_stages", [2, 3, 4])
        ],
        feature_kd_warmup_epochs=int(
            train_cfg.get("feature_kd_warmup_epochs", 20)
        ),
    )
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))
    cutmix_alpha = float(train_cfg.get("cutmix_alpha", 0.0))
    batch_augment = None
    mixup_cutmix_off_epoch = int(
        train_cfg.get("mixup_cutmix_off_epoch", train_cfg.get("epochs", 0))
    )
    if not 0 <= mixup_cutmix_off_epoch <= int(train_cfg.get("epochs", 0)):
        raise ValueError(
            "train.mixup_cutmix_off_epoch must be between 0 and train.epochs."
        )
    if mixup_alpha > 0.0 or cutmix_alpha > 0.0:
        if has_teacher:
            raise ValueError("Mixup/CutMix are currently supported only for CE runs.")
        batch_augment = MixupCutmix(
            num_classes=int(cfg["model"]["num_classes"]),
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            probability=float(train_cfg.get("mixup_cutmix_prob", 1.0)),
            switch_probability=float(train_cfg.get("mixup_switch_prob", 0.5)),
            seed=int(train_cfg.get("seed", 42)) + 1000,
        )
    amp = bool(train_cfg.get("amp", True))
    scaler = GradScaler(device.type, enabled=amp and device.type == "cuda")
    start_epoch = 0
    best_acc = 0.0
    best_epoch = -1
    best_model_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        if start_epoch > run_until_epoch:
            raise ValueError(
                f"Checkpoint already has {start_epoch} completed epochs, beyond "
                f"--run-until-epoch={run_until_epoch}."
            )
        if "optimizer" not in resume_checkpoint:
            raise ValueError("Resume checkpoint does not contain optimizer state.")
        scheduler_state = resume_checkpoint.get("scheduler")
        if scheduler_state is None:
            # Legacy project checkpoints stored the optimizer but not the scheduler.
            # Advance a fresh scheduler to the completed epoch, then restore the exact
            # optimizer LR/momentum state. This is the closest safe compatibility path.
            for _ in range(start_epoch):
                scheduler.step()
            print(
                "resume warning: legacy checkpoint has no scheduler state; "
                "the schedule was reconstructed from its saved config.",
                flush=True,
            )
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        if "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        else:
            print(
                "resume warning: legacy checkpoint has no AMP scaler state.",
                flush=True,
            )
        if batch_augment is not None:
            batch_augment_state = resume_checkpoint.get("batch_augment")
            if batch_augment_state is not None:
                batch_augment.load_state_dict(batch_augment_state)
            else:
                print(
                    "resume warning: legacy checkpoint has no Mixup/CutMix RNG state.",
                    flush=True,
                )
        checkpoint_history = resume_checkpoint.get("history", [])
        if not isinstance(checkpoint_history, list):
            raise TypeError("Checkpoint history must be a list of epoch records.")
        history = prepare_metrics_for_resume(
            metrics_path,
            test_metrics_path,
            start_epoch,
            checkpoint_history,
        )
        best_acc = float(resume_checkpoint.get("best_acc1", 0.0))
        best_epoch = int(resume_checkpoint.get("best_epoch", -1))
        embedded_best = resume_checkpoint.get("best_model")
        assert resume_path is not None
        output_best_path = output_dir / "best.pth"
        output_best_valid = False
        if isinstance(embedded_best, dict):
            best_model_state = embedded_best
        best_candidates = [output_best_path, resume_path.with_name("best.pth")]
        for candidate in best_candidates:
            if not candidate.is_file():
                continue
            candidate_checkpoint = read_checkpoint_file(candidate)
            try:
                validate_resume_checkpoint(cfg, candidate_checkpoint, candidate)
            except ValueError:
                continue
            if int(candidate_checkpoint.get("epoch", -1)) != best_epoch:
                continue
            candidate_model = candidate_checkpoint.get("model")
            if isinstance(candidate_model, dict):
                if best_model_state is None:
                    best_model_state = candidate_model
                if candidate.resolve() != output_best_path.resolve():
                    save_checkpoint(candidate_checkpoint, output_best_path)
                output_best_valid = True
                break
        if best_model_state is None and best_epoch == int(resume_checkpoint["epoch"]):
            best_model_state = cpu_model_state(model)
        if best_model_state is None:
            raise FileNotFoundError(
                "The resume checkpoint does not embed the previous best weights. "
                "For a legacy latest.pth, upload its matching best.pth in the same "
                "directory (or resume from best.pth and accept restarting at that epoch)."
            )
        if not output_best_valid:
            recovered_best = {
                key: value
                for key, value in resume_checkpoint.items()
                if key
                not in {
                    "optimizer",
                    "scheduler",
                    "scaler",
                    "rng_state",
                    "best_model",
                }
            }
            recovered_best.update(
                {
                    "checkpoint_role": "evaluation_only_recovered_best",
                    "epoch": best_epoch,
                    "model": best_model_state,
                }
            )
            save_checkpoint(recovered_best, output_best_path)
        rng_state = resume_checkpoint.get("rng_state")
        if rng_state is not None:
            restore_rng_state(
                rng_state,
                train_loader,
                val_loader,
                test_loader,
            )
        else:
            print(
                "resume warning: legacy checkpoint has no process/DataLoader RNG state; "
                "continuation is valid but not bitwise-identical to an uninterrupted run.",
                flush=True,
            )
        for heavy_key in ("model", "optimizer", "scheduler", "scaler", "rng_state"):
            resume_checkpoint.pop(heavy_key, None)
        del resume_model_state
        print(
            f"resume: completed_epochs={start_epoch} next_epoch={start_epoch + 1} "
            f"best_epoch={best_epoch + 1 if best_epoch >= 0 else 'n/a'} "
            f"best_val_acc1={best_acc:.2f}",
            flush=True,
        )
    if start_epoch == run_until_epoch:
        print("train: no epochs remain in this session.", flush=True)
    else:
        print(
            f"train: starting epoch={start_epoch + 1} "
            f"run_until_epoch={run_until_epoch} total_epochs={epochs} "
            f"skip_test={skip_test}",
            flush=True,
        )
    for epoch in range(start_epoch, run_until_epoch):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            epoch,
            teacher=teacher,
            amp=amp,
            scaler=scaler,
            limit_batches=args.limit_train_batches,
            progress=not args.no_progress,
            grad_clip_norm=(
                float(train_cfg["grad_clip_norm"])
                if train_cfg.get("grad_clip_norm") is not None
                else None
            ),
            batch_augment=(
                batch_augment if epoch < mixup_cutmix_off_epoch else None
            ),
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            amp=amp,
            limit_batches=args.limit_val_batches,
            progress=not args.no_progress,
            prefix="val",
        )
        scheduler.step()
        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], **train_metrics, **val_metrics}
        write_metrics(record, metrics_path)
        history.append(record)
        is_best = val_metrics["val_acc1"] >= best_acc
        if is_best:
            best_acc = val_metrics["val_acc1"]
            best_epoch = epoch
            best_model_state = cpu_model_state(model)
        checkpoint = {
            "format_version": 2,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": capture_rng_state(
                train_loader,
                val_loader,
                test_loader,
            ),
            "batch_augment": (
                batch_augment.state_dict() if batch_augment is not None else None
            ),
            "best_acc1": best_acc,
            "best_epoch": best_epoch,
            "best_model": best_model_state,
            "history": history,
            "config": cfg,
        }
        save_checkpoint(checkpoint, output_dir / "latest.pth")
        if is_best:
            best_checkpoint = dict(checkpoint)
            best_checkpoint.pop("best_model", None)
            save_checkpoint(best_checkpoint, output_dir / "best.pth")
        should_print = epoch == run_until_epoch - 1
        if args.print_every > 0:
            should_print = should_print or (epoch % args.print_every == 0)
        if should_print:
            if args.no_progress:
                line = (
                    "epoch={epoch} lr={lr:.6g} train_acc1={train_acc1:.2f} "
                    "train_loss={train_loss:.4f} val_acc1={val_acc1:.2f} val_loss={val_loss:.4f}".format(**record)
                )
                if "val_acc5" in record:
                    line += " val_acc5={val_acc5:.2f}".format(**record)
                for metric_name in (
                    "train_ce_loss",
                    "train_kd_loss",
                    "train_tckd_loss",
                    "train_nckd_loss",
                    "train_dkd_loss",
                    "train_attention_loss",
                    "train_cluster_balance_loss",
                    "train_cluster_balance_weighted_loss",
                ):
                    if metric_name in record:
                        label = metric_name.removeprefix("train_").removesuffix("_loss")
                        line += f" {label}={record[metric_name]:.4f}"
                print(line)
            else:
                print(json.dumps(record, indent=2))

    best_path = output_dir / "best.pth"
    training_complete = run_until_epoch == epochs
    if not training_complete:
        print(
            f"train: paused after {run_until_epoch}/{epochs} epochs; "
            f"resume from {output_dir / 'latest.pth'}",
            flush=True,
        )
    if not skip_test and training_complete:
        if not best_path.exists():
            raise FileNotFoundError(
                "Final test requires best.pth, but it could not be recovered from the "
                "resume artifacts and no later epoch produced a new best checkpoint."
            )
        if test_loader is None:
            raise RuntimeError("Final test evaluation was requested, but no test loader was created.")
        load_checkpoint(model, best_path, device, strict=True)
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            amp=amp,
            limit_batches=args.limit_test_batches,
            progress=not args.no_progress,
            prefix="test",
        )
        test_record = {
            "phase": "test",
            "epoch": best_epoch,
            "best_val_acc1": best_acc,
            "checkpoint": str(best_path.name),
            **test_metrics,
        }
        write_metrics(test_record, metrics_path)
        with test_metrics_path.open("w", encoding="utf-8") as f:
            json.dump(test_record, f, indent=2)
        if args.no_progress:
            line = (
                "test checkpoint={checkpoint} epoch={epoch} test_acc1={test_acc1:.2f} "
                "test_loss={test_loss:.4f}".format(**test_record)
            )
            if "test_acc5" in test_record:
                line += " test_acc5={test_acc5:.2f}".format(**test_record)
            print(line)
        else:
            print(json.dumps(test_record, indent=2))


if __name__ == "__main__":
    main()
