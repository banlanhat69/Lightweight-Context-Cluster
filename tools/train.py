from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    parser.add_argument("--resume")
    parser.add_argument("--teacher-config")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--limit-test-batches", type=int)
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


def is_controlled_comparison(cfg: dict[str, Any]) -> bool:
    purpose = str(cfg.get("protocol", {}).get("purpose", "")).lower()
    return "architecture_comparison" in purpose or "compare_model_architectures" in purpose


def adamw_parameter_groups(
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
    if args.resume and is_controlled_comparison(cfg):
        raise ValueError(
            "Controlled-comparison runs do not support --resume because optimizer, scheduler, "
            "and DataLoader RNG states are not restored. Start a fresh paired-seed run."
        )
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
    train_loader, val_loader, test_loader = build_loaders(cfg.get("data", {}), include_test=not skip_test)
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
    if cfg["model"].get("name") == "hbcc_food101_best" and not (
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
        if cfg["model"].get("name") == "hbcc_food101_best":
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
    if args.resume:
        print(f"setup: loading checkpoint {args.resume}", flush=True)
        load_checkpoint(model, args.resume, device, strict=False)

    optimizer = torch.optim.AdamW(
        adamw_parameter_groups(
            model,
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        ),
        lr=float(train_cfg.get("lr", 1e-3)),
    )
    epochs = int(train_cfg.get("epochs", 200))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
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
    print(f"train: starting epochs={epochs} skip_test={skip_test}", flush=True)
    best_acc = 0.0
    best_epoch = -1
    for epoch in range(epochs):
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
            batch_augment=batch_augment,
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
        is_best = val_metrics["val_acc1"] >= best_acc
        if is_best:
            best_acc = val_metrics["val_acc1"]
            best_epoch = epoch
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc1": best_acc,
            "best_epoch": best_epoch,
            "config": cfg,
        }
        save_checkpoint(checkpoint, output_dir / "latest.pth")
        if is_best:
            save_checkpoint(checkpoint, output_dir / "best.pth")
        should_print = epoch == epochs - 1
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
    if not skip_test and best_path.exists():
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
        test_record = {"phase": "test", "epoch": best_epoch, "checkpoint": str(best_path.name), **test_metrics}
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
