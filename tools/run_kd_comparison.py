from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from lightweight_hbcc.config import deep_update, load_config, save_config
from lightweight_hbcc.data import validate_no_augmentation_config
from lightweight_hbcc.models import build_model
from lightweight_hbcc.models.hbcc import validate_hbcc_pdf_cifar_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "cifar_fair"
CATALOG_PATH = CONFIG_ROOT / "model_catalog.yaml"
RECIPE_PATHS = {
    "cifar10": CONFIG_ROOT / "cifar10_no_augmentation.yaml",
    "cifar100": CONFIG_ROOT / "cifar100_no_augmentation.yaml",
}
METHODS = ("standard", "dkd")
TEACHER_MODEL_NAME = "resnet18_cifar"
_REMOVED_BATCH_AUGMENTATION_KEYS = {"mixup_alpha", "cutmix_alpha", "cutmix_prob"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare standard HBCC KD and HBCC-DKD using a frozen, verified "
            "no-augmentation ResNet-18 checkpoint."
        )
    )
    parser.add_argument("--dataset", choices=sorted(RECIPE_PATHS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--students", nargs="+", default=["hbcc_small", "hbcc_medium"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument(
        "--teacher-config",
        help="Optional config.yaml. If omitted, config is extracted from best.pth.",
    )
    parser.add_argument("--expected-teacher-epochs", type=int)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--standard-alpha", type=float, default=0.5)
    parser.add_argument("--dkd-tckd-weight", type=float, default=1.0)
    parser.add_argument("--dkd-nckd-weight", type=float, default=4.0)
    parser.add_argument("--dkd-warmup-epochs", type=int, default=20)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--download-data",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _validate_unique(label: str, values: list[Any]) -> None:
    if not values:
        raise ValueError(f"At least one {label} is required.")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}: {values}")


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")
    return payload


def _checkpoint_fingerprint(path: Path) -> str:
    stat = path.stat()
    identity = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]


def load_catalog() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ValueError(f"Invalid model catalog: {CATALOG_PATH}")
    return payload["models"]


def validate_recipe(recipe: dict[str, Any]) -> None:
    errors: list[str] = []
    if recipe.get("protocol", {}).get("augmentation") != "none":
        errors.append("protocol.augmentation must be 'none'")
    try:
        validate_no_augmentation_config(recipe.get("data", {}))
    except ValueError as exc:
        errors.append(str(exc))
    legacy_batch_keys = sorted(
        _REMOVED_BATCH_AUGMENTATION_KEYS.intersection(recipe.get("train", {}))
    )
    if legacy_batch_keys:
        errors.append(
            "batch augmentation is forbidden: " + ", ".join(legacy_batch_keys)
        )
    if errors:
        raise ValueError("No-augmentation recipe validation failed:\n- " + "\n- ".join(errors))


def resolve_teacher(
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[Path, Path, dict[str, Any], str]:
    checkpoint_path = Path(args.teacher_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    checkpoint = _torch_load(checkpoint_path)
    embedded = checkpoint.get("config")
    if not isinstance(embedded, dict):
        raise ValueError(
            "Teacher checkpoint must contain its training config under key 'config'."
        )

    if args.teacher_config:
        supplied_path = Path(args.teacher_config).expanduser().resolve()
        if not supplied_path.is_file():
            raise FileNotFoundError(f"Teacher config not found: {supplied_path}")
        teacher_cfg = load_config(supplied_path)
        for section in ("protocol", "data", "train", "model"):
            if teacher_cfg.get(section) != embedded.get(section):
                raise ValueError(
                    f"Teacher config section {section!r} does not match checkpoint metadata."
                )
    else:
        teacher_cfg = deepcopy(embedded)

    errors: list[str] = []
    try:
        validate_no_augmentation_config(teacher_cfg.get("data", {}))
    except ValueError as exc:
        errors.append(str(exc))
    if teacher_cfg.get("protocol", {}).get("augmentation") != "none":
        errors.append("teacher protocol.augmentation must be 'none'")
    if teacher_cfg.get("protocol", {}).get("session") != "baseline":
        errors.append("teacher must be a baseline CE run")
    if teacher_cfg.get("protocol", {}).get("dataset") != args.dataset:
        errors.append("teacher dataset does not match --dataset")
    if teacher_cfg.get("model", {}).get("name") != TEACHER_MODEL_NAME:
        errors.append(f"teacher model must be {TEACHER_MODEL_NAME}")
    if int(teacher_cfg.get("train", {}).get("seed", -1)) != args.seed:
        errors.append("teacher seed does not match --seed")
    if float(teacher_cfg.get("train", {}).get("kd_alpha", -1.0)) != 0.0:
        errors.append("teacher must be trained with CE (kd_alpha=0)")
    if args.expected_teacher_epochs is not None and int(
        teacher_cfg.get("train", {}).get("epochs", -1)
    ) != int(args.expected_teacher_epochs):
        errors.append(
            "teacher epochs do not match --expected-teacher-epochs "
            f"({teacher_cfg.get('train', {}).get('epochs')} != {args.expected_teacher_epochs})"
        )
    if errors:
        raise ValueError("Invalid no-augmentation teacher:\n- " + "\n- ".join(errors))

    num_classes = 100 if args.dataset == "cifar100" else 10
    build_cfg = deepcopy(teacher_cfg)
    build_cfg.setdefault("model", {})["num_classes"] = num_classes
    teacher = build_model(build_cfg).eval()
    state = checkpoint.get("model", checkpoint)
    teacher.load_state_dict(state, strict=True)
    with torch.inference_mode():
        output = teacher(torch.randn(1, 3, 32, 32))
    if output.shape != (1, num_classes):
        raise ValueError(
            f"Teacher output shape must be (1, {num_classes}), got {tuple(output.shape)}"
        )
    del teacher, checkpoint

    fingerprint = _checkpoint_fingerprint(checkpoint_path)
    materialized_path = (
        output_root
        / "_teacher_configs"
        / f"{args.dataset}_resnet18_seed{args.seed}_{fingerprint}.yaml"
    )
    if not args.dry_run:
        save_config(teacher_cfg, materialized_path)
    return checkpoint_path, materialized_path, teacher_cfg, fingerprint


def _number_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def experiment_name(
    args: argparse.Namespace,
    student: str,
    method: str,
    epochs: int,
    teacher_fingerprint: str,
) -> str:
    if method == "standard":
        method_token = f"standard_kd_t{_number_token(args.temperature)}_a{_number_token(args.standard_alpha)}"
    else:
        method_token = (
            f"dkd_t{_number_token(args.temperature)}"
            f"_tckd{_number_token(args.dkd_tckd_weight)}"
            f"_nckd{_number_token(args.dkd_nckd_weight)}"
            f"_w{args.dkd_warmup_epochs}"
        )
    smoke_token = "_smoke" if args.smoke else ""
    return (
        f"{args.dataset}_noaug_{student}_seed{args.seed}_{method_token}"
        f"_e{epochs}_teacher{teacher_fingerprint}{smoke_token}"
    )


def make_student_config(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    model_entry: dict[str, Any],
    student: str,
    method: str,
    teacher_cfg: dict[str, Any],
    teacher_fingerprint: str,
) -> dict[str, Any]:
    epochs = 1 if args.smoke else int(args.epochs)
    name = experiment_name(args, student, method, epochs, teacher_fingerprint)
    label_smoothing = (
        float(recipe.get("train", {}).get("label_smoothing", 0.0))
        if args.label_smoothing is None
        else float(args.label_smoothing)
    )
    patch: dict[str, Any] = {
        "experiment": {
            "name": name,
            "model_key": student,
            "teacher_run": teacher_cfg.get("experiment", {}).get("name"),
            "teacher_fingerprint": teacher_fingerprint,
        },
        "protocol": {
            "name": f"{args.dataset}_hbcc_{method}_noaug_v1",
            "purpose": "hbcc_standard_kd_vs_dkd_no_augmentation",
            "session": "kd",
            "augmentation": "none",
            "canonical": False,
            "effective_epochs": epochs,
            "distillation_method": method,
        },
        "data": {
            "root": str(Path(args.data_root).expanduser().resolve()),
            "download": bool(args.download_data),
            "loader_seed": args.seed,
        },
        "train": {
            "seed": args.seed,
            "epochs": epochs,
            "kd_method": method,
            "kd_alpha": args.standard_alpha if method == "standard" else 0.0,
            "kd_temperature": args.temperature,
            "dkd_tckd_weight": args.dkd_tckd_weight,
            "dkd_nckd_weight": args.dkd_nckd_weight,
            "dkd_warmup_epochs": args.dkd_warmup_epochs,
            "label_smoothing": label_smoothing,
        },
        "model": deepcopy(model_entry["model"]),
    }
    if args.workers is not None:
        patch["data"]["workers"] = int(args.workers)
    cfg = deep_update(recipe, patch)
    cfg["model"]["num_classes"] = 100 if args.dataset == "cifar100" else 10
    if args.smoke:
        cfg = deep_update(
            cfg,
            {
                "data": {
                    "name": "fake",
                    "download": False,
                    "workers": 0,
                    "fake_train_size": 16,
                    "fake_val_size": 8,
                    "fake_test_size": 8,
                    "batch_size": 4,
                    "val_batch_size": 4,
                    "test_batch_size": 4,
                    "drop_last": False,
                }
            },
        )
    return cfg


def completed_run_matches(output_root: Path, expected: dict[str, Any]) -> bool:
    run_dir = output_root / expected["experiment"]["name"]
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "best.pth"
    metrics_path = run_dir / "test_metrics.json"
    if not all(path.is_file() for path in (config_path, checkpoint_path, metrics_path)):
        return False
    try:
        actual = load_config(config_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        all(
            actual.get(section) == expected.get(section)
            for section in ("experiment", "protocol", "data", "train", "model")
        )
        and actual.get("distillation", {}).get("method")
        == expected.get("train", {}).get("kd_method")
        and metrics.get("test_acc1") is not None
    )


def train_command(
    args: argparse.Namespace,
    config_path: Path,
    output_root: Path,
    teacher_config: Path,
    teacher_checkpoint: Path,
) -> list[str]:
    command = [
        args.python,
        str(ROOT / "tools" / "train.py"),
        "--config",
        str(config_path),
        "--output",
        str(output_root),
        "--teacher-config",
        str(teacher_config),
        "--teacher-checkpoint",
        str(teacher_checkpoint),
        "--device",
        args.device,
        "--print-every",
        str(args.print_every),
    ]
    if not args.progress:
        command.append("--no-progress")
    if args.smoke:
        command.extend(
            [
                "--limit-train-batches",
                "1",
                "--limit-val-batches",
                "1",
                "--limit-test-batches",
                "1",
            ]
        )
    return command


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_unique("student", args.students)
    _validate_unique("method", args.methods)
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.expected_teacher_epochs is not None and args.expected_teacher_epochs <= 0:
        raise ValueError("--expected-teacher-epochs must be positive.")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive.")
    if not 0.0 < args.standard_alpha <= 1.0:
        raise ValueError("--standard-alpha must be in (0, 1].")
    if args.dkd_tckd_weight < 0.0 or args.dkd_nckd_weight < 0.0:
        raise ValueError("DKD weights must be non-negative.")
    if args.dkd_tckd_weight == 0.0 and args.dkd_nckd_weight == 0.0:
        raise ValueError("DKD requires at least one positive weight.")
    if args.dkd_warmup_epochs < 0:
        raise ValueError("--dkd-warmup-epochs must be non-negative.")
    if args.label_smoothing is not None and not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1).")

    recipe = load_config(RECIPE_PATHS[args.dataset])
    validate_recipe(recipe)
    catalog = load_catalog()
    unknown = sorted(set(args.students) - set(catalog))
    if unknown:
        raise ValueError(f"Unknown students: {unknown}")
    invalid_students = [
        name
        for name in args.students
        if catalog[name].get("kd_student") is not True
        or catalog[name].get("model", {}).get("name") != "hbcc"
    ]
    if invalid_students:
        raise ValueError(f"Only catalogued HBCC KD students are allowed: {invalid_students}")

    num_classes = 100 if args.dataset == "cifar100" else 10
    for student in args.students:
        architecture_errors = validate_hbcc_pdf_cifar_config(
            student,
            catalog[student]["model"],
        )
        if architecture_errors:
            raise ValueError(
                "HBCC PDF architecture validation failed:\n- "
                + "\n- ".join(architecture_errors)
            )
        model_cfg = deepcopy(catalog[student]["model"])
        model_cfg["num_classes"] = num_classes
        model = build_model({"model": model_cfg}).eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 3, 32, 32))
        if output.shape != (1, num_classes):
            raise ValueError(
                f"{student} output must be (1, {num_classes}), got {tuple(output.shape)}"
            )
        del model

    output_root = Path(args.output).expanduser().resolve()
    teacher_checkpoint, teacher_config, teacher_cfg, teacher_fingerprint = resolve_teacher(
        args,
        output_root,
    )
    print(
        "KD comparison preflight: "
        f"dataset={args.dataset} seed={args.seed} augmentation=none "
        f"students={args.students} methods={args.methods} "
        f"teacher={teacher_checkpoint} fingerprint={teacher_fingerprint}",
        flush=True,
    )
    if args.validate_only:
        return

    for method in args.methods:
        for student in args.students:
            cfg = make_student_config(
                args,
                recipe,
                catalog[student],
                student,
                method,
                teacher_cfg,
                teacher_fingerprint,
            )
            name = cfg["experiment"]["name"]
            run_dir = output_root / name
            if args.skip_completed and not args.force and completed_run_matches(output_root, cfg):
                print(f"skip completed compatible run: {name}", flush=True)
                continue
            if run_dir.exists() and not args.force and not args.dry_run:
                raise FileExistsError(
                    f"Refusing to overwrite run directory: {run_dir}. Use --force to replace it."
                )
            config_path = output_root / "_effective_configs" / f"{name}.yaml"
            if not args.dry_run:
                save_config(cfg, config_path)
            command = train_command(
                args,
                config_path,
                output_root,
                teacher_config,
                teacher_checkpoint,
            )
            print("$ " + shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
