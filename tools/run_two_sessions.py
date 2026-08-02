from __future__ import annotations

import argparse
from copy import deepcopy
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
SESSION_ORDER = ("baseline", "kd")
TEACHER_MODEL = "resnet18"
DEFAULT_SEEDS = (42,)
KD_ALPHA = 0.5
KD_TEMPERATURE = 4.0
_REMOVED_BATCH_AUGMENTATION_KEYS = {"mixup_alpha", "cutmix_alpha", "cutmix_prob"}


def load_model_catalog() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ValueError(f"Invalid model catalog: {CATALOG_PATH}")
    return payload["models"]


def baseline_models(catalog: dict[str, dict[str, Any]] | None = None) -> tuple[str, ...]:
    return tuple((catalog or load_model_catalog()).keys())


def kd_models(catalog: dict[str, dict[str, Any]] | None = None) -> tuple[str, ...]:
    entries = catalog or load_model_catalog()
    return tuple(name for name, entry in entries.items() if entry.get("kd_student") is True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    catalog = load_model_catalog()
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly two no-augmentation CIFAR sessions: CE for all models, "
            "then HBCC knowledge distillation from the matching ResNet-18 teacher."
        )
    )
    parser.add_argument("--dataset", choices=sorted(RECIPE_PATHS), default="cifar10")
    parser.add_argument(
        "--sessions",
        nargs="+",
        choices=SESSION_ORDER,
        default=list(SESSION_ORDER),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(catalog),
        default=list(baseline_models(catalog)),
        help="Baseline models to run; the KD session uses only selected HBCC students.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs_two_sessions_hbcc_pdf")
    parser.add_argument("--baseline-epochs", type=int)
    parser.add_argument("--kd-epochs", type=int)
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
        help="Reuse only complete runs whose effective config still matches.",
    )
    return parser.parse_args(argv)


def _validate_unique(label: str, values: list[Any]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} would overwrite runs: {values}")


def _assert_no_batch_augmentation(train_cfg: dict[str, Any]) -> None:
    legacy = sorted(_REMOVED_BATCH_AUGMENTATION_KEYS.intersection(train_cfg))
    if legacy:
        raise ValueError(f"Legacy batch augmentation is not allowed: {', '.join(legacy)}")


def validate_protocol(
    dataset: str,
    selected_models: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    recipe = load_config(RECIPE_PATHS[dataset])
    catalog = load_model_catalog()
    selected = selected_models or list(catalog)
    errors: list[str] = []

    protocol = recipe.get("protocol", {})
    if protocol.get("augmentation") != "none":
        errors.append("protocol.augmentation must be 'none'")
    if int(protocol.get("baseline_epochs", -1)) != 300:
        errors.append("protocol.baseline_epochs must be 300")
    if int(protocol.get("kd_epochs", -1)) != 250:
        errors.append("protocol.kd_epochs must be 250")
    if protocol.get("teacher") != TEACHER_MODEL:
        errors.append(f"protocol.teacher must be {TEACHER_MODEL}")
    try:
        validate_no_augmentation_config(recipe.get("data", {}))
        _assert_no_batch_augmentation(recipe.get("train", {}))
    except ValueError as exc:
        errors.append(str(exc))
    if float(recipe.get("train", {}).get("kd_alpha", -1.0)) != 0.0:
        errors.append("the baseline recipe must set train.kd_alpha=0")
    if float(recipe.get("train", {}).get("kd_temperature", -1.0)) != KD_TEMPERATURE:
        errors.append(f"train.kd_temperature must be {KD_TEMPERATURE}")

    teacher_entry = catalog.get(TEACHER_MODEL, {})
    if teacher_entry.get("teacher") is not True:
        errors.append("ResNet-18 must be marked as the teacher")
    if teacher_entry.get("model", {}).get("name") != "resnet18_cifar":
        errors.append("the teacher architecture must be resnet18_cifar")
    if not kd_models(catalog):
        errors.append("at least one HBCC KD student is required")

    num_classes = 100 if dataset == "cifar100" else 10
    for name in selected:
        entry = catalog[name]
        model_cfg = deepcopy(entry["model"])
        model_cfg["num_classes"] = num_classes
        if entry.get("kd_student") and model_cfg.get("name") != "hbcc":
            errors.append(f"{name}: KD students must use the HBCC architecture")
        if entry.get("kd_student"):
            errors.extend(validate_hbcc_pdf_cifar_config(name, entry["model"]))
        try:
            model = build_model({"model": model_cfg}).eval()
            with torch.inference_mode():
                output = model(torch.randn(1, 3, 32, 32))
            if output.shape != (1, num_classes):
                errors.append(
                    f"{name}: expected output (1, {num_classes}), got {tuple(output.shape)}"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: model build/forward failed: {exc!r}")
    if errors:
        raise ValueError("Two-session preflight failed:\n- " + "\n- ".join(errors))
    return recipe, catalog


def effective_epochs(
    recipe: dict[str, Any],
    session: str,
    args: argparse.Namespace,
) -> int:
    if args.smoke:
        return 1
    override = args.baseline_epochs if session == "baseline" else args.kd_epochs
    if override is not None:
        if int(override) <= 0:
            raise ValueError(f"{session} epochs must be positive")
        return int(override)
    key = "baseline_epochs" if session == "baseline" else "kd_epochs"
    return int(recipe["protocol"][key])


def run_suffix(
    recipe: dict[str, Any],
    args: argparse.Namespace,
    session: str,
) -> str:
    if args.smoke:
        return "_smoke"
    baseline_default = int(recipe["protocol"]["baseline_epochs"])
    kd_default = int(recipe["protocol"]["kd_epochs"])
    parts: list[str] = []
    baseline_changed = (
        args.baseline_epochs is not None and args.baseline_epochs != baseline_default
    )
    kd_changed = args.kd_epochs is not None and args.kd_epochs != kd_default
    if session == "baseline" and baseline_changed:
        parts.append(f"e{args.baseline_epochs}")
    if session == "kd":
        if baseline_changed:
            parts.append(f"teacherce{args.baseline_epochs}")
        if kd_changed:
            parts.append(f"e{args.kd_epochs}")
    return "_" + "_".join(parts) if parts else ""


def experiment_name(
    dataset: str,
    model_name: str,
    seed: int,
    session: str,
    suffix: str = "",
) -> str:
    loss_name = "ce" if session == "baseline" else "kd"
    return f"{dataset}_noaug_{model_name}_seed{seed}_{loss_name}{suffix}"


def make_effective_config(
    recipe: dict[str, Any],
    model_entry: dict[str, Any],
    model_name: str,
    session: str,
    seed: int,
    epochs: int,
    data_root: str,
    name: str,
    smoke: bool = False,
    teacher_run: str | None = None,
) -> dict[str, Any]:
    baseline_epochs = int(recipe["protocol"]["baseline_epochs"])
    kd_epochs = int(recipe["protocol"]["kd_epochs"])
    expected_epochs = baseline_epochs if session == "baseline" else kd_epochs
    cfg = deep_update(
        recipe,
        {
            "experiment": {
                "name": name,
                "model_key": model_name,
                "teacher_run": teacher_run,
            },
            "protocol": {
                "name": f"{recipe['protocol']['name']}_{session}",
                "session": session,
                "canonical": epochs == expected_epochs,
                "effective_epochs": epochs,
            },
            "data": {
                "root": data_root,
                "loader_seed": seed,
            },
            "train": {
                "seed": seed,
                "epochs": epochs,
                "kd_method": "reverse_kd" if session == "kd" else "none",
                "kd_alpha": KD_ALPHA if session == "kd" else 0.0,
                "kd_temperature": KD_TEMPERATURE,
            },
            "model": deepcopy(model_entry["model"]),
        },
    )
    if teacher_run is None:
        cfg["experiment"].pop("teacher_run", None)
    num_classes = 100 if recipe["protocol"]["dataset"] == "cifar100" else 10
    cfg["model"]["num_classes"] = num_classes
    if smoke:
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


def _output_root(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def effective_config_path(output_root: Path, name: str) -> Path:
    return output_root / "_effective_configs" / f"{name}.yaml"


def completed_run_matches(output_root: Path, expected: dict[str, Any]) -> bool:
    name = expected["experiment"]["name"]
    run_dir = output_root / name
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
    fields_match = all(
        actual.get(key) == expected.get(key)
        for key in ("experiment", "protocol", "data", "train", "model")
    )
    is_kd = expected["protocol"]["session"] == "kd"
    distillation = actual.get("distillation", {})
    distillation_matches = (
        bool(distillation.get("enabled")) is is_kd
        and distillation.get("method") == ("reverse_kd" if is_kd else "none")
        and float(distillation.get("alpha", 0.0))
        == (KD_ALPHA if is_kd else 0.0)
        and float(distillation.get("temperature", KD_TEMPERATURE)) == KD_TEMPERATURE
    )
    if is_kd:
        expected_teacher_run = expected.get("experiment", {}).get("teacher_run")
        actual_teacher_checkpoint = distillation.get("teacher_checkpoint")
        distillation_matches = distillation_matches and (
            distillation.get("teacher_model") == "resnet18_cifar"
            and distillation.get("kl_direction") == "student||teacher"
            and isinstance(actual_teacher_checkpoint, str)
            and Path(actual_teacher_checkpoint).parent.name == expected_teacher_run
        )
    return fields_match and distillation_matches and metrics.get("test_acc1") is not None


def ensure_run_target(
    output_root: Path,
    expected: dict[str, Any],
    force: bool,
    skip_completed: bool,
    dry_run: bool,
) -> bool:
    name = expected["experiment"]["name"]
    run_dir = output_root / name
    if skip_completed and not force and completed_run_matches(output_root, expected):
        print(f"skip completed compatible run: {name}", flush=True)
        return False
    if run_dir.exists() and not force and not dry_run:
        raise FileExistsError(
            f"Refusing to overwrite run directory: {run_dir}. "
            "Use --force or choose a different output path."
        )
    return True


def train_command(
    args: argparse.Namespace,
    config_path: Path,
    output_root: Path,
    teacher_config: Path | None = None,
    teacher_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        args.python,
        str(ROOT / "tools" / "train.py"),
        "--config",
        str(config_path),
        "--output",
        str(output_root),
        "--print-every",
        str(args.print_every),
    ]
    if not args.progress:
        command.append("--no-progress")
    if teacher_config is not None and teacher_checkpoint is not None:
        command.extend(
            [
                "--teacher-config",
                str(teacher_config),
                "--teacher-checkpoint",
                str(teacher_checkpoint),
            ]
        )
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


def _run(command: list[str], dry_run: bool) -> None:
    print("$ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _prepare_config(
    output_root: Path,
    cfg: dict[str, Any],
    dry_run: bool,
) -> Path:
    path = effective_config_path(output_root, cfg["experiment"]["name"])
    if not dry_run:
        save_config(cfg, path)
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_unique("sessions", args.sessions)
    _validate_unique("models", args.models)
    _validate_unique("seeds", args.seeds)
    recipe, catalog = validate_protocol(args.dataset, args.models)
    selected_kd_models = [name for name in args.models if name in kd_models(catalog)]
    if "kd" in args.sessions and not selected_kd_models:
        raise ValueError("The KD session requires at least one selected HBCC student.")

    output_root = _output_root(args.output)
    baseline_suffix = run_suffix(recipe, args, "baseline")
    kd_suffix = run_suffix(recipe, args, "kd")
    print(
        "two-session preflight: "
        f"dataset={args.dataset} augmentation=none sessions={args.sessions} "
        f"baseline_models={args.models} kd_models={selected_kd_models} seeds={args.seeds}",
        flush=True,
    )
    if args.validate_only:
        return

    for seed in args.seeds:
        if "baseline" in args.sessions:
            epochs = effective_epochs(recipe, "baseline", args)
            for model_name in args.models:
                name = experiment_name(
                    args.dataset,
                    model_name,
                    seed,
                    "baseline",
                    baseline_suffix,
                )
                cfg = make_effective_config(
                    recipe,
                    catalog[model_name],
                    model_name,
                    "baseline",
                    seed,
                    epochs,
                    args.data_root,
                    name,
                    smoke=args.smoke,
                )
                config_path = _prepare_config(output_root, cfg, args.dry_run)
                if ensure_run_target(
                    output_root,
                    cfg,
                    args.force,
                    args.skip_completed,
                    args.dry_run,
                ):
                    _run(train_command(args, config_path, output_root), args.dry_run)

        if "kd" in args.sessions:
            teacher_name = experiment_name(
                args.dataset,
                TEACHER_MODEL,
                seed,
                "baseline",
                baseline_suffix,
            )
            teacher_cfg = make_effective_config(
                recipe,
                catalog[TEACHER_MODEL],
                TEACHER_MODEL,
                "baseline",
                seed,
                effective_epochs(recipe, "baseline", args),
                args.data_root,
                teacher_name,
                smoke=args.smoke,
            )
            teacher_config_path = _prepare_config(output_root, teacher_cfg, args.dry_run)
            teacher_checkpoint = output_root / teacher_name / "best.pth"
            teacher_will_run = "baseline" in args.sessions and TEACHER_MODEL in args.models
            if not args.dry_run and not teacher_checkpoint.is_file() and not teacher_will_run:
                raise FileNotFoundError(
                    "KD requires the matching no-augmentation ResNet-18 checkpoint: "
                    f"{teacher_checkpoint}"
                )
            if not args.dry_run and not teacher_checkpoint.is_file():
                raise FileNotFoundError(
                    "The scheduled ResNet-18 teacher run did not produce: "
                    f"{teacher_checkpoint}"
                )

            epochs = effective_epochs(recipe, "kd", args)
            for model_name in selected_kd_models:
                name = experiment_name(
                    args.dataset,
                    model_name,
                    seed,
                    "kd",
                    kd_suffix,
                )
                cfg = make_effective_config(
                    recipe,
                    catalog[model_name],
                    model_name,
                    "kd",
                    seed,
                    epochs,
                    args.data_root,
                    name,
                    smoke=args.smoke,
                    teacher_run=teacher_name,
                )
                config_path = _prepare_config(output_root, cfg, args.dry_run)
                if ensure_run_target(
                    output_root,
                    cfg,
                    args.force,
                    args.skip_completed,
                    args.dry_run,
                ):
                    _run(
                        train_command(
                            args,
                            config_path,
                            output_root,
                            teacher_config_path,
                            teacher_checkpoint,
                        ),
                        args.dry_run,
                    )


if __name__ == "__main__":
    main()
