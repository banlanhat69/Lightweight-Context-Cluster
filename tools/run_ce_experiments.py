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
from lightweight_hbcc.models.hbcc import validate_hbcc_wide_cifar_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "cifar_fair"
CATALOG_PATH = CONFIG_ROOT / "model_catalog.yaml"
RECIPE_PATHS = {
    "cifar10": CONFIG_ROOT / "cifar10_no_augmentation.yaml",
    "cifar100": CONFIG_ROOT / "cifar100_no_augmentation.yaml",
}
DEFAULT_SEEDS = (42,)
HBCC_MODELS = ("hbcc_small", "hbcc_medium")
_REMOVED_BATCH_AUGMENTATION_KEYS = {"mixup_alpha", "cutmix_alpha", "cutmix_prob"}


def load_model_catalog() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ValueError(f"Invalid model catalog: {CATALOG_PATH}")
    return payload["models"]


def ce_models(catalog: dict[str, dict[str, Any]] | None = None) -> tuple[str, ...]:
    return tuple((catalog or load_model_catalog()).keys())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    catalog = load_model_catalog()
    parser = argparse.ArgumentParser(
        description=(
            "Train selected CIFAR architecture-comparison models with CE only "
            "and no data augmentation."
        )
    )
    parser.add_argument("--dataset", choices=sorted(RECIPE_PATHS), default="cifar10")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(catalog),
        default=list(ce_models(catalog)),
        help="CE models to train. Omit to train every catalogued model.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs_ce_hbcc_wide")
    parser.add_argument("--epochs", type=int)
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
    if protocol.get("session") != "baseline":
        errors.append("protocol.session must be 'baseline' for CE training")
    if int(protocol.get("ce_epochs", -1)) != 300:
        errors.append("protocol.ce_epochs must be 300")
    try:
        validate_no_augmentation_config(recipe.get("data", {}))
        _assert_no_batch_augmentation(recipe.get("train", {}))
    except ValueError as exc:
        errors.append(str(exc))
    if float(recipe.get("train", {}).get("kd_alpha", 0.0)) != 0.0:
        errors.append("the CE recipe must not enable knowledge distillation")

    num_classes = 100 if dataset == "cifar100" else 10
    for name in selected:
        entry = catalog[name]
        model_cfg = deepcopy(entry["model"])
        model_cfg["num_classes"] = num_classes
        if name in HBCC_MODELS:
            errors.extend(validate_hbcc_wide_cifar_config(name, entry["model"]))
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
        raise ValueError("CE preflight failed:\n- " + "\n- ".join(errors))
    return recipe, catalog


def effective_epochs(
    recipe: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    if args.smoke:
        return 1
    if args.epochs is not None:
        if int(args.epochs) <= 0:
            raise ValueError("epochs must be positive")
        return int(args.epochs)
    return int(recipe["protocol"]["ce_epochs"])


def run_suffix(
    recipe: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    if args.smoke:
        return "_smoke"
    default_epochs = int(recipe["protocol"]["ce_epochs"])
    if args.epochs is not None and args.epochs != default_epochs:
        return f"_e{args.epochs}"
    return ""


def experiment_name(
    dataset: str,
    model_name: str,
    seed: int,
    suffix: str = "",
) -> str:
    return f"{dataset}_noaug_{model_name}_seed{seed}_ce{suffix}"


def make_effective_config(
    recipe: dict[str, Any],
    model_entry: dict[str, Any],
    model_name: str,
    seed: int,
    epochs: int,
    data_root: str,
    name: str,
    smoke: bool = False,
) -> dict[str, Any]:
    expected_epochs = int(recipe["protocol"]["ce_epochs"])
    cfg = deep_update(
        recipe,
        {
            "experiment": {
                "name": name,
                "model_key": model_name,
            },
            "protocol": {
                "name": f"{recipe['protocol']['name']}_ce",
                "session": "baseline",
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
                "kd_method": "none",
            },
            "model": deepcopy(model_entry["model"]),
        },
    )
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
    distillation = actual.get("distillation", {})
    distillation_matches = (
        bool(distillation.get("enabled")) is False
        and distillation.get("method") == "none"
        and float(distillation.get("alpha", 0.0)) == 0.0
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
    _validate_unique("models", args.models)
    _validate_unique("seeds", args.seeds)
    recipe, catalog = validate_protocol(args.dataset, args.models)

    output_root = _output_root(args.output)
    suffix = run_suffix(recipe, args)
    epochs = effective_epochs(recipe, args)
    print(
        "CE preflight: "
        f"dataset={args.dataset} augmentation=none models={args.models} "
        f"seeds={args.seeds} epochs={epochs}",
        flush=True,
    )
    if args.validate_only:
        return

    for seed in args.seeds:
        for model_name in args.models:
            name = experiment_name(
                args.dataset,
                model_name,
                seed,
                suffix,
            )
            cfg = make_effective_config(
                recipe,
                catalog[model_name],
                model_name,
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


if __name__ == "__main__":
    main()
