from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lightweight_hbcc.config import deep_update, load_config
from lightweight_hbcc.models import build_model


MODEL_CONFIGS = {
    "resnet18": "resnet18.yaml",
    "mobilenet_v2": "mobilenet_v2.yaml",
    "shufflenet_v2_x1_0": "shufflenet_v2_x1_0.yaml",
    "coc_baseline": "coc_baseline.yaml",
    "hbcc_small": "hbcc_small.yaml",
    "hbcc_small_plus": "hbcc_small_plus.yaml",
    "phbcc_2m": "phbcc_2m.yaml",
    "hbcc_medium": "hbcc_medium.yaml",
}

BASELINE_MODELS = (
    "resnet18",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "coc_baseline",
)
HBCC_MODELS = (
    "hbcc_small",
    "hbcc_medium",
)
CORE_MODELS = (*BASELINE_MODELS, *HBCC_MODELS)
DEFAULT_SEEDS = (17,)
RECIPE_PATH = Path("configs/recipes/cifar_coc_paper_inspired.yaml")
_CANONICAL_RECIPE = load_config(RECIPE_PATH)
CANONICAL_PROTOCOL_NAME = str(_CANONICAL_RECIPE["protocol"]["name"])
CANONICAL_EPOCHS = int(_CANONICAL_RECIPE["protocol"]["effective_epochs"])


def validate_seeds(seeds: list[int]) -> None:
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds would overwrite runs: {seeds}")


def validate_models(models: list[str]) -> None:
    if len(models) != len(set(models)):
        raise ValueError(f"Duplicate models would overwrite runs: {models}")


def run_suffix(args: argparse.Namespace) -> str:
    if args.smoke:
        return "_smoke"
    if args.epochs is not None and args.epochs != CANONICAL_EPOCHS:
        return f"_e{args.epochs}"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the original HBCC-Small/Medium and report baselines with one shared "
            "paper-inspired CIFAR recipe and one shared seed."
        )
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=list(CORE_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs_fair_paper_inspired_200e")
    parser.add_argument("--benchmark-output", default="results/fair_paper_inspired_200e")
    parser.add_argument("--epochs", type=int, help="Apply the same epoch override to every selected model and seed.")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing run directory.")
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse only fully completed, metadata-compatible runs (default: enabled).",
    )
    return parser.parse_args()


def config_paths(dataset: str, models: list[str]) -> dict[str, Path]:
    root = Path("configs") / "fair_comparison" / dataset
    return {name: root / MODEL_CONFIGS[name] for name in models}


def _recipe_for_dataset(dataset: str) -> dict[str, Any]:
    recipe = load_config(RECIPE_PATH)
    if dataset == "cifar100":
        recipe = deep_update(recipe, {"data": {"name": "cifar100"}})
    return recipe


def validate_controlled_configs(dataset: str, paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Fail fast if any comparison config drifts from the shared recipe."""

    expected = _recipe_for_dataset(dataset)
    loaded: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, path in paths.items():
        cfg = load_config(path)
        loaded[name] = cfg
        if cfg.get("data") != expected.get("data"):
            errors.append(f"{name}: data block differs from controlled recipe")
        if cfg.get("train") != expected.get("train"):
            errors.append(f"{name}: train block differs from controlled recipe")
        if cfg.get("protocol") != expected.get("protocol"):
            errors.append(f"{name}: protocol metadata differs from controlled recipe")
        expected_classes = 100 if dataset == "cifar100" else 10
        if cfg.get("model", {}).get("num_classes") != expected_classes:
            errors.append(f"{name}: model.num_classes must be {expected_classes}")
        if int(cfg.get("data", {}).get("workers", 0)) <= 0:
            errors.append(f"{name}: controlled stochastic augmentation requires data.workers > 0")
        try:
            model = build_model(cfg).eval()
            with torch.inference_mode():
                output = model(torch.randn(1, 3, 32, 32))
            if output.shape != (1, expected_classes):
                errors.append(f"{name}: expected output (1, {expected_classes}), got {tuple(output.shape)}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: model build/forward failed: {exc!r}")
    if errors:
        raise ValueError("Fair-comparison preflight failed:\n- " + "\n- ".join(errors))
    return loaded


def _run(command: list[str], dry_run: bool) -> None:
    print("$ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def ensure_run_target(output: str, experiment_name: str, force: bool, dry_run: bool) -> None:
    run_dir = Path(output) / experiment_name
    if run_dir.exists() and not force and not dry_run:
        raise FileExistsError(
            f"Refusing to overwrite controlled run: {run_dir}. "
            "Use a new seed/output or pass --force intentionally."
        )


def effective_epochs(args: argparse.Namespace) -> int:
    if args.smoke:
        return 1
    return CANONICAL_EPOCHS if args.epochs is None else int(args.epochs)


def expected_protocol_name(args: argparse.Namespace) -> str:
    epochs = effective_epochs(args)
    if not args.smoke and epochs == CANONICAL_EPOCHS:
        return CANONICAL_PROTOCOL_NAME
    suffix = "smoke" if args.smoke else f"epochs{epochs}"
    return f"{CANONICAL_PROTOCOL_NAME}_{suffix}"


def completed_run_matches(args: argparse.Namespace, experiment_name: str, seed: int) -> bool:
    run_dir = Path(args.output) / experiment_name
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "best.pth"
    metrics_path = run_dir / "test_metrics.json"
    if not all(path.is_file() for path in (config_path, checkpoint_path, metrics_path)):
        return False
    try:
        cfg = load_config(config_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    epochs = effective_epochs(args)
    expected_canonical = not args.smoke and epochs == CANONICAL_EPOCHS
    expected_recipe = _recipe_for_dataset(args.dataset)
    expected_recipe = deep_update(
        expected_recipe,
        {
            "data": {
                "root": str(args.data_root),
                "loader_seed": int(seed),
            },
            "train": {
                "seed": int(seed),
                "epochs": epochs,
            },
            "protocol": {
                "name": expected_protocol_name(args),
                "canonical": expected_canonical,
                "effective_epochs": epochs,
            },
        },
    )
    return (
        cfg.get("experiment", {}).get("name") == experiment_name
        and cfg.get("data") == expected_recipe.get("data")
        and cfg.get("train") == expected_recipe.get("train")
        and cfg.get("protocol") == expected_recipe.get("protocol")
        and metrics.get("test_acc1") is not None
    )


def completed_benchmark_matches(args: argparse.Namespace, experiment_name: str) -> bool:
    path = Path(args.benchmark_output) / args.dataset / f"{experiment_name}.json"
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        record.get("config_id") == experiment_name
        and record.get("protocol_name") == expected_protocol_name(args)
        and int(record.get("effective_epochs", -1)) == effective_epochs(args)
        and record.get("latency_ms_b1") is not None
    )


def _overrides(args: argparse.Namespace, experiment_name: str, seed: int) -> list[str]:
    values = [
        f"data.root={args.data_root}",
        f"data.loader_seed={seed}",
        f"train.seed={seed}",
        f"experiment.name={experiment_name}",
    ]
    requested_epochs = 1 if args.smoke else args.epochs
    if requested_epochs is not None:
        values.extend(
            [
                f"train.epochs={requested_epochs}",
                f"protocol.effective_epochs={requested_epochs}",
            ]
        )
        if requested_epochs != CANONICAL_EPOCHS:
            values.extend(
                [
                    f"protocol.name={expected_protocol_name(args)}",
                    "protocol.canonical=false",
                ]
            )
    return values


def _train_command(
    args: argparse.Namespace,
    config_path: Path,
    experiment_name: str,
    seed: int,
) -> list[str]:
    command = [
        args.python,
        "tools/train.py",
        "--config",
        str(config_path),
        "--output",
        args.output,
        "--print-every",
        str(args.print_every),
    ]
    if not args.progress:
        command.append("--no-progress")
    for value in _overrides(args, experiment_name, seed):
        command.extend(["--override", value])
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


def _benchmark_command(
    args: argparse.Namespace,
    config_path: Path,
    experiment_name: str,
    seed: int,
) -> list[str]:
    command = [
        args.python,
        "tools/benchmark.py",
        "--config",
        str(config_path),
        "--checkpoint",
        str(Path(args.output) / experiment_name / "best.pth"),
        "--output",
        str(Path(args.benchmark_output) / args.dataset),
    ]
    for value in _overrides(args, experiment_name, seed):
        command.extend(["--override", value])
    if args.smoke:
        command.extend(["--batch-sizes", "1", "4", "--warmup", "1", "--runs", "2"])
    return command


def main() -> None:
    args = parse_args()
    validate_seeds(args.seeds)
    validate_models(args.models)
    paths = config_paths(args.dataset, args.models)
    configs = validate_controlled_configs(args.dataset, paths)
    recipe_name = next(iter(configs.values()))["protocol"]["name"]
    print(
        f"fairness preflight: protocol={recipe_name} dataset={args.dataset} "
        f"models={len(paths)} seeds={args.seeds} training_runs={len(paths) * len(args.seeds)}",
        flush=True,
    )
    if args.validate_only:
        return

    for seed in args.seeds:
        for model_name, config_path in paths.items():
            base_name = configs[model_name]["experiment"]["name"]
            experiment_name = f"{base_name}_seed{seed}{run_suffix(args)}"
            if (
                args.skip_completed
                and not args.force
                and not args.dry_run
                and completed_run_matches(args, experiment_name, seed)
            ):
                print(f"skip completed compatible run: {experiment_name}", flush=True)
                continue
            ensure_run_target(args.output, experiment_name, args.force, args.dry_run)
            _run(_train_command(args, config_path, experiment_name, seed), args.dry_run)

    if args.benchmark:
        benchmark_seed = args.seeds[0]
        print(f"benchmark phase: one checkpoint per architecture, seed={benchmark_seed}", flush=True)
        for model_name, config_path in paths.items():
            base_name = configs[model_name]["experiment"]["name"]
            experiment_name = f"{base_name}_seed{benchmark_seed}{run_suffix(args)}"
            if (
                args.skip_completed
                and not args.force
                and not args.dry_run
                and completed_benchmark_matches(args, experiment_name)
            ):
                print(f"skip completed benchmark: {experiment_name}", flush=True)
                continue
            _run(_benchmark_command(args, config_path, experiment_name, benchmark_seed), args.dry_run)


if __name__ == "__main__":
    main()
