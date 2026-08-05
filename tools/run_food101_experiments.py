from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "tools" / "train.py"
EXPERIMENTS = {
    "hbcc": {
        "config": PROJECT_ROOT / "configs" / "food101" / "hbcc_2p5m_coc_recipe.yaml",
        "run_name": "food101_hbcc_2p5m_coc_recipe_seed42",
    },
    "resnet18": {
        "config": PROJECT_ROOT / "configs" / "food101" / "resnet18_scratch_coc_recipe.yaml",
        "run_name": "food101_resnet18_scratch_coc_recipe_seed42",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical from-scratch Food-101 architecture comparison."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=["hbcc", "resnet18"],
    )
    parser.add_argument("--output", default="runs/food101")
    parser.add_argument("--data-root", default="data/food101")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--limit-test-batches", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()

    rows: list[dict[str, Any]] = []
    for model_key in args.models:
        spec = EXPERIMENTS[model_key]
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            str(spec["config"]),
            "--output",
            str(output_root),
            "--device",
            args.device,
            "--print-every",
            "1",
            "--override",
            f"data.root={data_root.as_posix()}",
            "--override",
            f"train.epochs={args.epochs}",
        ]
        if args.no_progress:
            command.append("--no-progress")
        for flag, value in (
            ("--limit-train-batches", args.limit_train_batches),
            ("--limit-val-batches", args.limit_val_batches),
            ("--limit-test-batches", args.limit_test_batches),
        ):
            if value is not None:
                command.extend((flag, str(value)))
        print(f"\n=== Running {model_key}: {' '.join(command)} ===", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        run_dir = output_root / str(spec["run_name"])
        records = read_jsonl(run_dir / "metrics.jsonl")
        validation = [record for record in records if "val_acc1" in record]
        tests = [record for record in records if record.get("phase") == "test"]
        if not validation or len(tests) != 1:
            raise RuntimeError(f"Incomplete metrics in {run_dir}")
        best = max(validation, key=lambda record: record["val_acc1"])
        test = tests[0]
        rows.append(
            {
                "model": model_key,
                "best_epoch": int(best["epoch"]) + 1,
                "best_val_acc1": best["val_acc1"],
                "best_val_acc5": best.get("val_acc5"),
                "test_acc1": test["test_acc1"],
                "test_acc5": test.get("test_acc5"),
                "test_loss": test["test_loss"],
                "run_dir": str(run_dir),
            }
        )

    comparison_path = output_root / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n=== Final held-out test comparison ===")
    for row in sorted(rows, key=lambda item: item["test_acc1"], reverse=True):
        print(
            f"{row['model']:>8} | best epoch={row['best_epoch']:>3} "
            f"| val@1={row['best_val_acc1']:.2f} "
            f"| test@1={row['test_acc1']:.2f}"
        )
    print(f"Saved: {comparison_path}")


if __name__ == "__main__":
    main()
