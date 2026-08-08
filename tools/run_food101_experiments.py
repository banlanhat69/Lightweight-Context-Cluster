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
        "config": PROJECT_ROOT / "configs" / "food101" / "hbcc_best100.yaml",
        "run_name": "food101_hbcc_2p5m_2x100_seed42",
        "mix_cooldown_epochs": 15,
    },
    "resnet18": {
        "config": PROJECT_ROOT / "configs" / "food101" / "resnet18_best100.yaml",
        "run_name": "food101_resnet18_scratch_2x100_seed42",
        "mix_cooldown_epochs": 10,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable Food-101 architecture comparison."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=["hbcc", "resnet18"],
    )
    parser.add_argument("--output", default="runs/food101")
    parser.add_argument("--data-root", default="data/food101")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--run-until-epoch",
        type=int,
        help="Pause at this total epoch while keeping --epochs as the scheduler horizon.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume-hbcc",
        help="Path to HBCC latest.pth from an interrupted run.",
    )
    parser.add_argument(
        "--resume-resnet18",
        help="Path to ResNet-18 latest.pth from an interrupted run.",
    )
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
    run_until_epoch = args.run_until_epoch or args.epochs
    if not 1 <= run_until_epoch <= args.epochs:
        raise ValueError("--run-until-epoch must be between 1 and --epochs.")
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()

    rows: list[dict[str, Any]] = []
    resume_paths = {
        "hbcc": args.resume_hbcc,
        "resnet18": args.resume_resnet18,
    }
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
            "--run-until-epoch",
            str(run_until_epoch),
            "--print-every",
            "1",
            "--override",
            f"data.root={data_root.as_posix()}",
            "--override",
            f"train.epochs={args.epochs}",
            "--override",
            "train.mixup_cutmix_off_epoch="
            f"{max(0, args.epochs - int(spec['mix_cooldown_epochs']))}",
        ]
        if args.no_progress:
            command.append("--no-progress")
        if resume_paths[model_key]:
            resume_path = Path(resume_paths[model_key]).expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(
                    f"Resume checkpoint for {model_key} not found: {resume_path}"
                )
            command.extend(("--resume", str(resume_path)))
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
        with (run_dir / "setup.json").open("r", encoding="utf-8") as handle:
            setup = json.load(handle)
        if not validation:
            raise RuntimeError(f"No validation metrics in {run_dir}")
        best = max(validation, key=lambda record: record["val_acc1"])
        complete = run_until_epoch == args.epochs
        if complete and len(tests) != 1:
            raise RuntimeError(f"Final test metrics are incomplete in {run_dir}")
        if not complete and tests:
            raise RuntimeError(f"A paused run must not evaluate the test split: {run_dir}")
        test = tests[0] if tests else None
        best_epoch = int(test.get("epoch", best["epoch"])) if test else int(best["epoch"])
        best_val_acc1 = (
            float(test.get("best_val_acc1", best["val_acc1"]))
            if test
            else float(best["val_acc1"])
        )
        best_record = next(
            (
                record
                for record in validation
                if int(record.get("epoch", -1)) == best_epoch
            ),
            None,
        )
        rows.append(
            {
                "model": model_key,
                "optimizer": setup["optimizer"],
                "parameters": setup["parameter_count"],
                "status": "complete" if complete else "paused",
                "completed_epochs": run_until_epoch,
                "best_epoch": best_epoch + 1,
                "best_val_acc1": best_val_acc1,
                "best_val_acc5": (
                    best_record.get("val_acc5") if best_record is not None else None
                ),
                "test_acc1": test["test_acc1"] if test else None,
                "test_acc5": test.get("test_acc5") if test else None,
                "test_loss": test["test_loss"] if test else None,
                "run_dir": str(run_dir),
            }
        )

    summary_path = output_root / (
        "comparison.csv" if run_until_epoch == args.epochs else "progress.csv"
    )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if run_until_epoch == args.epochs:
        print("\n=== Final held-out test comparison ===")
        for row in sorted(rows, key=lambda item: item["test_acc1"], reverse=True):
            print(
                f"{row['model']:>8} | best epoch={row['best_epoch']:>3} "
                f"| val@1={row['best_val_acc1']:.2f} "
                f"| test@1={row['test_acc1']:.2f}"
            )
    else:
        print(f"\n=== Paused after {run_until_epoch}/{args.epochs} epochs ===")
        for row in rows:
            print(
                f"{row['model']:>8} | best epoch={row['best_epoch']:>3} "
                f"| best val@1={row['best_val_acc1']:.2f}"
            )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
