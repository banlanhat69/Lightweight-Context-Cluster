from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lightweight_hbcc.config import apply_overrides, load_config
from lightweight_hbcc.engine import load_checkpoint, resolve_device
from lightweight_hbcc.models import build_model
from lightweight_hbcc.profiler import append_record_csv, benchmark_model, model_static_metrics, profile_operators, write_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark latency, throughput, memory, FLOPs and operator profile.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="results/benchmark")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 64, 128])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    device = resolve_device(args.device)
    model = build_model(cfg).to(device)
    image_size = int(cfg.get("data", {}).get("image_size", 32))
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device, strict=False)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = cfg.get("experiment", {}).get("name", Path(args.config).stem)
    record = {
        "model_name": cfg.get("model", {}).get("name", cfg.get("model", {}).get("type")),
        "config_id": name,
        "protocol_name": cfg.get("protocol", {}).get("name"),
        "protocol_canonical": cfg.get("protocol", {}).get("canonical"),
        "effective_epochs": cfg.get("protocol", {}).get("effective_epochs", cfg.get("train", {}).get("epochs")),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    record["image_size"] = image_size
    record.update(model_static_metrics(model, device, image_size=image_size))
    record.update(benchmark_model(model, device, args.batch_sizes, image_size=image_size, warmup=args.warmup, runs=args.runs))
    if args.profile:
        profile_path = output_dir / f"{name}_profile.txt"
        profile_operators(model, device, batch_size=1, image_size=image_size, out_path=profile_path)
        record["operator_profile"] = str(profile_path)
    out_path = output_dir / f"{name}.json"
    write_record(record, out_path)
    append_record_csv(record, output_dir / "summary.csv")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
