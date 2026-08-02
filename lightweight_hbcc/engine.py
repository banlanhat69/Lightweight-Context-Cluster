from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .losses import DistillationLoss
from .metrics import AverageMeter, accuracy


PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar:12}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
PROGRESS_NCOLS = 96


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed training RNGs and configure deterministic execution when requested."""

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device, strict: bool = True) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=strict)
    return ckpt


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: DistillationLoss,
    epoch: int,
    teacher: nn.Module | None = None,
    amp: bool = True,
    scaler: GradScaler | None = None,
    limit_batches: int | None = None,
    progress: bool = True,
) -> dict[str, float]:
    criterion.set_epoch(epoch)
    model.train()
    if teacher is not None:
        teacher.eval()
    if scaler is None:
        scaler = GradScaler(device.type, enabled=amp and device.type == "cuda")
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.perf_counter()
    pbar = tqdm(
        loader,
        desc=f"train {epoch}",
        leave=False,
        disable=not progress,
        dynamic_ncols=False,
        ncols=PROGRESS_NCOLS,
        mininterval=0.5,
        bar_format=PROGRESS_BAR_FORMAT,
    )
    for step, (images, target) in enumerate(pbar):
        if limit_batches is not None and step >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(images) if teacher is not None else None
        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            loss = criterion(output, target, teacher_logits)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc1 = accuracy(output.detach(), target, (1,))[0].item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1, images.size(0))
        if progress:
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc1=f"{acc_meter.avg:.2f}")
    return {"train_loss": loss_meter.avg, "train_acc1": acc_meter.avg, "train_time_s": time.perf_counter() - start}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = True,
    limit_batches: int | None = None,
    progress: bool = True,
    prefix: str = "val",
) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()
    start = time.perf_counter()
    for step, (images, target) in enumerate(
        tqdm(
            loader,
            desc=prefix,
            leave=False,
            disable=not progress,
            dynamic_ncols=False,
            ncols=PROGRESS_NCOLS,
            mininterval=0.5,
            bar_format=PROGRESS_BAR_FORMAT,
        )
    ):
        if limit_batches is not None and step >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            loss = loss_fn(output, target)
        if output.shape[1] >= 5:
            acc1, acc5 = accuracy(output, target, (1, 5))
            acc5_meter.update(acc5.item(), images.size(0))
        else:
            acc1 = accuracy(output, target, (1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))
    metrics = {
        f"{prefix}_loss": loss_meter.avg,
        f"{prefix}_acc1": acc_meter.avg,
        f"{prefix}_time_s": time.perf_counter() - start,
    }
    if acc5_meter.count > 0:
        metrics[f"{prefix}_acc5"] = acc5_meter.avg
    return metrics


def write_metrics(record: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
