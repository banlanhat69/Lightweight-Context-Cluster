from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)


def _seed_worker(_: int) -> None:
    """Seed non-Torch RNGs from the deterministic DataLoader worker seed."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def num_classes_for_dataset(name: str) -> int:
    name = name.lower()
    if name == "cifar10":
        return 10
    if name == "cifar100":
        return 100
    if name in {"stl10", "stl-10"}:
        return 10
    if name == "fake":
        return 10
    raise ValueError(f"Unsupported dataset: {name}")


def _random_erasing_value(cfg: dict[str, Any]) -> float | str | tuple[float, ...]:
    value = cfg.get("value", 0.0)
    if isinstance(value, str):
        if value.lower() != "random":
            raise ValueError("random_erasing.value must be numeric, a channel tuple, or 'random'")
        return "random"
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    return float(value)


def _transforms(name: str, train: bool, augment: bool, cfg: dict[str, Any] | None = None) -> transforms.Compose:
    cfg = cfg or {}
    name = name.lower()
    if name in {"stl10", "stl-10"}:
        image_size = int(cfg.get("image_size", 96))
        crop_pct = float(cfg.get("crop_pct", 1.0))
        interpolation = transforms.InterpolationMode.BICUBIC
        if str(cfg.get("interpolation", "bicubic")).lower() in {"bilinear", "linear"}:
            interpolation = transforms.InterpolationMode.BILINEAR
        ops: list[Any] = []
        if train and augment:
            if bool(cfg.get("random_resized_crop", False)):
                ops.append(
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=tuple(cfg.get("rrc_scale", [0.8, 1.0])),
                        interpolation=interpolation,
                    )
                )
            else:
                ops.append(transforms.RandomCrop(image_size, padding=int(cfg.get("padding", 12))))
            ops.append(transforms.RandomHorizontalFlip())
            randaugment = cfg.get("randaugment", {})
            if isinstance(randaugment, dict) and randaugment.get("enabled", False):
                ops.append(
                    transforms.RandAugment(
                        num_ops=int(randaugment.get("num_ops", 2)),
                        magnitude=int(randaugment.get("magnitude", 9)),
                    )
                )
        else:
            resize_size = int(round(image_size / crop_pct))
            if resize_size != image_size:
                ops.append(transforms.Resize(resize_size, interpolation=interpolation))
            ops.append(transforms.CenterCrop(image_size))
        ops.extend([transforms.ToTensor(), transforms.Normalize(STL10_MEAN, STL10_STD)])
        if train and augment:
            random_erasing = cfg.get("random_erasing", {})
            if isinstance(random_erasing, dict) and random_erasing.get("p", 0.0) > 0:
                ops.append(
                    transforms.RandomErasing(
                        p=float(random_erasing.get("p", 0.25)),
                        scale=tuple(random_erasing.get("scale", [0.02, 0.2])),
                        ratio=tuple(random_erasing.get("ratio", [0.3, 3.3])),
                        value=_random_erasing_value(random_erasing),
                    )
                )
        return transforms.Compose(ops)
    if name == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    ops: list[Any] = []
    if train and augment:
        ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
        randaugment = cfg.get("randaugment", {})
        if isinstance(randaugment, dict) and randaugment.get("enabled", False):
            ops.append(
                transforms.RandAugment(
                    num_ops=int(randaugment.get("num_ops", 2)),
                    magnitude=int(randaugment.get("magnitude", 9)),
                )
            )
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if train and augment:
        random_erasing = cfg.get("random_erasing", {})
        if isinstance(random_erasing, dict) and random_erasing.get("p", 0.0) > 0:
            ops.append(
                transforms.RandomErasing(
                    p=float(random_erasing.get("p", 0.25)),
                    scale=tuple(random_erasing.get("scale", [0.02, 0.2])),
                    ratio=tuple(random_erasing.get("ratio", [0.3, 3.3])),
                    value=_random_erasing_value(random_erasing),
                )
            )
    return transforms.Compose(ops)


def _train_val_indices(length: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    if val_size <= 0 or val_size >= length:
        raise ValueError(f"val_size must be between 1 and {length - 1}, got {val_size}")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def build_datasets(
    cfg: dict[str, Any],
    include_test: bool = True,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset | None]:
    name = str(cfg.get("name", "cifar10")).lower()
    root = Path(cfg.get("root", "data"))
    download = bool(cfg.get("download", True))
    augment = bool(cfg.get("augment", True))
    if name == "cifar10":
        train_full = datasets.CIFAR10(
            root=root,
            train=True,
            transform=_transforms(name, True, augment, cfg),
            download=download,
        )
        val_full = datasets.CIFAR10(
            root=root,
            train=True,
            transform=_transforms(name, False, False, cfg),
            download=download,
        )
        test = (
            datasets.CIFAR10(
                root=root,
                train=False,
                transform=_transforms(name, False, False, cfg),
                download=download,
            )
            if include_test
            else None
        )
        val_size = int(cfg.get("val_size", 5000))
        split_seed = int(cfg.get("split_seed", 42))
        train_indices, val_indices = _train_val_indices(len(train_full), val_size, split_seed)
        train = Subset(train_full, train_indices)
        val = Subset(val_full, val_indices)
    elif name == "cifar100":
        train_full = datasets.CIFAR100(
            root=root,
            train=True,
            transform=_transforms(name, True, augment, cfg),
            download=download,
        )
        val_full = datasets.CIFAR100(
            root=root,
            train=True,
            transform=_transforms(name, False, False, cfg),
            download=download,
        )
        test = (
            datasets.CIFAR100(
                root=root,
                train=False,
                transform=_transforms(name, False, False, cfg),
                download=download,
            )
            if include_test
            else None
        )
        val_size = int(cfg.get("val_size", 5000))
        split_seed = int(cfg.get("split_seed", 42))
        train_indices, val_indices = _train_val_indices(len(train_full), val_size, split_seed)
        train = Subset(train_full, train_indices)
        val = Subset(val_full, val_indices)
    elif name in {"stl10", "stl-10"}:
        train_full = datasets.STL10(
            root=root,
            split="train",
            transform=_transforms(name, True, augment, cfg),
            download=download,
        )
        val_full = datasets.STL10(
            root=root,
            split="train",
            transform=_transforms(name, False, False, cfg),
            download=download,
        )
        test = (
            datasets.STL10(
                root=root,
                split="test",
                transform=_transforms(name, False, False, cfg),
                download=download,
            )
            if include_test
            else None
        )
        val_size = int(cfg.get("val_size", 500))
        split_seed = int(cfg.get("split_seed", 42))
        train_indices, val_indices = _train_val_indices(len(train_full), val_size, split_seed)
        train = Subset(train_full, train_indices)
        val = Subset(val_full, val_indices)
    elif name == "fake":
        transform = _transforms("cifar10", False, False, cfg)
        train = datasets.FakeData(size=int(cfg.get("fake_train_size", 512)), image_size=(3, 32, 32), num_classes=10, transform=transform)
        val = datasets.FakeData(size=int(cfg.get("fake_val_size", 128)), image_size=(3, 32, 32), num_classes=10, transform=transform)
        test = (
            datasets.FakeData(
                size=int(cfg.get("fake_test_size", cfg.get("fake_val_size", 128))),
                image_size=(3, 32, 32),
                num_classes=10,
                transform=transform,
            )
            if include_test
            else None
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    train_limit = cfg.get("train_limit")
    val_limit = cfg.get("val_limit")
    test_limit = cfg.get("test_limit")
    if train_limit:
        train = Subset(train, range(min(int(train_limit), len(train))))
    if val_limit:
        val = Subset(val, range(min(int(val_limit), len(val))))
    if test is not None and test_limit:
        test = Subset(test, range(min(int(test_limit), len(test))))
    return train, val, test


def build_loaders(cfg: dict[str, Any], include_test: bool = True) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    train_set, val_set, test_set = build_datasets(cfg, include_test=include_test)
    batch_size = int(cfg.get("batch_size", 128))
    val_batch_size = int(cfg.get("val_batch_size", batch_size))
    test_batch_size = int(cfg.get("test_batch_size", val_batch_size))
    workers = int(cfg.get("workers", 2))
    pin_memory = bool(cfg.get("pin_memory", True))
    loader_seed = int(cfg.get("loader_seed", 0))
    train_generator = torch.Generator().manual_seed(loader_seed)
    val_generator = torch.Generator().manual_seed(loader_seed + 1)
    test_generator = torch.Generator().manual_seed(loader_seed + 2)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=bool(cfg.get("drop_last", True)),
        worker_init_fn=_seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=val_generator,
    )
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=test_generator,
        )
    return train_loader, val_loader, test_loader
