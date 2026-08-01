from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

_REMOVED_AUGMENTATION_KEYS = {
    "augment",
    "randaugment",
    "random_erasing",
    "random_resized_crop",
    "rrc_scale",
}


def num_classes_for_dataset(name: str) -> int:
    name = name.lower()
    if name == "cifar10":
        return 10
    if name == "cifar100":
        return 100
    if name == "fake":
        return 10
    raise ValueError(f"Unsupported dataset: {name}")


def validate_no_augmentation_config(cfg: dict[str, Any]) -> None:
    """Reject legacy augmentation settings instead of silently ignoring them."""

    legacy_keys = sorted(_REMOVED_AUGMENTATION_KEYS.intersection(cfg))
    if legacy_keys:
        joined = ", ".join(legacy_keys)
        raise ValueError(
            "This pipeline intentionally has no data augmentation. "
            f"Remove legacy data settings: {joined}"
        )


def build_transform(name: str) -> transforms.Compose:
    """Build the identical normalization-only transform for every split."""

    name = name.lower()
    if name == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    elif name in {"cifar10", "fake"}:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


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
    validate_no_augmentation_config(cfg)
    name = str(cfg.get("name", "cifar10")).lower()
    root = Path(cfg.get("root", "data"))
    download = bool(cfg.get("download", True))

    if name in {"cifar10", "cifar100"}:
        dataset_type = datasets.CIFAR100 if name == "cifar100" else datasets.CIFAR10
        transform = build_transform(name)
        train_full = dataset_type(
            root=root,
            train=True,
            transform=transform,
            download=download,
        )
        val_full = dataset_type(
            root=root,
            train=True,
            transform=build_transform(name),
            download=download,
        )
        test = (
            dataset_type(
                root=root,
                train=False,
                transform=build_transform(name),
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
    elif name == "fake":
        transform = build_transform(name)
        train = datasets.FakeData(
            size=int(cfg.get("fake_train_size", 512)),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
        )
        val = datasets.FakeData(
            size=int(cfg.get("fake_val_size", 128)),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
        )
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


def build_loaders(
    cfg: dict[str, Any],
    include_test: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
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
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
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
            generator=test_generator,
        )
    return train_loader, val_loader, test_loader
