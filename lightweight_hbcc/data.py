from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as transform_functional


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_REMOVED_AUGMENTATION_KEYS = {
    "augment",
    "randaugment",
    "random_erasing",
    "random_resized_crop",
    "rrc_scale",
}

FOOD101_AUGMENTATION_RECIPES = {
    "coc_food101_v1",
    "fair_food101_v1",
    "hbcc_best100_v1",
    "resnet18_best100_v1",
}


class RandomResizedCropRandomInterpolation(transforms.RandomResizedCrop):
    """CoC/timm-style crop with a per-image bilinear/bicubic choice."""

    def forward(self, image: Any) -> Any:
        top, left, height, width = self.get_params(image, self.scale, self.ratio)
        interpolation = random.choice(
            (
                transforms.InterpolationMode.BILINEAR,
                transforms.InterpolationMode.BICUBIC,
            )
        )
        return transform_functional.resized_crop(
            image,
            top,
            left,
            height,
            width,
            self.size,
            interpolation,
            antialias=getattr(self, "antialias", True),
        )


class CoCRandAugment:
    """torchvision equivalent of timm ``rand-m9-mstd0.5-inc1``."""

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        magnitude_std: float = 0.5,
        num_magnitude_bins: int = 11,
    ) -> None:
        if num_magnitude_bins <= 1:
            raise ValueError("num_magnitude_bins must be greater than one.")
        self.magnitude = int(magnitude)
        self.magnitude_std = float(magnitude_std)
        self.maximum_magnitude = int(num_magnitude_bins) - 1
        self.transforms = {
            level: transforms.RandAugment(
                num_ops=int(num_ops),
                magnitude=level,
                num_magnitude_bins=int(num_magnitude_bins),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
            for level in range(int(num_magnitude_bins))
        }

    def __call__(self, image: Any) -> Any:
        level = int(round(random.gauss(self.magnitude, self.magnitude_std)))
        level = max(0, min(level, self.maximum_magnitude))
        return self.transforms[level](image)


def num_classes_for_dataset(name: str) -> int:
    name = name.lower()
    if name == "cifar10":
        return 10
    if name == "cifar100":
        return 100
    if name == "fake":
        return 10
    if name == "food101":
        return 101
    raise ValueError(f"Unsupported dataset: {name}")


def validate_no_augmentation_config(cfg: dict[str, Any]) -> None:
    """Reject legacy augmentation settings instead of silently ignoring them."""

    legacy_keys = sorted(_REMOVED_AUGMENTATION_KEYS.intersection(cfg))
    augmentation = str(cfg.get("augmentation", "none")).lower()
    if legacy_keys and augmentation == "none":
        joined = ", ".join(legacy_keys)
        raise ValueError(
            "This pipeline intentionally has no data augmentation. "
            f"Remove legacy data settings: {joined}"
        )


def build_transform(name: str, image_size: int = 224) -> transforms.Compose:
    """Build the identical normalization-only transform for every split."""

    name = name.lower()
    if name == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    elif name in {"cifar10", "fake"}:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif name == "food101":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def build_food101_transform(cfg: dict[str, Any], training: bool) -> transforms.Compose:
    """Build a configured Food-101 train or deterministic evaluation transform.

    CoC and fair recipes share the implementation but use different strengths.
    Validation follows a configurable center-crop ratio. All tensors reaching
    a model are exactly ``image_size``.
    """

    image_size = int(cfg.get("image_size", 224))
    augmentation = str(cfg.get("augmentation", "none")).lower()
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if training and augmentation in FOOD101_AUGMENTATION_RECIPES:
        scale = tuple(float(value) for value in cfg.get("rrc_scale", (0.08, 1.0)))
        ratio = tuple(float(value) for value in cfg.get("rrc_ratio", (0.75, 4.0 / 3.0)))
        interpolation = str(cfg.get("train_interpolation", "random")).lower()
        if interpolation == "random":
            crop: transforms.RandomResizedCrop = RandomResizedCropRandomInterpolation(
                image_size,
                scale=scale,
                ratio=ratio,
            )
        elif interpolation in {"bilinear", "bicubic"}:
            crop = transforms.RandomResizedCrop(
                image_size,
                scale=scale,
                ratio=ratio,
                interpolation=getattr(transforms.InterpolationMode, interpolation.upper()),
            )
        else:
            raise ValueError(
                "train_interpolation must be random, bilinear, or bicubic; "
                f"got {interpolation!r}."
            )
        return transforms.Compose(
            [
                crop,
                transforms.RandomHorizontalFlip(p=float(cfg.get("hflip", 0.5))),
                CoCRandAugment(
                    num_ops=int(cfg.get("randaugment_num_ops", 2)),
                    magnitude=int(cfg.get("randaugment_magnitude", 9)),
                    magnitude_std=float(cfg.get("randaugment_magnitude_std", 0.5)),
                    num_magnitude_bins=int(
                        cfg.get("randaugment_num_magnitude_bins", 11)
                    ),
                ),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(
                    p=float(cfg.get("random_erasing", 0.25)),
                    value="random",
                ),
            ]
        )
    if augmentation not in {"none", *FOOD101_AUGMENTATION_RECIPES}:
        raise ValueError(f"Unsupported Food-101 augmentation recipe: {augmentation}")
    if augmentation in FOOD101_AUGMENTATION_RECIPES:
        crop_pct = float(cfg.get("eval_crop_pct", 0.9))
        resize_size = int(round(image_size / crop_pct))
        return transforms.Compose(
            [
                transforms.Resize(
                    resize_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return build_transform("food101", image_size=image_size)


def _train_val_indices(length: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    if val_size <= 0 or val_size >= length:
        raise ValueError(f"val_size must be between 1 and {length - 1}, got {val_size}")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def _food101_stratified_indices(
    dataset: torch.utils.data.Dataset,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create the canonical per-class Food-101 train/validation split."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    raw_labels = list(getattr(dataset, "_labels", []))
    classes = list(getattr(dataset, "classes", []))
    if len(raw_labels) != len(dataset) or not classes:
        raise RuntimeError("Unsupported torchvision Food101 label metadata.")
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    if isinstance(raw_labels[0], str):
        labels = [class_to_idx[label] for label in raw_labels]
    else:
        labels = [int(label) for label in raw_labels]

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    label_tensor = torch.tensor(labels, dtype=torch.long)
    for class_idx in range(len(classes)):
        class_indices = torch.nonzero(label_tensor == class_idx, as_tuple=False).flatten()
        order = torch.randperm(class_indices.numel(), generator=generator)
        class_indices = class_indices[order]
        val_count = int(round(class_indices.numel() * val_fraction))
        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())
    if set(train_indices).intersection(val_indices):
        raise RuntimeError("Food-101 train/validation split contains overlapping indices.")
    if len(train_indices) + len(val_indices) != len(dataset):
        raise RuntimeError("Food-101 train/validation split lost samples.")
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
    elif name == "food101":
        train_transform = build_food101_transform(cfg, training=True)
        eval_transform = build_food101_transform(cfg, training=False)
        train_full = datasets.Food101(
            root=root,
            split="train",
            transform=train_transform,
            download=download,
        )
        val_full = datasets.Food101(
            root=root,
            split="train",
            transform=eval_transform,
            download=download,
        )
        test = (
            datasets.Food101(
                root=root,
                split="test",
                transform=eval_transform,
                download=download,
            )
            if include_test
            else None
        )
        train_indices, val_indices = _food101_stratified_indices(
            train_full,
            val_fraction=float(cfg.get("val_fraction", 0.10)),
            seed=int(cfg.get("split_seed", 42)),
        )
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
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        generator=val_generator,
        persistent_workers=workers > 0,
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
            persistent_workers=workers > 0,
        )
    return train_loader, val_loader, test_loader
