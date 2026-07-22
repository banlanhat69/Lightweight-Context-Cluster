from __future__ import annotations

import json
from pathlib import Path
import random

import pytest
import torch

from lightweight_hbcc import data
from lightweight_hbcc.config import load_config
from lightweight_hbcc.engine import apply_mixup_cutmix


CONFIG_ROOT = Path("configs/cifar_fair")


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
def test_hbcc_legacy_profile_matches_the_reported_recipe(dataset: str) -> None:
    cfg = load_config(CONFIG_ROOT / f"{dataset}_hbcc_legacy.yaml")

    assert cfg["protocol"]["augmentation_profile"] == "hbcc_legacy"
    assert cfg["protocol"]["canonical"] is True
    assert cfg["protocol"]["effective_epochs"] == 300
    assert cfg["data"]["name"] == dataset
    assert cfg["data"]["randaugment"] == {"enabled": True, "num_ops": 2, "magnitude": 9}
    assert cfg["data"]["random_erasing"] == {
        "p": 0.25,
        "scale": [0.02, 0.2],
        "ratio": [0.3, 3.3],
        "value": 0.0,
    }
    assert cfg["train"]["epochs"] == 300
    assert cfg["train"]["warmup_epochs"] == 5
    assert cfg["train"]["lr"] == 0.001
    assert cfg["train"]["weight_decay"] == 0.05
    assert cfg["train"]["label_smoothing"] == 0.1
    assert cfg["train"]["mixup_alpha"] == 0.2
    assert cfg["train"]["cutmix_alpha"] == 1.0
    assert cfg["train"]["cutmix_prob"] == 0.5
    assert cfg["train"]["amp"] is True
    assert cfg["model_overrides"] == {
        "hbcc_small": {"drop_path_rate": 0.08},
        "hbcc_medium": {"drop_path_rate": 0.12},
    }

    signature = repr(data._transforms(dataset, True, True, cfg["data"]))
    assert "RandomCrop" in signature
    assert "RandomHorizontalFlip" in signature
    assert "RandAugment" in signature
    assert "RandomErasing" in signature


def test_hbcc_legacy_mixes_every_training_batch() -> None:
    images = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
    target = torch.tensor([0, 1, 2, 3])

    mixup_images, mixup_target = apply_mixup_cutmix(
        images,
        target,
        num_classes=100,
        label_smoothing=0.1,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        cutmix_prob=0.0,
        rng=random.Random(17),
        torch_generator=torch.Generator().manual_seed(17),
    )
    cutmix_images, cutmix_target = apply_mixup_cutmix(
        images,
        target,
        num_classes=100,
        label_smoothing=0.1,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        cutmix_prob=1.0,
        rng=random.Random(17),
        torch_generator=torch.Generator().manual_seed(17),
    )

    assert not torch.equal(mixup_images, images)
    assert mixup_target.shape == (4, 100)
    assert cutmix_images.shape == images.shape
    assert cutmix_target.shape == (4, 100)
    assert torch.allclose(mixup_target.sum(dim=1), torch.ones(4))
    assert torch.allclose(cutmix_target.sum(dim=1), torch.ones(4))


def test_notebook_defaults_to_hbcc_legacy_and_compiles() -> None:
    notebook_path = Path("notebooks/cifar_fair_training.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "AUGMENTATION_PROFILE = None" in code
    assert "'cifar10': 'hbcc_legacy'" in code
    assert "'cifar100': 'hbcc_legacy'" in code
    assert "cifar10_hbcc_legacy.yaml" in code
    assert "cifar100_hbcc_legacy.yaml" in code
    assert "profile_model_overrides" in code
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"notebook-cell-{index}", "exec")
