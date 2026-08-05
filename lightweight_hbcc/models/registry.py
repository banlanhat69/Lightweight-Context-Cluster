from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from .baselines import (
    mobilenet_v2_cifar,
    resnet18_cifar,
    resnet18_224,
    shufflenet_v2_x1_0_cifar,
)
from .hbcc import HBCCNet
from .food101 import hbcc_food101_best


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "resnet18_cifar": resnet18_cifar,
    "resnet18_224": resnet18_224,
    "mobilenet_v2_cifar": mobilenet_v2_cifar,
    "shufflenet_v2_x1_0_cifar": shufflenet_v2_x1_0_cifar,
    "hbcc": HBCCNet,
    "hbcc_food101_best": hbcc_food101_best,
}


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", config)
    name = model_cfg.get("name") or model_cfg.get("type")
    if not name:
        raise ValueError("Model config requires 'name' or 'type'.")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(list_models())}")
    kwargs = {k: v for k, v in model_cfg.items() if k not in {"name", "type"}}
    return MODEL_REGISTRY[name](**kwargs)
