from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from .baselines import (
    mobilenet_v2_cifar,
    resnet18_cifar,
    resnet18_stl10,
    shufflenet_v2_x1_0_cifar,
)
from .hbcc import (
    HBCCNet,
    coc_cifar_baseline,
    hbcc_current_reference,
    hbcc_latency_small,
    hbcc_latency_tiny,
    phbcc_2m,
)
from .official_context_cluster import (
    official_coc_medium,
    official_coc_small,
    official_coc_tiny,
    official_coc_tiny_plain,
    official_context_cluster,
)


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "resnet18_cifar": resnet18_cifar,
    "resnet18_stl10": resnet18_stl10,
    "mobilenet_v2_cifar": mobilenet_v2_cifar,
    "shufflenet_v2_x1_0_cifar": shufflenet_v2_x1_0_cifar,
    "official_context_cluster": official_context_cluster,
    "official_coc_tiny": official_coc_tiny,
    "official_coc_tiny_plain": official_coc_tiny_plain,
    "official_coc_small": official_coc_small,
    "official_coc_medium": official_coc_medium,
    "coc_cifar_baseline": coc_cifar_baseline,
    "hbcc_current_reference": hbcc_current_reference,
    "hbcc_latency_tiny": hbcc_latency_tiny,
    "hbcc_latency_small": hbcc_latency_small,
    "phbcc_2m": phbcc_2m,
    "hbcc": HBCCNet,
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
