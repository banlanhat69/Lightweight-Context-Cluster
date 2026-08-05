from __future__ import annotations

from typing import Any

from torch import nn

from .hbcc import HBCCNet


HBCC_FOOD101_BEST_CONFIG: dict[str, Any] = {
    "use_coord": True,
    "embed_dims": [64, 96, 192, 288],
    "depths": [2, 2, 4, 2],
    "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
    # Avoid the former 192/216 -> 64 cluster-projection bottleneck.
    "heads": [2, 3, 6, 7],
    "head_dim": [16, 16, 32, 32],
    "proposals": [[2, 2], [2, 2], [2, 2], [2, 2]],
    # 56/28/14/7 feature maps become uniform 7x7 clustering regions.
    "folds": [[8, 8], [4, 4], [2, 2], [1, 1]],
    "similarities": ["cosine", "cosine", "cosine", "cosine"],
    "assignment_modes": ["hard", "hard", "hard", "hard"],
    "assignment_temperatures": [1.0, 1.0, 1.0, 1.0],
    "positive_similarity_scales": [True, True, True, True],
    "stage_modes": ["hybrid", "hybrid", "cluster", "hybrid"],
    "local_branches": ["lbpconv", "dwconv", "identity", "dwconv"],
    "local_ratios": [0.5, 0.5, 0.0, 0.25],
    "channel_shuffle": [True, True, False, True],
    "layer_scale_init_values": 1.0e-3,
    "norm": "gn",
    "stem_patch_size": 7,
    "stem_stride": 4,
    "stem_padding": 3,
    "down_patch_size": 3,
    "down_stride": 2,
    "down_padding": 1,
    "drop_rate": 0.10,
    "drop_path_rate": 0.15,
    # Forward assignment remains purely hard. Soft probabilities are used only
    # by this small training-only center-usage regularizer.
    "cluster_balance_loss_weight": 0.01,
    "cluster_balance_temperature": 1.0,
}


def hbcc_food101_best(num_classes: int = 101, **overrides: Any) -> nn.Module:
    """Canonical accuracy-oriented HBCC for 224x224 Food-101 experiments."""

    unexpected = sorted(overrides)
    if unexpected:
        raise ValueError(
            "hbcc_food101_best is a locked architecture; unexpected overrides: "
            + ", ".join(unexpected)
        )
    return HBCCNet(num_classes=num_classes, **HBCC_FOOD101_BEST_CONFIG)

