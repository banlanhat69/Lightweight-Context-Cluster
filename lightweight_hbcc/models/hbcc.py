from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .cluster import Stage
from .layers import CoordinateAugment, PointReducer, make_norm


HBCC_ACCURACY_CIFAR_CONFIGS: dict[str, dict[str, Any]] = {
    "hbcc_small": {
        "name": "hbcc",
        "use_coord": True,
        "embed_dims": [48, 80, 160, 256],
        "depths": [1, 1, 2, 2],
        "mlp_ratios": 3.0,
        "heads": [2, 2, 4, 4],
        "head_dim": [16, 16, 16, 16],
        "proposals": [[2, 2], [2, 2], [2, 2], [2, 2]],
        "folds": [[4, 4], [2, 2], [1, 1], [1, 1]],
        "similarities": ["cosine", "cosine", "cosine", "cosine"],
        "assignment_modes": ["hard_st", "hard_st", "hard_st", "hard_st"],
        "assignment_temperatures": [1.0, 1.0, 1.0, 0.7],
        "positive_similarity_scales": [False, False, False, False],
        "stage_modes": ["hybrid", "hybrid", "cluster", "hybrid"],
        "local_branches": ["lbpconv", "dwconv", "identity", "dwconv"],
        "local_ratios": [0.5, 0.5, 0.0, 0.25],
        "channel_shuffle": [True, True, False, True],
        "layer_scale_init_values": 1.0e-5,
        "norm": "bn",
        "stem_patch_size": 3,
        "stem_stride": 1,
        "stem_padding": 1,
        "down_patch_size": 3,
        "down_stride": 2,
        "down_padding": 1,
        "drop_rate": 0.0,
        "drop_path_rate": 0.05,
    },
    "hbcc_medium": {
        "name": "hbcc",
        "use_coord": True,
        "embed_dims": [64, 96, 192, 288],
        "depths": [1, 1, 2, 2],
        "mlp_ratios": 3.0,
        "heads": [2, 3, 4, 4],
        "head_dim": [16, 16, 16, 16],
        "proposals": [[2, 2], [2, 2], [2, 2], [2, 2]],
        "folds": [[4, 4], [2, 2], [1, 1], [1, 1]],
        "similarities": ["cosine", "cosine", "cosine", "cosine"],
        "assignment_modes": ["hard_st", "hard_st", "hard_st", "hard_st"],
        "assignment_temperatures": [1.0, 1.0, 1.0, 0.7],
        "positive_similarity_scales": [False, False, False, False],
        "stage_modes": ["hybrid", "hybrid", "cluster", "hybrid"],
        "local_branches": ["lbpconv", "dwconv", "identity", "dwconv"],
        "local_ratios": [0.5, 0.5, 0.0, 0.25],
        "channel_shuffle": [True, True, False, True],
        "layer_scale_init_values": 1.0e-5,
        "norm": "bn",
        "stem_patch_size": 3,
        "stem_stride": 1,
        "stem_padding": 1,
        "down_patch_size": 3,
        "down_stride": 2,
        "down_padding": 1,
        "drop_rate": 0.0,
        "drop_path_rate": 0.08,
    },
}


def validate_hbcc_accuracy_cifar_config(model_key: str, config: dict[str, Any]) -> list[str]:
    """Return catalog drift errors for the accuracy-oriented HBCC variants."""

    expected = HBCC_ACCURACY_CIFAR_CONFIGS.get(model_key)
    if expected is None:
        return [f"unknown accuracy-oriented HBCC variant: {model_key}"]
    return [
        f"{model_key}.{field} must be {value!r}, got {config.get(field)!r}"
        for field, value in expected.items()
        if config.get(field) != value
    ]


def _as_list(value: Any, length: int) -> list[Any]:
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"Expected list of length {length}, got {len(value)}")
        return list(value)
    return [value for _ in range(length)]


def _tuple2(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected pair, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class HBCCNet(nn.Module):
    """Accuracy-oriented CIFAR HBCC with an information-preserving Stage 4.

    The stem keeps the 32x32 CIFAR resolution and the stages operate at
    32/16/8/4. Stage 4 uses four proposals, a 25% depthwise-convolution branch,
    straight-through hard assignment, and two blocks to avoid a one-center
    bottleneck before global pooling.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_chans: int = 3,
        use_coord: bool = True,
        embed_dims: list[int] | tuple[int, int, int, int] = (48, 80, 160, 256),
        depths: list[int] | tuple[int, int, int, int] = (1, 1, 2, 2),
        mlp_ratios: float | list[float] = 3.0,
        heads: list[int] | tuple[int, int, int, int] = (2, 2, 4, 4),
        head_dim: int | list[int] = 16,
        proposals: list[tuple[int, int]] | tuple[tuple[int, int], ...] = ((2, 2), (2, 2), (2, 2), (2, 2)),
        folds: list[tuple[int, int]] | tuple[tuple[int, int], ...] = ((4, 4), (2, 2), (1, 1), (1, 1)),
        similarities: str | list[str] = "cosine",
        assignment_modes: str | list[str] = "hard_st",
        assignment_temperatures: float | list[float] = (1.0, 1.0, 1.0, 0.7),
        positive_similarity_scales: bool | list[bool] = False,
        local_branches: str | list[str] = ("lbpconv", "dwconv", "identity", "dwconv"),
        local_ratios: float | list[float] = (0.5, 0.5, 0.0, 0.25),
        stage_modes: str | list[str] = ("hybrid", "hybrid", "cluster", "hybrid"),
        channel_shuffle: bool | list[bool] = (True, True, False, True),
        layer_scale_init_values: float | list[float] = 1e-5,
        norm: str = "bn",
        stem_patch_size: int = 3,
        stem_stride: int = 1,
        stem_padding: int = 1,
        down_patch_size: int | list[int] | tuple[int, int, int] = 3,
        down_stride: int | list[int] | tuple[int, int, int] = 2,
        down_padding: int | list[int] | tuple[int, int, int] = 1,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.05,
    ) -> None:
        super().__init__()
        if len(embed_dims) != 4 or len(depths) != 4:
            raise ValueError("HBCCNet expects four stages.")
        self.num_classes = num_classes
        self.embed_dims = list(embed_dims)
        self.depths = list(depths)
        self.use_coord = use_coord
        self.stem_stride = int(stem_stride)
        self.coord = CoordinateAugment(enabled=use_coord)
        stem_in = in_chans + (2 if use_coord else 0)
        self.stem = PointReducer(stem_in, embed_dims[0], stem_patch_size, stem_stride, stem_padding, norm=norm)

        mlp_ratios = _as_list(mlp_ratios, 4)
        head_dim = _as_list(head_dim, 4)
        similarities = _as_list(similarities, 4)
        assignment_modes = _as_list(assignment_modes, 4)
        assignment_temperatures = _as_list(assignment_temperatures, 4)
        positive_similarity_scales = _as_list(positive_similarity_scales, 4)
        local_branches = _as_list(local_branches, 4)
        local_ratios = _as_list(local_ratios, 4)
        stage_modes = _as_list(stage_modes, 4)
        channel_shuffle = _as_list(channel_shuffle, 4)
        layer_scale_init_values = _as_list(layer_scale_init_values, 4)
        proposals = [_tuple2(v) for v in proposals]
        folds = [_tuple2(v) for v in folds]
        self.proposals = proposals
        self.folds = folds
        down_patch_sizes = [int(v) for v in _as_list(down_patch_size, 3)]
        down_strides = [int(v) for v in _as_list(down_stride, 3)]
        down_paddings = [int(v) for v in _as_list(down_padding, 3)]
        if any(value <= 0 for value in down_patch_sizes):
            raise ValueError(f"down_patch_size values must be positive, got {down_patch_sizes}")
        if any(value <= 0 for value in down_strides):
            raise ValueError(f"down_stride values must be positive, got {down_strides}")
        if any(value < 0 for value in down_paddings):
            raise ValueError(f"down_padding values must be non-negative, got {down_paddings}")
        self.down_strides = down_strides

        total_depth = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist() if total_depth > 0 else []
        cursor = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for idx in range(4):
            rates = dpr[cursor : cursor + depths[idx]]
            cursor += depths[idx]
            self.stages.append(
                Stage(
                    dim=embed_dims[idx],
                    depth=depths[idx],
                    mlp_ratio=float(mlp_ratios[idx]),
                    proposal=proposals[idx],
                    fold=folds[idx],
                    heads=int(heads[idx]),
                    head_dim=int(head_dim[idx]),
                    similarity=str(similarities[idx]),
                    local_branch=str(local_branches[idx]),
                    local_ratio=float(local_ratios[idx]),
                    mode=str(stage_modes[idx]),
                    channel_shuffle_enabled=bool(channel_shuffle[idx]),
                    norm=norm,
                    drop=drop_rate,
                    drop_path_rates=rates,
                    layer_scale_init_value=float(layer_scale_init_values[idx]),
                    assignment_mode=str(assignment_modes[idx]),
                    assignment_temperature=float(assignment_temperatures[idx]),
                    positive_similarity_scale=bool(positive_similarity_scales[idx]),
                )
            )
            if idx < 3:
                self.downsamples.append(
                    PointReducer(
                        embed_dims[idx],
                        embed_dims[idx + 1],
                        down_patch_sizes[idx],
                        down_strides[idx],
                        down_paddings[idx],
                        norm=norm,
                    )
                )
        self.norm = make_norm(norm, embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

    def forward_intermediates(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return the four post-stage feature maps at 32/16/8/4 resolution."""

        x = self.coord(x)
        x = self.stem(x)
        features: list[torch.Tensor] = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx == len(self.stages) - 1:
                x = self.norm(x)
            features.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        return tuple(features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_intermediates(x)[-1]

    def forward_with_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        features = self.forward_intermediates(x)
        logits = self.head(features[-1].mean(dim=(-2, -1)))
        return logits, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(x)
        return logits
