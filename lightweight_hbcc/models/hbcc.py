from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .cluster import ContextClusterOp, Stage
from .layers import CoordinateAugment, PointReducer, make_norm


HBCC_WIDE_CIFAR_CONFIGS: dict[str, dict[str, Any]] = {
    "hbcc_small": {
        "name": "hbcc",
        "use_coord": True,
        "embed_dims": [48, 80, 160, 256],
        "depths": [1, 1, 2, 1],
        "mlp_ratios": 3.0,
        "heads": [2, 2, 4, 4],
        "head_dim": [16, 16, 16, 16],
        "proposals": [[2, 2], [2, 2], [2, 2], [1, 1]],
        "folds": [[4, 4], [2, 2], [1, 1], [1, 1]],
        "similarities": ["cosine", "cosine", "cosine", "cosine"],
        "assignment_modes": ["hard", "hard", "hard", "hard"],
        "assignment_temperatures": [1.0, 1.0, 1.0, 1.0],
        "positive_similarity_scales": [False, False, False, False],
        "stage_modes": ["hybrid", "hybrid", "cluster", "cluster"],
        "local_branches": ["lbpconv", "dwconv", "identity", "identity"],
        "local_ratios": [0.5, 0.5, 0.0, 0.0],
        "channel_shuffle": [True, True, False, False],
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
        "depths": [1, 1, 2, 1],
        "mlp_ratios": 3.0,
        "heads": [2, 3, 4, 4],
        "head_dim": [16, 16, 16, 16],
        "proposals": [[2, 2], [2, 2], [2, 2], [1, 1]],
        "folds": [[4, 4], [2, 2], [1, 1], [1, 1]],
        "similarities": ["cosine", "cosine", "cosine", "cosine"],
        "assignment_modes": ["hard", "hard", "hard", "hard"],
        "assignment_temperatures": [1.0, 1.0, 1.0, 1.0],
        "positive_similarity_scales": [False, False, False, False],
        "stage_modes": ["hybrid", "hybrid", "cluster", "cluster"],
        "local_branches": ["lbpconv", "dwconv", "identity", "identity"],
        "local_ratios": [0.5, 0.5, 0.0, 0.0],
        "channel_shuffle": [True, True, False, False],
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


HBCC_STAGE4_ABLATION_CIFAR_CONFIGS: dict[str, dict[str, Any]] = {
    "hbcc_medium_stage4_ablation": {
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
        "assignment_modes": ["hard", "hard", "hard", "hard_st"],
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
        # Preserve the best CE model's Stage 1-3 stochastic-depth schedule.
        "stage_drop_path_rates": [[0.0], [0.02], [0.04, 0.06], [0.08, 0.08]],
    },
}


_STAGE_INDEXED_ABLATION_FIELDS = (
    "depths",
    "proposals",
    "assignment_modes",
    "assignment_temperatures",
    "stage_modes",
    "local_branches",
    "local_ratios",
    "channel_shuffle",
)


def validate_hbcc_wide_cifar_config(model_key: str, config: dict[str, Any]) -> list[str]:
    """Return catalog drift errors for the restored widened-Stage-4 variants."""

    expected = HBCC_WIDE_CIFAR_CONFIGS.get(model_key)
    if expected is None:
        return [f"unknown widened HBCC variant: {model_key}"]
    return [
        f"{model_key}.{field} must be {value!r}, got {config.get(field)!r}"
        for field, value in expected.items()
        if config.get(field) != value
    ]


def validate_hbcc_stage4_ablation_config(
    model_key: str,
    config: dict[str, Any],
) -> list[str]:
    """Validate the Stage-4-only ablation and guard Stage 1-3 from drift."""

    expected = HBCC_STAGE4_ABLATION_CIFAR_CONFIGS.get(model_key)
    if expected is None:
        return [f"unknown HBCC Stage 4 ablation variant: {model_key}"]

    errors = [
        f"{model_key}.{field} must be {value!r}, got {config.get(field)!r}"
        for field, value in expected.items()
        if config.get(field) != value
    ]
    baseline = HBCC_WIDE_CIFAR_CONFIGS["hbcc_medium"]
    for field in _STAGE_INDEXED_ABLATION_FIELDS:
        actual = config.get(field)
        if not isinstance(actual, (list, tuple)) or list(actual[:3]) != list(
            baseline[field][:3]
        ):
            errors.append(
                f"{model_key}.{field} Stage 1-3 must remain exactly "
                f"{baseline[field][:3]!r}, got {actual!r}"
            )
    for field, value in baseline.items():
        if field not in _STAGE_INDEXED_ABLATION_FIELDS:
            if config.get(field) != value:
                errors.append(
                    f"{model_key}.{field} is outside Stage 4 and must remain "
                    f"{value!r}, got {config.get(field)!r}"
                )
    return errors


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
    """Four-stage hybrid binary/context-cluster image classifier."""

    def __init__(
        self,
        num_classes: int = 10,
        in_chans: int = 3,
        use_coord: bool = True,
        embed_dims: list[int] | tuple[int, int, int, int] = (48, 80, 160, 256),
        depths: list[int] | tuple[int, int, int, int] = (1, 1, 2, 1),
        mlp_ratios: float | list[float] = 3.0,
        heads: list[int] | tuple[int, int, int, int] = (2, 2, 4, 4),
        head_dim: int | list[int] = 16,
        proposals: list[tuple[int, int]] | tuple[tuple[int, int], ...] = ((2, 2), (2, 2), (2, 2), (1, 1)),
        folds: list[tuple[int, int]] | tuple[tuple[int, int], ...] = ((4, 4), (2, 2), (1, 1), (1, 1)),
        similarities: str | list[str] = "cosine",
        assignment_modes: str | list[str] = "hard",
        assignment_temperatures: float | list[float] = 1.0,
        positive_similarity_scales: bool | list[bool] = False,
        local_branches: str | list[str] = ("lbpconv", "dwconv", "identity", "identity"),
        local_ratios: float | list[float] = (0.5, 0.5, 0.0, 0.0),
        stage_modes: str | list[str] = ("hybrid", "hybrid", "cluster", "cluster"),
        channel_shuffle: bool | list[bool] = (True, True, False, False),
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
        stage_drop_path_rates: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
        cluster_balance_loss_weight: float = 0.0,
        cluster_balance_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if len(embed_dims) != 4 or len(depths) != 4:
            raise ValueError("HBCCNet expects four stages.")
        self.num_classes = num_classes
        self.embed_dims = list(embed_dims)
        self.depths = list(depths)
        self.use_coord = use_coord
        self.stem_stride = int(stem_stride)
        self.cluster_balance_loss_weight = float(cluster_balance_loss_weight)
        self.cluster_balance_temperature = float(cluster_balance_temperature)
        if self.cluster_balance_loss_weight < 0.0:
            raise ValueError("cluster_balance_loss_weight must be non-negative.")
        if self.cluster_balance_temperature <= 0.0:
            raise ValueError("cluster_balance_temperature must be positive.")
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

        if stage_drop_path_rates is None:
            total_depth = sum(depths)
            flat_dpr = (
                torch.linspace(0, drop_path_rate, total_depth).tolist()
                if total_depth > 0
                else []
            )
            stage_dpr: list[list[float]] = []
            cursor = 0
            for depth in depths:
                stage_dpr.append(flat_dpr[cursor : cursor + depth])
                cursor += depth
        else:
            if len(stage_drop_path_rates) != 4:
                raise ValueError("stage_drop_path_rates must contain four stage lists.")
            stage_dpr = [
                torch.tensor(list(map(float, rates)), dtype=torch.float32).tolist()
                for rates in stage_drop_path_rates
            ]
            for idx, (rates, depth) in enumerate(zip(stage_dpr, depths, strict=True)):
                if len(rates) != depth:
                    raise ValueError(
                        f"stage_drop_path_rates[{idx}] must contain {depth} values, "
                        f"got {len(rates)}."
                    )
                if any(rate < 0.0 or rate >= 1.0 for rate in rates):
                    raise ValueError(
                        f"stage_drop_path_rates[{idx}] values must be in [0, 1)."
                    )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for idx in range(4):
            rates = stage_dpr[idx]
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
        for module in self.modules():
            if isinstance(module, ContextClusterOp):
                module.compute_balance_loss = self.cluster_balance_loss_weight > 0.0
                module.balance_temperature = self.cluster_balance_temperature

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

    def auxiliary_loss(self) -> torch.Tensor:
        """Return the unweighted differentiable center-usage balance loss."""

        losses = [
            module.last_balance_loss
            for module in self.modules()
            if isinstance(module, ContextClusterOp)
            and module.last_balance_loss is not None
        ]
        if not losses:
            return self.head.weight.new_zeros(())
        return torch.stack(losses).mean()
