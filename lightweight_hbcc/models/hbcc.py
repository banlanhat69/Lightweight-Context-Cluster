from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .cluster import Stage
from .layers import CoordinateAugment, PointReducer, make_norm


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
    """CIFAR-oriented lightweight Context Cluster/HBCC network."""

    def __init__(
        self,
        num_classes: int = 10,
        in_chans: int = 3,
        use_coord: bool = True,
        embed_dims: list[int] | tuple[int, int, int, int] = (32, 48, 96, 160),
        depths: list[int] | tuple[int, int, int, int] = (1, 1, 2, 1),
        mlp_ratios: float | list[float] = 2.0,
        heads: list[int] | tuple[int, int, int, int] = (1, 2, 2, 4),
        head_dim: int | list[int] = 16,
        proposals: list[tuple[int, int]] | tuple[tuple[int, int], ...] = ((2, 2), (2, 2), (2, 2), (2, 2)),
        folds: list[tuple[int, int]] | tuple[tuple[int, int], ...] = ((4, 4), (2, 2), (1, 1), (1, 1)),
        similarities: str | list[str] = "cosine",
        local_branches: str | list[str] = "dwconv",
        local_ratios: float | list[float] = 0.0,
        stage_modes: str | list[str] = ("local", "hybrid", "cluster", "cluster"),
        channel_shuffle: bool | list[bool] = False,
        norm: str = "bn",
        stem_patch_size: int = 3,
        stem_stride: int = 2,
        stem_padding: int = 1,
        down_patch_size: int | list[int] | tuple[int, int, int] = 3,
        down_stride: int | list[int] | tuple[int, int, int] = 2,
        down_padding: int | list[int] | tuple[int, int, int] = 1,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        pruning_mask: bool = False,
    ) -> None:
        super().__init__()
        if len(embed_dims) != 4 or len(depths) != 4:
            raise ValueError("HBCCNet expects four stages.")
        self.num_classes = num_classes
        self.embed_dims = list(embed_dims)
        self.depths = list(depths)
        self.use_coord = use_coord
        self.coord = CoordinateAugment(enabled=use_coord)
        stem_in = in_chans + (2 if use_coord else 0)
        self.stem = PointReducer(stem_in, embed_dims[0], stem_patch_size, stem_stride, stem_padding, norm=norm)

        mlp_ratios = _as_list(mlp_ratios, 4)
        head_dim = _as_list(head_dim, 4)
        similarities = _as_list(similarities, 4)
        local_branches = _as_list(local_branches, 4)
        local_ratios = _as_list(local_ratios, 4)
        stage_modes = _as_list(stage_modes, 4)
        channel_shuffle = _as_list(channel_shuffle, 4)
        proposals = [_tuple2(v) for v in proposals]
        folds = [_tuple2(v) for v in folds]
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
                    pruning_mask=pruning_mask,
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coord(x)
        x = self.stem(x)
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x.mean(dim=(-2, -1)))

    def pruning_regularization(self, sparsity_weight: float, crispness_weight: float) -> torch.Tensor:
        loss = torch.zeros((), device=next(self.parameters()).device)
        for stage in self.stages:
            gate = getattr(stage, "gate", None)
            if hasattr(gate, "regularization"):
                loss = loss + gate.regularization(sparsity_weight, crispness_weight)
        return loss


def hbcc_latency_tiny(num_classes: int = 10, **kwargs) -> HBCCNet:
    return HBCCNet(
        num_classes=num_classes,
        embed_dims=[32, 48, 96, 160],
        depths=[1, 1, 2, 1],
        mlp_ratios=2.0,
        heads=[1, 2, 2, 4],
        head_dim=[16, 16, 16, 16],
        proposals=[(2, 2), (2, 2), (2, 2), (2, 2)],
        folds=[(4, 4), (2, 2), (1, 1), (1, 1)],
        stage_modes=["local", "hybrid", "cluster", "cluster"],
        local_branches=["dwconv", "dwconv", "identity", "identity"],
        local_ratios=[1.0, 0.5, 0.0, 0.0],
        use_coord=True,
        **kwargs,
    )


def hbcc_latency_small(num_classes: int = 10, **kwargs) -> HBCCNet:
    return HBCCNet(
        num_classes=num_classes,
        embed_dims=[32, 64, 128, 192],
        depths=[1, 2, 2, 1],
        mlp_ratios=3.0,
        heads=[1, 2, 4, 4],
        head_dim=[16, 16, 16, 16],
        proposals=[(2, 2), (2, 2), (2, 2), (2, 2)],
        folds=[(4, 4), (2, 2), (1, 1), (1, 1)],
        stage_modes=["local", "hybrid", "cluster", "cluster"],
        local_branches=["dwconv", "dwconv", "identity", "identity"],
        local_ratios=[1.0, 0.5, 0.0, 0.0],
        use_coord=True,
        **kwargs,
    )


def hbcc_current_reference(num_classes: int = 10, **kwargs) -> HBCCNet:
    """Reference matching the diagnosed slow path: no xy and first stage at 32x32."""

    return HBCCNet(
        num_classes=num_classes,
        embed_dims=[32, 64, 128, 192],
        depths=[1, 1, 2, 1],
        mlp_ratios=3.0,
        heads=[2, 2, 4, 4],
        head_dim=[16, 16, 16, 16],
        proposals=[(2, 2), (2, 2), (2, 2), (2, 2)],
        folds=[(1, 1), (2, 2), (1, 1), (1, 1)],
        stage_modes=["cluster", "hybrid", "cluster", "cluster"],
        local_branches=["identity", "dwconv", "identity", "identity"],
        local_ratios=[0.0, 0.5, 0.0, 0.0],
        use_coord=False,
        stem_stride=1,
        **kwargs,
    )


def phbcc_2m(num_classes: int = 10, **kwargs) -> HBCCNet:
    """Progressive-capacity HBCC constrained to fewer than 2M CIFAR params."""

    model_kwargs: dict[str, Any] = {
        "use_coord": True,
        "embed_dims": [56, 88, 176, 232],
        "depths": [2, 2, 4, 1],
        "mlp_ratios": [3.0, 3.0, 3.0, 2.5],
        "heads": [2, 2, 4, 4],
        "head_dim": [16, 16, 16, 16],
        "proposals": [(2, 2), (2, 2), (2, 2), (2, 2)],
        "folds": [(4, 4), (2, 2), (1, 1), (1, 1)],
        "similarities": ["cosine", "cosine", "cosine", "cosine"],
        "stage_modes": ["hybrid", "hybrid", "cluster", "cluster"],
        "local_branches": ["lbpconv", "dwconv", "identity", "identity"],
        "local_ratios": [0.5, 0.5, 0.0, 0.0],
        "channel_shuffle": [True, True, False, False],
        "norm": "bn",
        "stem_patch_size": 3,
        "stem_stride": 2,
        "stem_padding": 1,
        "down_patch_size": 3,
        "down_stride": 2,
        "down_padding": 1,
        "drop_rate": 0.0,
        "drop_path_rate": 0.10,
    }
    model_kwargs.update(kwargs)
    return HBCCNet(num_classes=num_classes, **model_kwargs)


def coc_cifar_baseline(num_classes: int = 10, **kwargs) -> HBCCNet:
    """CoC-style CIFAR baseline: RGB+XY, cluster blocks in every stage."""

    return HBCCNet(
        num_classes=num_classes,
        embed_dims=[32, 64, 196, 320],
        depths=[2, 2, 3, 1],
        mlp_ratios=[4.0, 4.0, 3.0, 3.0],
        heads=[2, 4, 4, 4],
        head_dim=[16, 16, 24, 24],
        proposals=[(2, 2), (2, 2), (2, 2), (2, 2)],
        folds=[(4, 4), (2, 2), (1, 1), (1, 1)],
        stage_modes=["cluster", "cluster", "cluster", "cluster"],
        local_ratios=[0.0, 0.0, 0.0, 0.0],
        use_coord=True,
        **kwargs,
    )
