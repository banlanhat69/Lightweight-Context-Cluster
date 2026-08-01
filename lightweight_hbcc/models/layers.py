from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class GroupNorm1(nn.GroupNorm):
    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(1, num_channels, **kwargs)


def make_norm(name: str | None, channels: int) -> nn.Module:
    if name is None or name.lower() in {"none", "identity"}:
        return nn.Identity()
    name = name.lower()
    if name in {"bn", "batchnorm", "batchnorm2d"}:
        return nn.BatchNorm2d(channels)
    if name in {"gn", "groupnorm", "layernorm2d"}:
        return GroupNorm1(channels)
    raise ValueError(f"Unsupported norm layer: {name}")


def make_act(name: str | None) -> nn.Module:
    if name is None or name.lower() in {"none", "identity"}:
        return nn.Identity()
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    if name in {"hswish", "hardswish"}:
        return nn.Hardswish(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        norm: str | None = "bn",
        act: str | None = "gelu",
        bias: bool | None = None,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        if bias is None:
            bias = norm is None or str(norm).lower() in {"none", "identity"}
        super().__init__(
            nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding, groups=groups, bias=bias),
            make_norm(norm, out_chans),
            make_act(act),
        )


class CoordinateAugment(nn.Module):
    """Append normalized xy coordinates in [-0.5, 0.5]."""

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        b, _, h, w = x.shape
        yy = torch.linspace(0.0, 1.0, h, device=x.device, dtype=x.dtype) - 0.5
        xx = torch.linspace(0.0, 1.0, w, device=x.device, dtype=x.dtype) - 0.5
        y_grid, x_grid = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack((x_grid, y_grid), dim=0).expand(b, -1, -1, -1)
        return torch.cat((x, coords), dim=1)


class PointReducer(nn.Module):
    """Anchor/k-neighbor reduction implemented as a conv projection."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        patch_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        norm: str | None = "bn",
        act: str | None = "gelu",
    ) -> None:
        super().__init__()
        self.proj = ConvNormAct(in_chans, out_chans, patch_size, stride, padding, norm=norm, act=act)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SignSTE(torch.autograd.Function):
    """Sign with hard-tanh straight-through gradients."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        grad = grad_output.clone()
        return grad * (x.abs() <= 1).to(grad.dtype)


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


def pairwise_cosine(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return torch.matmul(x1, x2.transpose(-2, -1))


def pairwise_hamming_simulated(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    x1b = sign_ste(x1)
    x2b = sign_ste(x2)
    sim = torch.matmul(x1b, x2b.transpose(-2, -1)) / x1.shape[-1]
    if scale is not None:
        sim = sim * F.softplus(scale).view(1, 1, 1)
    return sim


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.shape
    if groups <= 1 or c % groups != 0:
        return x
    x = x.reshape(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(b, c, h, w)


def make_lbp_filters(channels: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    filters = torch.zeros(8, 1, 3, 3, dtype=dtype)
    neighbors = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (2, 1),
        (2, 0),
        (1, 0),
    ]
    for idx, (row, col) in enumerate(neighbors):
        filters[idx, 0, row, col] = 1.0
        filters[idx, 0, 1, 1] = -1.0
    return filters.repeat(channels, 1, 1, 1)


class IdentityBranch(nn.Module):
    def __init__(self, in_chans: int, out_chans: int) -> None:
        super().__init__()
        self.proj = nn.Identity() if in_chans == out_chans else nn.Conv2d(in_chans, out_chans, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DWConvBranch(nn.Sequential):
    def __init__(self, in_chans: int, out_chans: int, norm: str = "bn", act: str = "gelu") -> None:
        super().__init__(
            ConvNormAct(in_chans, in_chans, 3, 1, 1, groups=in_chans, norm=norm, act=act),
            ConvNormAct(in_chans, out_chans, 1, 1, 0, norm=norm, act=act),
        )


class LBPConvBranch(nn.Module):
    """Fixed sparse LBP-like filters followed by a pointwise projection."""

    def __init__(self, in_chans: int, out_chans: int, norm: str = "bn", act: str = "gelu") -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_chans, in_chans * 8, 3, 1, 1, groups=in_chans, bias=False)
        with torch.no_grad():
            self.depthwise.weight.copy_(make_lbp_filters(in_chans, self.depthwise.weight.dtype))
        self.depthwise.weight.requires_grad_(False)
        self.post = nn.Sequential(
            make_norm(norm, in_chans * 8),
            make_act(act),
            ConvNormAct(in_chans * 8, out_chans, 1, 1, 0, norm=norm, act=act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post(self.depthwise(x))


def make_local_branch(name: str, in_chans: int, out_chans: int, norm: str = "bn", act: str = "gelu") -> nn.Module:
    name = name.lower()
    factories: dict[str, Callable[[], nn.Module]] = {
        "identity": lambda: IdentityBranch(in_chans, out_chans),
        "dwconv": lambda: DWConvBranch(in_chans, out_chans, norm=norm, act=act),
        "lbpconv": lambda: LBPConvBranch(in_chans, out_chans, norm=norm, act=act),
    }
    if name not in factories:
        raise ValueError(f"Unsupported local branch: {name}")
    return factories[name]()
