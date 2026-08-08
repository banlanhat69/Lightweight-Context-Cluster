from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


class MixupCutmix:
    """Batch-level Mixup/CutMix with probability targets."""

    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 0.5,
        probability: float = 0.8,
        switch_probability: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.num_classes = int(num_classes)
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.probability = float(probability)
        self.switch_probability = float(switch_probability)
        self.numpy_rng = np.random.default_rng(int(seed))
        self.torch_generator = torch.Generator().manual_seed(int(seed) + 1)
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if self.mixup_alpha < 0.0 or self.cutmix_alpha < 0.0:
            raise ValueError("Mixup/CutMix alpha values must be non-negative.")
        if self.mixup_alpha == 0.0 and self.cutmix_alpha == 0.0:
            raise ValueError("At least one of Mixup or CutMix must be enabled.")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1].")
        if not 0.0 <= self.switch_probability <= 1.0:
            raise ValueError("switch_probability must be in [0, 1].")

    def state_dict(self) -> dict[str, Any]:
        """Return both RNG states so an interrupted run can resume exactly."""

        return {
            "numpy_rng": copy.deepcopy(self.numpy_rng.bit_generator.state),
            "torch_generator": self.torch_generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore RNG states saved by :meth:`state_dict`."""

        if "numpy_rng" not in state or "torch_generator" not in state:
            raise KeyError("MixupCutmix state must contain NumPy and Torch RNG states.")
        self.numpy_rng.bit_generator.state = copy.deepcopy(state["numpy_rng"])
        self.torch_generator.set_state(state["torch_generator"].cpu())

    def _sample_beta(self, alpha: float) -> float:
        if alpha <= 0.0:
            return 1.0
        return float(self.numpy_rng.beta(alpha, alpha))

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.shape[0] < 2 or self.numpy_rng.random() > self.probability:
            return images, targets
        permutation = torch.randperm(
            images.shape[0],
            generator=self.torch_generator,
        ).to(images.device)
        target_a = F.one_hot(targets, self.num_classes).to(images.dtype)
        target_b = target_a[permutation]
        use_cutmix = (
            self.cutmix_alpha > 0.0
            and (
                self.mixup_alpha == 0.0
                or self.numpy_rng.random() < self.switch_probability
            )
        )
        if not use_cutmix:
            lam = self._sample_beta(self.mixup_alpha)
            mixed_images = images * lam + images[permutation] * (1.0 - lam)
            return mixed_images, target_a * lam + target_b * (1.0 - lam)

        lam = self._sample_beta(self.cutmix_alpha)
        height, width = images.shape[-2:]
        cut_ratio = math.sqrt(1.0 - lam)
        cut_height = int(height * cut_ratio)
        cut_width = int(width * cut_ratio)
        center_y = int(self.numpy_rng.integers(height))
        center_x = int(self.numpy_rng.integers(width))
        y1 = max(center_y - cut_height // 2, 0)
        y2 = min(center_y + cut_height // 2, height)
        x1 = max(center_x - cut_width // 2, 0)
        x2 = min(center_x + cut_width // 2, width)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        adjusted_lam = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
        return (
            mixed_images,
            target_a * adjusted_lam + target_b * (1.0 - adjusted_lam),
        )
