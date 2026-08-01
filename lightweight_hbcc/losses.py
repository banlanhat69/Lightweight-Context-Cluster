from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DistillationLoss(nn.Module):
    """Response-based KD from equation (16) of ``hbcc.pdf``.

    The PDF defines ``KL(student || teacher)`` after temperature scaling, so
    this implementation intentionally preserves that direction instead of the
    reversed argument order commonly used with ``torch.nn.functional.kl_div``.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.label_smoothing = float(label_smoothing)
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )

    def forward(
        self,
        student_logits: torch.Tensor,
        target: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target.ndim != 1:
            raise ValueError("No-augmentation training requires one hard class label per sample.")
        ce = F.cross_entropy(
            student_logits,
            target,
            label_smoothing=self.label_smoothing,
        )
        if teacher_logits is None:
            if self.alpha > 0.0:
                raise ValueError("kd_alpha is positive but teacher logits were not provided.")
            return ce
        if teacher_logits.shape != student_logits.shape:
            raise ValueError(
                "Teacher and student logits must have the same shape, got "
                f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}."
            )
        if self.alpha == 0.0:
            return ce

        temperature = self.temperature
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_log_probs = F.log_softmax(
            teacher_logits.detach() / temperature,
            dim=1,
        )
        student_probs = student_log_probs.exp()
        kl_student_teacher = (
            student_probs * (student_log_probs - teacher_log_probs)
        ).sum(dim=1).mean()
        kd = kl_student_teacher * (temperature * temperature)
        return (1.0 - self.alpha) * ce + self.alpha * kd
