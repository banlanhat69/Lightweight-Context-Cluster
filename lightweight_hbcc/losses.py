from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


DISTILLATION_METHODS = ("none", "reverse_kd", "standard", "dkd")


def _collapse_target_and_other(
    probabilities: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Collapse a class distribution into target and non-target probabilities."""

    target_probability = (probabilities * target_mask).sum(dim=1, keepdim=True)
    other_probability = (probabilities * ~target_mask).sum(dim=1, keepdim=True)
    return torch.cat((target_probability, other_probability), dim=1)


class DistillationLoss(nn.Module):
    """CE, standard KD, legacy reverse-KL KD, and DKD without augmentation.

    ``standard`` follows the usual soft-target direction
    ``KL(teacher || student)``. ``reverse_kd`` preserves equation (16) from
    ``hbcc.pdf`` for controlled comparison. ``dkd`` implements target-class
    and non-target-class knowledge distillation with an optional linear
    warmup of the distillation terms.
    """

    def __init__(
        self,
        method: str = "none",
        temperature: float = 4.0,
        alpha: float = 0.5,
        label_smoothing: float = 0.0,
        dkd_tckd_weight: float = 1.0,
        dkd_nckd_weight: float = 4.0,
        dkd_warmup_epochs: int = 20,
    ) -> None:
        super().__init__()
        self.method = str(method).lower()
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.label_smoothing = float(label_smoothing)
        self.dkd_tckd_weight = float(dkd_tckd_weight)
        self.dkd_nckd_weight = float(dkd_nckd_weight)
        self.dkd_warmup_epochs = int(dkd_warmup_epochs)
        self.epoch = 0

        if self.method not in DISTILLATION_METHODS:
            raise ValueError(
                f"Unknown distillation method {self.method!r}; "
                f"expected one of {DISTILLATION_METHODS}."
            )
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )
        if self.dkd_tckd_weight < 0.0 or self.dkd_nckd_weight < 0.0:
            raise ValueError("DKD weights must be non-negative.")
        if self.method == "dkd" and (
            self.dkd_tckd_weight == 0.0 and self.dkd_nckd_weight == 0.0
        ):
            raise ValueError("DKD requires at least one positive distillation weight.")
        if self.dkd_warmup_epochs < 0:
            raise ValueError("dkd_warmup_epochs must be non-negative.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _standard_kd(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        temperature = self.temperature
        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits.detach().float() / temperature, dim=1)
        return F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (temperature * temperature)

    def _reverse_kd(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        temperature = self.temperature
        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=1)
        teacher_log_probs = F.log_softmax(
            teacher_logits.detach().float() / temperature,
            dim=1,
        )
        student_probs = student_log_probs.exp()
        return (
            student_probs * (student_log_probs - teacher_log_probs)
        ).sum(dim=1).mean() * (temperature * temperature)

    def _dkd(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temperature = self.temperature
        student_scaled = student_logits.float() / temperature
        teacher_scaled = teacher_logits.detach().float() / temperature
        target_mask = F.one_hot(
            target,
            num_classes=student_logits.shape[1],
        ).bool()

        student_probs = _collapse_target_and_other(
            F.softmax(student_scaled, dim=1),
            target_mask,
        )
        teacher_probs = _collapse_target_and_other(
            F.softmax(teacher_scaled, dim=1),
            target_mask,
        )
        tckd = F.kl_div(
            student_probs.clamp_min(1e-12).log(),
            teacher_probs,
            reduction="batchmean",
        ) * (temperature * temperature)

        masked_student = student_scaled.masked_fill(target_mask, -1e9)
        masked_teacher = teacher_scaled.masked_fill(target_mask, -1e9)
        nckd = F.kl_div(
            F.log_softmax(masked_student, dim=1),
            F.softmax(masked_teacher, dim=1),
            reduction="batchmean",
        ) * (temperature * temperature)
        return tckd, nckd

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
            if self.method != "none":
                raise ValueError(
                    f"distillation method {self.method!r} requires teacher logits."
                )
            return ce
        if self.method == "none":
            raise ValueError("Teacher logits were provided while distillation method is 'none'.")
        if teacher_logits.shape != student_logits.shape:
            raise ValueError(
                "Teacher and student logits must have the same shape, got "
                f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}."
            )

        if self.method == "standard":
            kd = self._standard_kd(student_logits, teacher_logits)
            return (1.0 - self.alpha) * ce + self.alpha * kd
        if self.method == "reverse_kd":
            kd = self._reverse_kd(student_logits, teacher_logits)
            return (1.0 - self.alpha) * ce + self.alpha * kd

        tckd, nckd = self._dkd(student_logits, teacher_logits, target)
        if self.dkd_warmup_epochs == 0:
            warmup = 1.0
        else:
            warmup = min((self.epoch + 1) / self.dkd_warmup_epochs, 1.0)
        return ce + warmup * (
            self.dkd_tckd_weight * tckd
            + self.dkd_nckd_weight * nckd
        )
