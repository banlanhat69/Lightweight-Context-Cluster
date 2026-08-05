from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


DISTILLATION_METHODS = ("none", "reverse_kd", "standard", "dkd", "dkd_at")


def _collapse_target_and_other(
    probabilities: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Collapse a class distribution into target and non-target probabilities."""

    target_probability = (probabilities * target_mask).sum(dim=1, keepdim=True)
    other_probability = (probabilities * ~target_mask).sum(dim=1, keepdim=True)
    return torch.cat((target_probability, other_probability), dim=1)


class DistillationLoss(nn.Module):
    """CE (hard or mixed targets), standard KD, reverse-KL KD, and DKD.

    ``standard`` follows the usual soft-target direction
    ``KL(teacher || student)``. ``reverse_kd`` preserves equation (16) from
    ``hbcc.pdf`` for controlled comparison. ``dkd`` implements target-class
    and non-target-class knowledge distillation with an optional linear
    warmup of the distillation terms. ``dkd_at`` adds multi-stage attention
    transfer without requiring equal teacher/student channel counts.
    """

    def __init__(
        self,
        method: str = "none",
        temperature: float = 4.0,
        alpha: float = 0.5,
        label_smoothing: float = 0.0,
        dkd_tckd_weight: float = 1.0,
        dkd_nckd_weight: float = 4.0,
        dkd_scale: float = 1.0,
        dkd_warmup_epochs: int = 20,
        feature_kd_weight: float = 0.25,
        feature_kd_stages: list[int] | tuple[int, ...] = (2, 3, 4),
        feature_kd_warmup_epochs: int = 20,
    ) -> None:
        super().__init__()
        self.method = str(method).lower()
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.label_smoothing = float(label_smoothing)
        self.dkd_tckd_weight = float(dkd_tckd_weight)
        self.dkd_nckd_weight = float(dkd_nckd_weight)
        self.dkd_scale = float(dkd_scale)
        self.dkd_warmup_epochs = int(dkd_warmup_epochs)
        self.feature_kd_weight = float(feature_kd_weight)
        self.feature_kd_stages = tuple(int(stage) for stage in feature_kd_stages)
        self.feature_kd_warmup_epochs = int(feature_kd_warmup_epochs)
        self.epoch = 0
        self.last_components: dict[str, torch.Tensor] = {}

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
        if self.method in {"dkd", "dkd_at"} and (
            self.dkd_tckd_weight == 0.0 and self.dkd_nckd_weight == 0.0
        ):
            raise ValueError("DKD requires at least one positive distillation weight.")
        if self.dkd_scale < 0.0:
            raise ValueError("dkd_scale must be non-negative.")
        if self.method in {"dkd", "dkd_at"} and self.dkd_scale == 0.0:
            raise ValueError("DKD requires dkd_scale > 0.")
        if self.dkd_warmup_epochs < 0:
            raise ValueError("dkd_warmup_epochs must be non-negative.")
        if not self.feature_kd_stages:
            raise ValueError("feature_kd_stages must contain at least one stage.")
        if len(self.feature_kd_stages) != len(set(self.feature_kd_stages)):
            raise ValueError("feature_kd_stages must not contain duplicates.")
        if any(stage < 1 or stage > 4 for stage in self.feature_kd_stages):
            raise ValueError("feature_kd_stages must be in the inclusive range [1, 4].")
        if self.feature_kd_weight < 0.0:
            raise ValueError("feature_kd_weight must be non-negative.")
        if self.method == "dkd_at" and self.feature_kd_weight == 0.0:
            raise ValueError("dkd_at requires feature_kd_weight > 0.")
        if self.feature_kd_warmup_epochs < 0:
            raise ValueError("feature_kd_warmup_epochs must be non-negative.")

    @property
    def requires_feature_distillation(self) -> bool:
        return self.method == "dkd_at"

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _warmup(self, epochs: int) -> float:
        if epochs == 0:
            return 1.0
        return min((self.epoch + 1) / epochs, 1.0)

    @staticmethod
    def _attention_map(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(
                "Attention transfer expects NCHW feature maps, got "
                f"shape {tuple(feature.shape)}."
            )
        attention = feature.float().pow(2).sum(dim=1).flatten(1)
        return F.normalize(attention, p=2, dim=1, eps=1e-12)

    def _attention_transfer(
        self,
        student_features: tuple[torch.Tensor, ...] | list[torch.Tensor],
        teacher_features: tuple[torch.Tensor, ...] | list[torch.Tensor],
    ) -> torch.Tensor:
        if len(student_features) < 4 or len(teacher_features) < 4:
            raise ValueError(
                "Multi-stage attention transfer requires four teacher and student features."
            )
        stage_losses: list[torch.Tensor] = []
        for stage in self.feature_kd_stages:
            student_feature = student_features[stage - 1]
            teacher_feature = teacher_features[stage - 1].detach()
            if student_feature.shape[0] != teacher_feature.shape[0]:
                raise ValueError(f"Feature batch mismatch at stage {stage}.")
            if student_feature.shape[-2:] != teacher_feature.shape[-2:]:
                raise ValueError(
                    f"Feature spatial mismatch at stage {stage}: "
                    f"student={tuple(student_feature.shape[-2:])}, "
                    f"teacher={tuple(teacher_feature.shape[-2:])}."
                )
            student_attention = self._attention_map(student_feature)
            teacher_attention = self._attention_map(teacher_feature)
            stage_losses.append(
                (student_attention - teacher_attention).pow(2).sum(dim=1).mean()
            )
        return torch.stack(stage_losses).mean()

    def _record(self, **components: torch.Tensor) -> None:
        self.last_components = {
            name: value.detach() for name, value in components.items()
        }

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
        student_features: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
        teacher_features: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if target.ndim == 1:
            ce = F.cross_entropy(
                student_logits,
                target,
                label_smoothing=self.label_smoothing,
            )
        elif target.ndim == 2 and target.shape == student_logits.shape:
            if teacher_logits is not None or self.method != "none":
                raise ValueError("Probability targets are supported only for CE training.")
            probabilities = target.float()
            if self.label_smoothing > 0.0:
                probabilities = (
                    probabilities * (1.0 - self.label_smoothing)
                    + self.label_smoothing / student_logits.shape[1]
                )
            ce = -(
                probabilities * F.log_softmax(student_logits.float(), dim=1)
            ).sum(dim=1).mean()
        else:
            raise ValueError(
                "Target must contain hard class indices or class probabilities, got "
                f"shape {tuple(target.shape)}."
            )
        if teacher_logits is None:
            if self.method != "none":
                raise ValueError(
                    f"distillation method {self.method!r} requires teacher logits."
                )
            self._record(ce=ce)
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
            total = (1.0 - self.alpha) * ce + self.alpha * kd
            self._record(ce=ce, kd=kd)
            return total
        if self.method == "reverse_kd":
            kd = self._reverse_kd(student_logits, teacher_logits)
            total = (1.0 - self.alpha) * ce + self.alpha * kd
            self._record(ce=ce, kd=kd)
            return total

        tckd, nckd = self._dkd(student_logits, teacher_logits, target)
        dkd = self.dkd_tckd_weight * tckd + self.dkd_nckd_weight * nckd
        total = ce + self.dkd_scale * self._warmup(self.dkd_warmup_epochs) * dkd
        components = {"ce": ce, "tckd": tckd, "nckd": nckd, "dkd": dkd}
        if self.method == "dkd_at":
            if student_features is None or teacher_features is None:
                raise ValueError("dkd_at requires teacher and student feature maps.")
            attention = self._attention_transfer(student_features, teacher_features)
            total = total + (
                self.feature_kd_weight
                * self._warmup(self.feature_kd_warmup_epochs)
                * attention
            )
            components["attention"] = attention
        self._record(**components)
        return total
