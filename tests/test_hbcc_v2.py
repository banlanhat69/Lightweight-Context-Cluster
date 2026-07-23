from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from lightweight_hbcc.models import build_model
from lightweight_hbcc.models.cluster import ContextClusterOp


CATALOG_PATH = Path("configs/cifar_fair/model_catalog.yaml")


def _catalog_models() -> dict:
    return yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))["models"]


def _build_catalog_model(name: str):
    model_cfg = deepcopy(_catalog_models()[name]["model"])
    model_cfg["num_classes"] = 100
    return build_model({"model": model_cfg})


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("hbcc_medium_stable", 2_863_992),
        ("hbcc_medium_v2", 2_950_264),
    ],
)
def test_hbcc_v2_variants_keep_original_spatial_schedule_and_budget(
    name: str,
    params: int,
) -> None:
    model = _build_catalog_model(name).eval()
    feature_shapes: list[tuple[int, ...]] = []
    hooks = [
        module.register_forward_hook(
            lambda _module, _inputs, output: feature_shapes.append(tuple(output.shape[-2:]))
        )
        for module in [model.stem, *model.downsamples]
    ]
    try:
        with torch.no_grad():
            output = model(torch.randn(1, 3, 32, 32))
    finally:
        for hook in hooks:
            hook.remove()

    assert feature_shapes == [(16, 16), (8, 8), (4, 4), (2, 2)]
    assert output.shape == (1, 100)
    assert sum(parameter.numel() for parameter in model.parameters()) == params
    assert params < 3_000_000

    blocks = [block for stage in model.stages for block in stage.blocks]
    assert all(block.cluster.assignment_mode == "hard_st" for block in blocks)
    assert all(block.cluster.positive_similarity_scale for block in blocks)
    assert all(torch.allclose(block.layer_scale_1, torch.full_like(block.layer_scale_1, 0.001)) for block in blocks)
    assert model.stages[-1].blocks[0].cluster.proposal == (1, 1)


def test_stable_candidate_changes_only_cluster_optimization_controls() -> None:
    models = _catalog_models()
    baseline = deepcopy(models["hbcc_medium"]["model"])
    stable = deepcopy(models["hbcc_medium_stable"]["model"])

    assert stable.pop("assignment_modes") == ["hard_st"] * 4
    assert stable.pop("assignment_temperatures") == [1.0] * 4
    assert stable.pop("positive_similarity_scales") == [True] * 4
    assert stable.pop("layer_scale_init_values") == [0.001] * 4
    assert stable["proposals"][-1] == [1, 1]
    stable["proposals"][-1] = baseline["proposals"][-1]
    assert stable == baseline


def test_v2_only_widens_late_cluster_embeddings_over_stable() -> None:
    models = _catalog_models()
    stable = deepcopy(models["hbcc_medium_stable"]["model"])
    v2 = deepcopy(models["hbcc_medium_v2"]["model"])

    assert stable["head_dim"] == [16, 16, 16, 16]
    assert v2["head_dim"] == [16, 16, 20, 24]
    v2["head_dim"] = stable["head_dim"]
    assert v2 == stable


def test_positive_similarity_scale_cannot_reverse_assignment_order() -> None:
    op = ContextClusterOp(dim=4, positive_similarity_scale=True)
    with torch.no_grad():
        op.sim_alpha.fill_(-100.0)

    assert op.effective_sim_alpha().item() > 0.0


def test_hard_st_matches_hard_forward_and_adds_assignment_gradients() -> None:
    torch.manual_seed(7)
    hard = ContextClusterOp(
        dim=4,
        proposal=(2, 1),
        heads=1,
        head_dim=4,
        assignment_mode="hard",
    ).train()
    hard_st = ContextClusterOp(
        dim=4,
        proposal=(2, 1),
        heads=1,
        head_dim=4,
        assignment_mode="hard_st",
    ).train()
    hard_st.load_state_dict(hard.state_dict())

    x_hard = torch.randn(2, 4, 4, 4, requires_grad=True)
    x_st = x_hard.detach().clone().requires_grad_(True)
    output_hard = hard(x_hard)
    output_st = hard_st(x_st)
    output_hard.square().mean().backward()
    output_st.square().mean().backward()

    assert torch.allclose(output_hard, output_st, atol=1e-6, rtol=1e-5)
    assert x_hard.grad is not None
    assert x_st.grad is not None
    assert not torch.allclose(x_hard.grad, x_st.grad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"assignment_mode": "unknown"},
        {"assignment_temperature": 0.0},
    ],
)
def test_invalid_assignment_configuration_fails_fast(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ContextClusterOp(dim=4, **kwargs)
