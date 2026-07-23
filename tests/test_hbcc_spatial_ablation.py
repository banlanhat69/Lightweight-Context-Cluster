from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from lightweight_hbcc.models import build_model


CATALOG_PATH = Path("configs/cifar_fair/model_catalog.yaml")


def _catalog_models() -> dict:
    return yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))["models"]


def _build_catalog_model(name: str):
    model_cfg = deepcopy(_catalog_models()[name]["model"])
    model_cfg["num_classes"] = 100
    return build_model({"model": model_cfg})


@pytest.mark.parametrize(
    "name",
    ["hbcc_small_keep4", "hbcc_medium_keep4", "hbcc_medium_keep4_late_hybrid"],
)
def test_keep4_variants_preserve_final_spatial_resolution(name: str) -> None:
    model = _build_catalog_model(name).eval()
    feature_shapes: list[tuple[int, ...]] = []
    modules = [model.stem, *model.downsamples]
    hooks = [
        module.register_forward_hook(
            lambda _module, _inputs, output: feature_shapes.append(tuple(output.shape))
        )
        for module in modules
    ]
    try:
        with torch.no_grad():
            output = model(torch.randn(1, 3, 32, 32))
    finally:
        for hook in hooks:
            hook.remove()

    assert [shape[-2:] for shape in feature_shapes] == [
        (16, 16),
        (8, 8),
        (4, 4),
        (4, 4),
    ]
    assert model.down_strides == [2, 2, 1]
    assert [module.stride for module in model.downsamples] == [2, 2, 1]
    assert output.shape == (1, 100)


@pytest.mark.parametrize(
    ("baseline", "variant"),
    [
        ("hbcc_small", "hbcc_small_keep4"),
        ("hbcc_medium", "hbcc_medium_keep4"),
    ],
)
def test_keep4_is_a_single_variable_spatial_ablation(baseline: str, variant: str) -> None:
    models = _catalog_models()
    baseline_cfg = deepcopy(models[baseline]["model"])
    variant_cfg = deepcopy(models[variant]["model"])

    assert "down_stride" not in baseline_cfg
    assert variant_cfg.pop("down_stride") == [2, 2, 1]
    assert variant_cfg == baseline_cfg


def test_late_hybrid_is_incremental_to_medium_keep4() -> None:
    models = _catalog_models()
    keep4 = deepcopy(models["hbcc_medium_keep4"]["model"])
    late_hybrid = deepcopy(models["hbcc_medium_keep4_late_hybrid"]["model"])

    assert late_hybrid["stage_modes"][-1] == "hybrid"
    assert late_hybrid["local_branches"][-1] == "dwconv"
    assert late_hybrid["local_ratios"][-1] == 0.5
    assert late_hybrid["channel_shuffle"][-1] is True

    for key in ("stage_modes", "local_branches", "local_ratios", "channel_shuffle"):
        late_hybrid[key][-1] = keep4[key][-1]
    assert late_hybrid == keep4


def test_downsample_schedule_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="Expected list of length 3"):
        build_model(
            {
                "model": {
                    "name": "hbcc",
                    "num_classes": 10,
                    "down_stride": [2, 1],
                }
            }
        )


def test_scalar_downsample_schedule_remains_backward_compatible() -> None:
    model = build_model({"model": {"name": "hbcc", "num_classes": 10}}).eval()

    assert model.down_strides == [2, 2, 2]
    assert [module.stride for module in model.downsamples] == [2, 2, 2]
    assert [stage.blocks[0].mode for stage in model.stages] == [
        "local",
        "hybrid",
        "cluster",
        "cluster",
    ]
    with torch.no_grad():
        assert model(torch.randn(1, 3, 32, 32)).shape == (1, 10)
