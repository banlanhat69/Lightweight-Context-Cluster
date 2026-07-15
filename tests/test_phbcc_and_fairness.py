from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from lightweight_hbcc import data
from lightweight_hbcc.config import deep_update, load_config, save_config
from lightweight_hbcc.engine import apply_mixup_cutmix
from lightweight_hbcc.models import build_model
from tools.run_fair_comparison import (
    CANONICAL_EPOCHS,
    CANONICAL_PROTOCOL_NAME,
    CORE_MODELS,
    DEFAULT_SEEDS,
    _overrides,
    completed_benchmark_matches,
    completed_run_matches,
    config_paths,
    ensure_run_target,
    parse_args,
    run_suffix,
    validate_controlled_configs,
    validate_models,
    validate_seeds,
)
from tools.train import is_controlled_comparison, resolve_seed_config


FAIR_MODELS = [
    "resnet18",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "coc_baseline",
    "hbcc_small",
    "hbcc_small_plus",
    "phbcc_2m",
    "hbcc_medium",
]


class PatternDataset(Dataset):
    def __init__(self, transform, size: int = 24) -> None:
        self.transform = transform
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        yy, xx = np.mgrid[0:32, 0:32]
        image = np.stack(
            (
                (xx * 7 + index * 11) % 256,
                (yy * 9 + index * 13) % 256,
                ((xx + yy) * 5 + index * 17) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        return self.transform(Image.fromarray(image)), index


def test_phbcc_2m_architecture_contract() -> None:
    cfg = load_config("configs/hbcc_accuracy_phbcc_2m.yaml")
    model_cfg = cfg["model"]

    assert model_cfg["name"] == "phbcc_2m"
    assert model_cfg["use_coord"] is True
    assert model_cfg["embed_dims"] == [56, 88, 176, 232]
    assert model_cfg["depths"] == [2, 2, 4, 1]
    assert model_cfg["mlp_ratios"] == [3.0, 3.0, 3.0, 2.5]
    assert model_cfg["heads"] == [2, 2, 4, 4]
    assert model_cfg["head_dim"] == [16, 16, 16, 16]
    assert model_cfg["proposals"] == [[2, 2], [2, 2], [2, 2], [2, 2]]
    assert model_cfg["folds"] == [[4, 4], [2, 2], [1, 1], [1, 1]]
    assert model_cfg["similarities"] == ["cosine", "cosine", "cosine", "cosine"]
    assert model_cfg["stage_modes"] == ["hybrid", "hybrid", "cluster", "cluster"]
    assert model_cfg["local_branches"] == ["lbpconv", "dwconv", "identity", "identity"]
    assert model_cfg["local_ratios"] == [0.5, 0.5, 0.0, 0.0]
    assert model_cfg["channel_shuffle"] == [True, True, False, False]
    assert model_cfg["norm"] == "bn"
    assert model_cfg["stem_patch_size"] == 3
    assert model_cfg["stem_stride"] == 2
    assert model_cfg["stem_padding"] == 1
    assert model_cfg["down_patch_size"] == 3
    assert model_cfg["down_stride"] == 2
    assert model_cfg["down_padding"] == 1
    assert model_cfg["drop_rate"] == 0.0
    assert model_cfg["drop_path_rate"] == 0.10


def test_phbcc_2m_factory_defaults_are_first_class_and_under_budget() -> None:
    model = build_model({"model": {"name": "phbcc_2m", "num_classes": 10}})
    assert model.embed_dims == [56, 88, 176, 232]
    assert model.depths == [2, 2, 4, 1]
    assert model.use_coord is True
    assert [block.mode for stage in model.stages for block in stage.blocks] == [
        "hybrid",
        "hybrid",
        "hybrid",
        "hybrid",
        "cluster",
        "cluster",
        "cluster",
        "cluster",
        "cluster",
    ]
    first_blocks = [stage.blocks[0] for stage in model.stages]
    assert [block.cluster.similarity for block in first_blocks] == ["cosine"] * 4
    assert [(block.cluster.heads, block.cluster.head_dim) for block in first_blocks] == [
        (2, 16),
        (2, 16),
        (4, 16),
        (4, 16),
    ]
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_953_056


@pytest.mark.parametrize(
    ("path", "num_classes", "params_total", "params_trainable"),
    [
        ("configs/fair_comparison/cifar10/phbcc_2m.yaml", 10, 1_953_056, 1_949_024),
        ("configs/fair_comparison/cifar100/phbcc_2m.yaml", 100, 1_974_026, 1_969_994),
    ],
)
def test_phbcc_2m_forward_shapes_and_parameter_budget(
    path: str,
    num_classes: int,
    params_total: int,
    params_trainable: int,
) -> None:
    cfg = load_config(path)
    model = build_model(cfg).eval()
    feature_shapes: list[tuple[int, ...]] = []
    modules = [model.stem, *model.downsamples]
    hooks = [
        module.register_forward_hook(lambda _module, _inputs, output: feature_shapes.append(tuple(output.shape)))
        for module in modules
    ]
    try:
        with torch.no_grad():
            output = model(torch.randn(1, 3, 32, 32))
    finally:
        for hook in hooks:
            hook.remove()

    assert feature_shapes == [
        (1, 56, 16, 16),
        (1, 88, 8, 8),
        (1, 176, 4, 4),
        (1, 232, 2, 2),
    ]
    assert output.shape == (1, num_classes)
    assert sum(parameter.numel() for parameter in model.parameters()) == params_total
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == params_trainable
    assert params_total < 2_000_000


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
def test_fair_comparison_configs_share_the_exact_controlled_recipe(dataset: str) -> None:
    expected = load_config("configs/recipes/cifar_coc_paper_inspired.yaml")
    if dataset == "cifar100":
        expected = deep_update(expected, {"data": {"name": "cifar100"}})

    transform_signatures: set[str] = set()
    for model_name in FAIR_MODELS:
        cfg = load_config(f"configs/fair_comparison/{dataset}/{model_name}.yaml")
        assert cfg["protocol"] == expected["protocol"], model_name
        assert cfg["data"] == expected["data"], model_name
        assert cfg["train"] == expected["train"], model_name
        assert cfg["model"]["num_classes"] == (100 if dataset == "cifar100" else 10)
        transform_signatures.add(repr(data._transforms(dataset, True, True, cfg["data"])))

    assert len(transform_signatures) == 1
    assert "RandAugment" not in next(iter(transform_signatures))
    assert expected["protocol"]["name"] == "cifar_coc_paper_inspired_300e_v1"
    assert expected["data"]["randaugment"] == {"enabled": False, "num_ops": 2, "magnitude": 9}
    assert expected["data"]["random_erasing"] == {
        "p": 0.25,
        "scale": [0.02, 0.3333333333],
        "ratio": [0.3, 3.3],
        "value": "random",
    }
    assert expected["train"]["epochs"] == CANONICAL_EPOCHS == 300
    assert expected["train"]["warmup_epochs"] == 5
    assert expected["train"]["mixup_alpha"] == 0.8
    assert expected["train"]["cutmix_alpha"] == 1.0
    assert expected["train"]["cutmix_prob"] == 0.5
    assert expected["train"]["kd_alpha"] == 0.0
    assert list(CORE_MODELS) == [
        "resnet18",
        "mobilenet_v2",
        "shufflenet_v2_x1_0",
        "coc_baseline",
        "hbcc_small",
        "hbcc_medium",
    ]
    assert "phbcc_2m" not in CORE_MODELS
    validated = validate_controlled_configs(dataset, config_paths(dataset, FAIR_MODELS))
    assert set(validated) == set(FAIR_MODELS)


def test_hbcc_family_uses_the_same_drop_path_in_controlled_comparison() -> None:
    for name in ["hbcc_small", "hbcc_small_plus", "phbcc_2m", "hbcc_medium"]:
        cfg = load_config(f"configs/fair_comparison/cifar10/{name}.yaml")
        assert cfg["model"]["drop_path_rate"] == 0.10


def test_hbcc_small_plus_changes_only_stage3_depth() -> None:
    small = load_config("configs/fair_comparison/cifar10/hbcc_small.yaml")["model"]
    small_plus = load_config("configs/fair_comparison/cifar10/hbcc_small_plus.yaml")["model"]
    assert small["depths"] == [2, 2, 3, 1]
    assert small_plus["depths"] == [2, 2, 4, 1]
    normalized_plus = {**small_plus, "depths": small["depths"]}
    assert normalized_plus == small


@pytest.mark.parametrize(
    ("path", "params_total", "params_trainable"),
    [
        ("configs/fair_comparison/cifar10/hbcc_small_plus.yaml", 1_724_828, 1_721_372),
        ("configs/fair_comparison/cifar100/hbcc_small_plus.yaml", 1_745_078, 1_741_622),
    ],
)
def test_hbcc_small_plus_parameter_contract(path: str, params_total: int, params_trainable: int) -> None:
    model = build_model(load_config(path))
    assert sum(parameter.numel() for parameter in model.parameters()) == params_total
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == params_trainable


@pytest.mark.parametrize("cutmix_prob", [0.0, 1.0])
def test_mixup_cutmix_has_an_independent_reproducible_rng(cutmix_prob: float) -> None:
    images = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
    target = torch.tensor([0, 1, 2, 3])
    train_recipe = load_config("configs/recipes/cifar_coc_paper_inspired.yaml")["train"]

    def apply_once() -> tuple[torch.Tensor, torch.Tensor]:
        return apply_mixup_cutmix(
            images,
            target,
            num_classes=10,
            label_smoothing=float(train_recipe["label_smoothing"]),
            mixup_alpha=float(train_recipe["mixup_alpha"]),
            cutmix_alpha=float(train_recipe["cutmix_alpha"]),
            cutmix_prob=cutmix_prob,
            rng=random.Random(123),
            torch_generator=torch.Generator().manual_seed(456),
        )

    images_a, target_a = apply_once()
    _ = torch.rand(1000)
    images_b, target_b = apply_once()
    assert torch.equal(images_a, images_b)
    assert torch.equal(target_a, target_b)


def test_dataloader_shuffle_uses_loader_seed() -> None:
    cfg = {
        "name": "fake",
        "fake_train_size": 32,
        "fake_val_size": 8,
        "batch_size": 8,
        "workers": 0,
        "drop_last": False,
        "loader_seed": 123,
    }
    loader_a, _, _ = data.build_loaders(deepcopy(cfg))
    loader_b, _, _ = data.build_loaders(deepcopy(cfg))
    targets_a = torch.cat([targets for _, targets in loader_a])
    targets_b = torch.cat([targets for _, targets in loader_b])
    assert torch.equal(targets_a, targets_b)


def test_loader_seed_defaults_to_train_seed_and_data_override_wins() -> None:
    cfg = {"train": {"seed": 17}, "data": {}}
    assert resolve_seed_config(cfg) == (17, 17, False)
    assert cfg["data"]["loader_seed"] == 17

    overridden = {"train": {"seed": 17}, "data": {"loader_seed": 999}}
    assert resolve_seed_config(overridden) == (17, 999, False)
    assert overridden["data"]["loader_seed"] == 999


def test_worker_seed_pairs_stochastic_augmentation_batches() -> None:
    recipe = load_config("configs/recipes/cifar_coc_paper_inspired.yaml")

    def collect(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        transform = data._transforms("cifar10", True, True, recipe["data"])
        loader = DataLoader(
            PatternDataset(transform),
            batch_size=6,
            shuffle=True,
            num_workers=2,
            worker_init_fn=data._seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )
        batches = list(loader)
        return torch.cat([images for images, _ in batches]), torch.cat([targets for _, targets in batches])

    images_a, targets_a = collect(123)
    images_b, targets_b = collect(123)
    images_c, targets_c = collect(456)
    assert torch.equal(images_a, images_b)
    assert torch.equal(targets_a, targets_b)
    assert not (torch.equal(images_a, images_c) and torch.equal(targets_a, targets_c))


def test_runner_marks_short_runs_noncanonical_and_rejects_overwrites(tmp_path: Path) -> None:
    smoke_args = SimpleNamespace(data_root="data", smoke=True, epochs=None)
    overrides = _overrides(smoke_args, "smoke_run", 17)
    assert "data.loader_seed=17" in overrides
    assert "train.seed=17" in overrides
    assert "train.epochs=1" in overrides
    assert "protocol.effective_epochs=1" in overrides
    assert f"protocol.name={CANONICAL_PROTOCOL_NAME}_smoke" in overrides
    assert "protocol.canonical=false" in overrides
    assert run_suffix(smoke_args) == "_smoke"

    canonical_args = SimpleNamespace(data_root="data", smoke=False, epochs=CANONICAL_EPOCHS)
    canonical_overrides = _overrides(canonical_args, "canonical_run", 17)
    assert "protocol.canonical=false" not in canonical_overrides
    assert run_suffix(canonical_args) == ""
    short_args = SimpleNamespace(data_root="data", smoke=False, epochs=30)
    assert run_suffix(short_args) == "_e30"
    old_200_args = SimpleNamespace(data_root="data", smoke=False, epochs=200)
    old_200_overrides = _overrides(old_200_args, "old_200_run", 17)
    assert run_suffix(old_200_args) == "_e200"
    assert f"protocol.name={CANONICAL_PROTOCOL_NAME}_epochs200" in old_200_overrides
    assert "protocol.canonical=false" in old_200_overrides
    validate_seeds([17])
    with pytest.raises(ValueError, match="Duplicate seeds"):
        validate_seeds([17, 17])
    validate_models(["resnet18", "hbcc_small"])
    with pytest.raises(ValueError, match="Duplicate models"):
        validate_models(["hbcc_small", "hbcc_small"])

    existing = tmp_path / "existing_run"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ensure_run_target(str(tmp_path), "existing_run", force=False, dry_run=False)
    ensure_run_target(str(tmp_path), "existing_run", force=True, dry_run=False)
    ensure_run_target(str(tmp_path), "existing_run", force=False, dry_run=True)


def test_runner_defaults_to_the_old_hbcc_6_run_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_fair_comparison.py"])
    args = parse_args()
    assert tuple(args.models) == CORE_MODELS
    assert tuple(args.seeds) == DEFAULT_SEEDS == (17,)
    assert len(args.models) * len(args.seeds) == 6
    assert args.output == "runs_fair_paper_inspired_300e"
    assert args.benchmark_output == "results/fair_paper_inspired_300e"
    assert args.skip_completed is True


def test_single_seed_notebook_does_not_report_fake_uncertainty() -> None:
    notebook = json.loads(Path("notebooks/hbcc_fair_training.ipynb").read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "accuracy_seed17" in code
    assert "single_seed_differences" in code
    assert "std_acc1" not in code
    assert "ci95_" not in code
    assert "runs_fair_paper_inspired_300e" in code


def test_completed_run_reuse_requires_matching_metadata_and_artifacts(tmp_path: Path) -> None:
    experiment_name = "fair_cifar10_hbcc_small_seed17"
    args = SimpleNamespace(
        output=str(tmp_path / "runs"),
        benchmark_output=str(tmp_path / "benchmarks"),
        dataset="cifar10",
        data_root="data",
        smoke=False,
        epochs=None,
    )
    run_dir = Path(args.output) / experiment_name
    run_dir.mkdir(parents=True)
    completed_cfg = deep_update(
        load_config("configs/fair_comparison/cifar10/hbcc_small.yaml"),
        {
            "experiment": {"name": experiment_name},
            "data": {"root": "data", "loader_seed": 17},
            "train": {"seed": 17},
        },
    )
    save_config(completed_cfg, run_dir / "config.yaml")
    (run_dir / "best.pth").write_bytes(b"checkpoint")
    (run_dir / "test_metrics.json").write_text(json.dumps({"test_acc1": 90.0}), encoding="utf-8")
    assert completed_run_matches(args, experiment_name, 17)
    assert not completed_run_matches(args, experiment_name, 29)

    drifted_cfg = deep_update(completed_cfg, {"data": {"random_erasing": {"p": 0.0}}})
    save_config(drifted_cfg, run_dir / "config.yaml")
    assert not completed_run_matches(args, experiment_name, 17)
    save_config(completed_cfg, run_dir / "config.yaml")

    benchmark_dir = Path(args.benchmark_output) / args.dataset
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / f"{experiment_name}.json").write_text(
        json.dumps(
            {
                "config_id": experiment_name,
                "protocol_name": CANONICAL_PROTOCOL_NAME,
                "effective_epochs": CANONICAL_EPOCHS,
                "latency_ms_b1": 1.0,
            }
        ),
        encoding="utf-8",
    )
    assert completed_benchmark_matches(args, experiment_name)


def test_paper_inspired_protocol_is_still_resume_guarded() -> None:
    recipe = load_config("configs/recipes/cifar_coc_paper_inspired.yaml")
    assert is_controlled_comparison(recipe)


def test_config_base_inheritance_and_cycle_detection(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("data:\n  name: cifar10\ntrain:\n  epochs: 300\n", encoding="utf-8")
    child.write_text("_base_: base.yaml\ntrain:\n  seed: 17\n", encoding="utf-8")
    assert load_config(child) == {
        "data": {"name": "cifar10"},
        "train": {"epochs": 300, "seed": 17},
    }

    override.write_text("data:\n  root: shared_data\ntrain:\n  epochs: 250\n", encoding="utf-8")
    child.write_text(
        "_base_:\n  - base.yaml\n  - override.yaml\ntrain:\n  seed: 29\n",
        encoding="utf-8",
    )
    assert load_config(child) == {
        "data": {"name": "cifar10", "root": "shared_data"},
        "train": {"epochs": 250, "seed": 29},
    }

    base.write_text("_base_: child.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Circular config inheritance"):
        load_config(child)


@pytest.mark.parametrize("invalid_root", ["[]\n", "false\n", "0\n", "text\n"])
def test_config_rejects_non_mapping_roots(tmp_path: Path, invalid_root: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(invalid_root, encoding="utf-8")
    with pytest.raises(ValueError, match="Config root must be a mapping"):
        load_config(path)
