# Lightweight HBCC for Context Cluster

This workspace implements the research plan in `docs/lightweight_hbcc_research_plan.md` as a reproducible CIFAR pipeline.

## Environment

The target conda environment is `CoC`. On this machine it has been initialized with Python 3.11 and CUDA PyTorch.

```powershell
conda activate CoC
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the env has to be rebuilt:

```powershell
conda env update -n CoC -f environment.yml
```

When `conda activate` is unreliable, call the interpreter directly:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\shape_trace.py --config configs\smoke.yaml
```

## What Is Implemented

- CoC-style context clustering with cosine similarity, hard assignment, sigmoid-scaled aggregation and dispatch.
- Optional RGB+XY coordinate augmentation.
- Early reducer schedule for CIFAR: `32 -> 16 -> 8 -> 4 -> 2`.
- HBCC-Latency Tiny/Small configs.
- HBCC-Small/Medium accuracy-oriented configs used by the primary report.
- Archived P-HBCC-2M experimental architecture; it is no longer in the primary comparison.
- Current-reference HBCC config that intentionally keeps the first cluster stage at `32x32`.
- Local branch ablations: identity, DWConv, fixed LBPConv.
- Similarity ablations: cosine and simulated Hamming with STE.
- Structured channel mask training and materialized pruned-config export.
- CE-only and KD training.
- Benchmark protocol for batch `1, 16, 64, 128`, strict latency, streaming throughput, peak memory, FLOPs best effort and torch operator profile.
- Pareto report generation from JSON benchmark records.
- Architecture-controlled CIFAR configs with one inherited augmentation/training recipe and a shared-seed runner.

## Quick Checks

```powershell
& D:\Anaconda\envs\CoC\python.exe -m pytest
& D:\Anaconda\envs\CoC\python.exe tools\shape_trace.py --config configs\hbcc_latency_tiny.yaml
& D:\Anaconda\envs\CoC\python.exe tools\shape_trace.py --config configs\hbcc_accuracy_small.yaml
& D:\Anaconda\envs\CoC\python.exe tools\shape_trace.py --config configs\hbcc_accuracy_medium.yaml
& D:\Anaconda\envs\CoC\python.exe tools\run_fair_comparison.py --dataset cifar10 --validate-only
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\smoke.yaml --batch-sizes 1 4 --warmup 2 --runs 3
```

## Training

For CIFAR-10, CIFAR-100, and STL-10, the training pipeline uses the original
training split for model selection and keeps the official test split for final
reporting:

- CIFAR-10/CIFAR-100 `train`: 45k images from the original training split
- CIFAR-10/CIFAR-100 `val`: 5k images from the original training split, controlled by `data.split_seed`
- CIFAR-10/CIFAR-100 `test`: official 10k test split, evaluated once on `best.pth`
- STL-10 `train`: 4.5k images from official STL-10 train
- STL-10 `val`: 500 images from official STL-10 train, controlled by `data.split_seed`
- STL-10 `test`: official 8k test split, evaluated once on `best.pth`

Validation metrics are written into `metrics.jsonl` every epoch. Final test
metrics are written to both `metrics.jsonl` and `test_metrics.json`.

CIFAR-100 follows the same method as the CIFAR-10 pipeline: reuse the CIFAR-10
configs and add overrides for `data.name=cifar100`, `model.num_classes=100`,
and a CIFAR-100-specific `experiment.name`. See
`notebooks/cifar100_training_pipeline.ipynb` for the phase-by-phase workflow.

Student CE-only:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\hbcc_latency_tiny.yaml --output runs
```

Short proxy training for search:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\hbcc_latency_tiny.yaml --output runs_proxy --override train.epochs=30
```

Knowledge distillation:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\hbcc_latency_tiny.yaml --output runs_kd --teacher-config configs\baselines\resnet18_cifar.yaml --teacher-checkpoint runs\resnet18_cifar\best.pth --override train.kd_alpha=0.5 --override train.kd_temperature=4.0
```

Full STL-10 experiment matrix:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_stl10_experiments.py
```

Full CIFAR-100 pipeline using the CIFAR-10 method:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_cifar100_experiments.py
```

Quick STL-10 smoke check:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_stl10_experiments.py --smoke --only resnet18_reference_stl10 hbcc_small_no_mix_stl10 --skip-kd
```

Quick CIFAR-100 smoke check:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_cifar100_experiments.py --smoke --only resnet18_cifar100 hbcc_latency_tiny_cifar100 --skip-kd
```

## Fair Architecture Comparison

The primary architecture table uses one shared paper-inspired recipe under
`configs/fair_comparison`. It compares the original HBCC-Small/Medium approach
against the four baselines already used in the report (ResNet-18, MobileNetV2,
ShuffleNetV2 and CoC) using one shared seed (`17`): 6 training runs of 200 epochs.
P-HBCC-2M is retained only as an optional experimental artifact and is not part
of the default matrix. The
recipe follows the augmentation list in Context Cluster section 4.1 with a
CIFAR RandomCrop adaptation; RandAugment is disabled. The runner validates that
the effective `data`, `train` and protocol blocks are identical before launch.

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_fair_comparison.py `
  --dataset cifar10 `
  --benchmark
```

Use `--dataset cifar100` for the corresponding CIFAR-100 matrix. The training
seed controls model initialization, DataLoader workers/order and independent
MixUp/CutMix RNG streams; `data.split_seed` remains fixed at 42. The recipe is
paper-inspired rather than an exact ImageNet reproduction: it uses CIFAR,
200 epochs, batch size 128 and no EMA. With one seed, results are descriptive
and do not support standard-deviation or confidence-interval claims. See
`configs/fair_comparison/README.md` for the fairness contract.

An executable notebook with safe smoke defaults, shared-seed controls and
result aggregation is available at `notebooks/hbcc_fair_training.ipynb`.

### CIFAR 4x4 spatial ablations

The current Kaggle workflow in `notebooks/cifar_fair_training.ipynb` exposes
three opt-in HBCC candidates from `configs/cifar_fair/model_catalog.yaml`:

- `hbcc_small_keep4`: changes only the reducer strides from `[2, 2, 2]` to
  `[2, 2, 1]`, producing `32 -> 16 -> 8 -> 4 -> 4`.
- `hbcc_medium_keep4`: applies the same single-variable spatial ablation to
  HBCC-Medium.
- `hbcc_medium_keep4_late_hybrid`: builds on `hbcc_medium_keep4` and changes
  only the last stage to a 50/50 cluster + depthwise-convolution hybrid.

The original `hbcc_small` and `hbcc_medium` entries remain unchanged. Enable
one candidate at a time in `MODEL_SWITCHES` and compare it with its original
counterpart under the same dataset, profile, seed, and epoch count.

### HBCC cluster-operator redesign

The Keep-4x4 result isolates spatial resolution, but it does not address three
operator-level limitations: gradients only reach the selected cluster,
unconstrained similarity scale can become negative and reverse nearest-center
assignment, and the late cluster projections compress 192/256 channels to 64.

Two additional opt-in candidates keep the efficient
`32 -> 16 -> 8 -> 4 -> 2` schedule:

- `hbcc_medium_stable` keeps the same 2.864M parameters as HBCC-Medium. It uses
  hard assignments with soft straight-through gradients, a strictly positive
  similarity scale, `1e-3` LayerScale initialization, and one global proposal
  in the final 2x2 stage.
- `hbcc_medium_v2` adds only wider late cluster embeddings
  (`head_dim=[16, 16, 20, 24]`) and remains below 3M parameters.

Run `hbcc_medium_stable` first against `hbcc_medium`. Run `hbcc_medium_v2`
only after that comparison so the value of the extra late-stage capacity stays
measurable.

## Benchmark Matrix

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\baselines\resnet18_cifar.yaml --output results\benchmark --profile
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\baselines\mobilenet_v2_cifar.yaml --output results\benchmark --profile
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\coc_cifar_baseline.yaml --output results\benchmark --profile
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\hbcc_current_reference.yaml --output results\benchmark --profile
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\hbcc_latency_tiny.yaml --output results\benchmark --profile
```

Trained accuracy can be added to benchmark JSON records from `runs/*/metrics.jsonl` before building final Pareto tables.
For CIFAR/STL-10 runs, use `test_acc1` for held-out reporting.

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\pareto_report.py results\benchmark --output results\pareto.md
```

## Ablations

Generate one config per ablation from the Tiny base:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\ablation_matrix.py --base configs\hbcc_latency_tiny.yaml --output-dir configs\generated_ablations
```

Main manual ablation configs are also available in `configs/ablations`.

## Pruning Export

Train with masks:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\ablations\hbcc_tiny_pruning_mask.yaml --output runs_pruning
```

Export a smaller materialized config:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\export_pruned.py --config configs\ablations\hbcc_tiny_pruning_mask.yaml --checkpoint runs_pruning\hbcc_tiny_pruning_mask\best.pth --output configs\generated_ablations\hbcc_tiny_pruned_export.yaml
```

Then fine-tune and benchmark the exported config.
