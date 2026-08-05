# Lightweight HBCC experiments

## Food-101 224x224: canonical HBCC comparison

The Food-101 pipeline is source-based and uses
[`notebooks/food101_hbcc_vs_resnet18_kaggle.ipynb`](notebooks/food101_hbcc_vs_resnet18_kaggle.ipynb)
as its main entry point. It downloads `torchvision.datasets.Food101`, creates a
fixed per-class split of 675 train / 75 validation / 250 held-out test images,
and compares two randomly initialized models under the same CoC-style
ImageNet training recipe:

- `hbcc_food101_best`: the locked Food-101 HBCC architecture in
  `lightweight_hbcc/models/food101.py`;
- `resnet18_224`: standard torchvision ResNet-18 with `weights=None`.

The train transform ports the original CoC ImageNet-1K defaults to Food-101:
`RandomResizedCrop(224, scale=0.08..1.0)`, horizontal flip, RandAugment,
ImageNet normalization, Random Erasing 0.25, Mixup 0.8, CutMix 1.0, and label
smoothing 0.1. Validation/test use resize-to-249 then center-crop-to-224 and no
random operation. Both models receive the identical recipe; ResNet-18 explicitly
uses `weights=None`.

`hbcc_food101_best` has 2,566,391 total parameters (2,562,935 trainable). It
keeps hard assignment and the four-stage hybrid design, while using uniform
7x7 cluster regions, GroupNorm, positive similarity scales, stochastic depth,
and a small training-only differentiable center-balance regularizer. The test
split is evaluated only after selecting the best validation checkpoint.

Command-line equivalent:

```powershell
python tools/run_food101_experiments.py `
  --models hbcc resnet18 `
  --data-root data/food101 `
  --output runs/food101 `
  --epochs 60
```

Các thí nghiệm CIFAR-10/CIFAR-100 cũ vẫn giữ giao thức không augmentation;
recipe CoC ở trên chỉ áp dụng cho pipeline Food-101 mới.

## Kiến trúc HBCC-Wide Stage 4 v1 đã khôi phục

Hai model HBCC dùng đúng kiến trúc CE tốt nhất đã đạt 84,20% trên CIFAR-10, seed 42 và 300 epoch:

- độ phân giải bốn stage: `32x32 -> 16x16 -> 8x8 -> 4x4` (`stem_stride=1`);
- fold: `4x4, 2x2, 1x1, 1x1`;
- proposal: `2x2, 2x2, 2x2, 1x1`;
- độ sâu của cả Small và Medium: `[1, 1, 2, 1]`;
- embed dim Small: `[48, 80, 160, 256]`, Medium: `[64, 96, 192, 288]`;
- assignment: hard ở cả bốn stage, temperature `1.0`;
- Stage 4 là Context Cluster thuần, không có nhánh DWConv và không channel shuffle;
- Drop Path Small/Medium: `0.05/0.08`.

PointReducer giữ convolution `3x3`, stride 2. Kiến trúc được gắn định danh `hbcc_wide_stage4_v1`; checkpoint CE 84,20% của chính kiến trúc này tương thích. Toàn bộ DKD và Attention Transfer mới vẫn được giữ nguyên để thử nghiệm distillation trên baseline tốt nhất.

## Ablation chỉ thay Stage 4

Catalog có thêm `hbcc_medium_stage4_ablation` để so sánh trực tiếp với `hbcc_medium`. Variant giữ nguyên resolution, PointReducer, embed dim, classification head và toàn bộ Stage 1–3. Chỉ Stage 4 thay đổi:

- depth `1 -> 2`;
- proposal `1x1 -> 2x2`;
- assignment `hard -> hard_st`, temperature `1.0 -> 0.7`;
- Context Cluster thuần thành hybrid với 25% DWConv và channel shuffle.

`assignment_modes` đúng là `[hard, hard, hard, hard_st]`; Stage 1–3 không dùng `hard_st`. Variant còn khai báo lịch DropPath theo stage để việc thêm block Stage 4 không âm thầm thay đổi DropPath của Stage 1–3. Runner kiểm tra toàn bộ các invariant này trước khi train.

## So sánh kiến trúc bằng CE

Mở [notebooks/run_ce_experiments.ipynb](notebooks/run_ce_experiments.ipynb) để huấn luyện CE:

- ResNet-18;
- MobileNetV2;
- ShuffleNetV2;
- CoC baseline;
- HBCC-Small-Wide;
- HBCC-Medium-Wide;
- HBCC-Medium-Stage4-Ablation.

Chỉnh `CE_EPOCHS` và các switch `TRAIN_*` trong ô cấu hình. Mặc định notebook chỉ bật HBCC-Medium baseline và Stage-4 ablation để tạo phép so sánh cô lập; cột `delta_vs_hbcc_medium` báo chênh lệch accuracy theo cùng dataset/seed. `SMOKE=True` dùng FakeData, một epoch và một batch để kiểm tra nhanh. Runner này không chứa teacher hoặc loss KD.

## Standard KD, DKD và DKD + Attention trên Kaggle

Mở [notebooks/run_hbcc_kd_dkd_kaggle.ipynb](notebooks/run_hbcc_kd_dkd_kaggle.ipynb). Notebook hỗ trợ ba phương pháp:

- Standard KD (`KL(teacher || student)`);
- DKD với hệ số tổng `DKD_SCALE`;
- `dkd_at`: DKD kết hợp Attention Transfer trên feature Stage 2–4.

Có thể bật/tắt từng student và từng phương pháp bằng các switch `TRAIN_HBCC_*`, `RUN_STANDARD_KD`, `RUN_DKD` và `RUN_DKD_AT`. Mặc định notebook chỉ chạy HBCC-Medium với `dkd_at`, 300 epoch, label smoothing 0, `DKD_SCALE=0.5` và Attention Transfer Stage 2–4. Mỗi run được gắn định danh `hbcc_wide_stage4_v1`; preflight xác nhận toàn bộ catalog và feature resolution `32/16/8/4` của teacher/student.

Nếu `REFERENCE_RESULTS_ROOTS` chứa run CE của ResNet-18, bảng cuối tự tính `gap_to_teacher` và `within_target_gap` với ngưỡng chỉnh được `TARGET_GAP_TO_TEACHER=0.5`.

Các đường dẫn Kaggle quan trọng được gom trong một ô cấu hình:

- `REPO_ROOT`;
- `DATA_ROOTS`;
- `OUTPUT_ROOT`;
- `TEACHER_CHECKPOINTS`;
- `TEACHER_CONFIGS`;
- `REFERENCE_RESULTS_ROOTS`.

Teacher phải là `best.pth` của ResNet-18 baseline no-augmentation, đúng dataset và seed. Checkpoint do pipeline này tạo đã chứa toàn bộ config; vì vậy có thể đặt `TEACHER_CONFIGS[(dataset, seed)] = None`. Runner sẽ trích config từ checkpoint và kiểm tra state dict, kiến trúc, dataset, seed, epoch và `augmentation: none` trước khi huấn luyện.

Checkpoint HBCC-CE chỉ được đọc để tổng hợp kết quả tham chiếu; student KD luôn khởi tạo lại từ cùng seed để phép so sánh không bị ảnh hưởng bởi warm-start.

## Command line cho DKD + Attention

```powershell
python tools/run_kd_comparison.py `
  --dataset cifar10 `
  --seed 42 `
  --teacher-checkpoint path/to/resnet18/best.pth `
  --expected-teacher-epochs 300 `
  --data-root data `
  --output runs_kd_comparison `
  --epochs 300 `
  --methods dkd_at `
  --students hbcc_medium `
  --label-smoothing 0 `
  --dkd-scale 0.5 `
  --feature-kd-weight 0.25 `
  --feature-kd-stages 2 3 4 `
  --download-data
```

## Runtime chính

- `configs/cifar_fair/`
- `lightweight_hbcc/`
- `tools/train.py`
- `tools/run_ce_experiments.py`
- `tools/run_kd_comparison.py`
- `notebooks/run_ce_experiments.ipynb`
- `notebooks/run_hbcc_kd_dkd_kaggle.ipynb`
