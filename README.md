# Lightweight HBCC - no-augmentation experiments

Repository chỉ giữ pipeline cần thiết để chạy các thí nghiệm CIFAR-10/CIFAR-100 không augmentation. Train, validation và test chỉ dùng `ToTensor` và `Normalize`.

## Kiến trúc HBCC Accuracy Stage 4

Hai model HBCC giữ pipeline CIFAR không augmentation nhưng thay Stage 4 một-center bằng biến thể ưu tiên accuracy:

- độ phân giải bốn stage: `32x32 -> 16x16 -> 8x8 -> 4x4` (`stem_stride=1`);
- fold: `4x4, 2x2, 1x1, 1x1`;
- proposal: `2x2, 2x2, 2x2, 2x2`;
- độ sâu của cả Small và Medium: `[1, 1, 2, 2]`;
- embed dim Small: `[48, 80, 160, 256]`, Medium: `[64, 96, 192, 288]`;
- assignment: hard straight-through; temperature Stage 4 là `0.7`;
- Stage 4 hybrid với 25% DWConv, channel shuffle và 75% Context Cluster;
- Drop Path Small/Medium: `0.05/0.08`.

PointReducer giữ convolution `3x3`, stride 2. Runner in số tham số chính xác khi dựng model. Checkpoint HBCC tạo trước kiến trúc `hbcc_accuracy_stage4_v2` không tương thích; ResNet-18 teacher no-augmentation vẫn dùng lại được trong pipeline KD riêng.

## So sánh kiến trúc bằng CE

Mở [notebooks/run_ce_experiments.ipynb](notebooks/run_ce_experiments.ipynb) để huấn luyện CE:

- ResNet-18;
- MobileNetV2;
- ShuffleNetV2;
- CoC baseline;
- HBCC-Small-Accuracy;
- HBCC-Medium-Accuracy.

Chỉnh `CE_EPOCHS` và các switch `TRAIN_*` trong ô cấu hình. `SMOKE=True` dùng FakeData, một epoch và một batch để kiểm tra nhanh. Runner này không chứa teacher hoặc loss KD.

## Standard KD, DKD và DKD + Attention trên Kaggle

Mở [notebooks/run_hbcc_kd_dkd_kaggle.ipynb](notebooks/run_hbcc_kd_dkd_kaggle.ipynb). Notebook hỗ trợ ba phương pháp:

- Standard KD (`KL(teacher || student)`);
- DKD với hệ số tổng `DKD_SCALE`;
- `dkd_at`: DKD kết hợp Attention Transfer trên feature Stage 2–4.

Có thể bật/tắt từng student và từng phương pháp bằng các switch `TRAIN_HBCC_*`, `RUN_STANDARD_KD`, `RUN_DKD` và `RUN_DKD_AT`. Mặc định notebook chỉ chạy HBCC-Medium với `dkd_at`, 300 epoch, label smoothing 0, `DKD_SCALE=0.5` và Attention Transfer Stage 2–4. Mỗi run được gắn định danh `hbcc_accuracy_stage4_v2` và preflight xác nhận feature resolution `32/16/8/4` của cả teacher/student.

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
