# Lightweight HBCC - no-augmentation experiments

Repository chỉ giữ pipeline cần thiết để chạy các thí nghiệm CIFAR-10/CIFAR-100 không augmentation. Train, validation và test chỉ dùng `ToTensor` và `Normalize`.

## So sánh kiến trúc và KD theo báo cáo HBCC

Mở [notebooks/run_two_sessions.ipynb](notebooks/run_two_sessions.ipynb) để chạy:

1. Baseline CE: ResNet-18, MobileNetV2, ShuffleNetV2, CoC baseline, HBCC-Small và HBCC-Medium.
2. HBCC KD theo phương trình trong báo cáo: HBCC-Small và HBCC-Medium dùng ResNet-18 cùng dataset/seed làm teacher.

Chỉnh `BASELINE_EPOCHS` và `KD_EPOCHS` trong ô cấu hình. `SMOKE=True` dùng FakeData, một epoch và một batch để kiểm tra nhanh.

## So sánh Standard KD và DKD trên Kaggle

Mở [notebooks/run_hbcc_kd_dkd_kaggle.ipynb](notebooks/run_hbcc_kd_dkd_kaggle.ipynb). Notebook chạy bốn thí nghiệm cho mỗi dataset/seed:

- HBCC-Small với Standard KD (`KL(teacher || student)`);
- HBCC-Medium với Standard KD;
- HBCC-Small với DKD;
- HBCC-Medium với DKD.

Các đường dẫn Kaggle quan trọng được gom trong một ô cấu hình:

- `REPO_ROOT`;
- `DATA_ROOTS`;
- `OUTPUT_ROOT`;
- `TEACHER_CHECKPOINTS`;
- `TEACHER_CONFIGS`;
- `REFERENCE_RESULTS_ROOTS`.

Teacher phải là `best.pth` của ResNet-18 baseline no-augmentation, đúng dataset và seed. Checkpoint do pipeline này tạo đã chứa toàn bộ config; vì vậy có thể đặt `TEACHER_CONFIGS[(dataset, seed)] = None`. Runner sẽ trích config từ checkpoint và kiểm tra state dict, kiến trúc, dataset, seed, epoch và `augmentation: none` trước khi huấn luyện.

Checkpoint HBCC-CE chỉ được đọc để tổng hợp kết quả tham chiếu; student KD luôn khởi tạo lại từ cùng seed để phép so sánh không bị ảnh hưởng bởi warm-start.

## Command line cho Standard KD và DKD

```powershell
python tools/run_kd_comparison.py `
  --dataset cifar10 `
  --seed 42 `
  --teacher-checkpoint path/to/resnet18/best.pth `
  --expected-teacher-epochs 200 `
  --data-root data `
  --output runs_kd_comparison `
  --epochs 200 `
  --methods standard dkd `
  --students hbcc_small hbcc_medium `
  --download-data
```

## Runtime chính

- `configs/cifar_fair/`
- `lightweight_hbcc/`
- `tools/train.py`
- `tools/run_two_sessions.py`
- `tools/run_kd_comparison.py`
- `notebooks/run_two_sessions.ipynb`
- `notebooks/run_hbcc_kd_dkd_kaggle.ipynb`
