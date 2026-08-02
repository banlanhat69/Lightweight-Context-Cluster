# Lightweight HBCC - no-augmentation experiments

Repository chỉ giữ pipeline cần thiết để chạy các thí nghiệm CIFAR-10/CIFAR-100 không augmentation. Train, validation và test chỉ dùng `ToTensor` và `Normalize`.

## Kiến trúc HBCC theo PDF

Hai model HBCC trong `configs/cifar_fair/model_catalog.yaml` dùng đúng pipeline CIFAR đã sửa suy biến token trong báo cáo:

- độ phân giải bốn stage: `32x32 -> 16x16 -> 8x8 -> 4x4` (`stem_stride=1`);
- fold: `4x4, 2x2, 1x1, 1x1`;
- proposal: `2x2, 2x2, 2x2, 1x1`, nên mỗi cluster có `r=16` điểm;
- độ sâu của cả Small và Medium: `[1, 1, 2, 1]`;
- embed dim Small: `[48, 80, 160, 224]`, Medium: `[64, 96, 192, 256]`;
- Drop Path Small/Medium: `0.05/0.08`.

PointReducer giữ convolution `3x3`, stride 2. Cấu hình này khớp số tham số được báo cáo trong PDF: khoảng 1.27M/1.76M trên CIFAR-10 và 1.30M/1.78M trên CIFAR-100. Checkpoint HBCC tạo từ kiến trúc cũ không tương thích về giao thức thực nghiệm và không được dùng để tiếp tục so sánh; ResNet-18 teacher no-augmentation vẫn dùng lại được.

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
