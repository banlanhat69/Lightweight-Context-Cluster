# Lightweight HBCC - no-augmentation experiments

Repository chỉ giữ pipeline cần thiết để chạy các thí nghiệm CIFAR-10/CIFAR-100 không augmentation. Train, validation và test chỉ dùng `ToTensor` và `Normalize`.

## Kiến trúc HBCC mở rộng Stage 4

Hai model HBCC giữ pipeline CIFAR đã sửa suy biến token trong báo cáo và mở rộng riêng số channel Stage 4:

- độ phân giải bốn stage: `32x32 -> 16x16 -> 8x8 -> 4x4` (`stem_stride=1`);
- fold: `4x4, 2x2, 1x1, 1x1`;
- proposal: `2x2, 2x2, 2x2, 1x1`, nên mỗi cluster có `r=16` điểm;
- độ sâu của cả Small và Medium: `[1, 1, 2, 1]`;
- embed dim Small: `[48, 80, 160, 256]`, Medium: `[64, 96, 192, 288]`;
- Drop Path Small/Medium: `0.05/0.08`.

PointReducer giữ convolution `3x3`, stride 2. Sau khi mở rộng, HBCC-Small có khoảng 1.42M tham số và HBCC-Medium khoảng 1.92M tham số trên CIFAR-10. Checkpoint HBCC tạo trước thay đổi Stage 4 không tương thích; ResNet-18 teacher no-augmentation vẫn dùng lại được trong pipeline KD riêng.

## So sánh kiến trúc bằng CE

Mở [notebooks/run_ce_experiments.ipynb](notebooks/run_ce_experiments.ipynb) để huấn luyện CE:

- ResNet-18;
- MobileNetV2;
- ShuffleNetV2;
- CoC baseline;
- HBCC-Small-Wide;
- HBCC-Medium-Wide.

Chỉnh `CE_EPOCHS` và các switch `TRAIN_*` trong ô cấu hình. `SMOKE=True` dùng FakeData, một epoch và một batch để kiểm tra nhanh. Runner này không chứa teacher hoặc loss KD.

## So sánh Standard KD và DKD trên Kaggle

Mở [notebooks/run_hbcc_kd_dkd_kaggle.ipynb](notebooks/run_hbcc_kd_dkd_kaggle.ipynb). Notebook chạy bốn thí nghiệm cho mỗi dataset/seed:

- HBCC-Small với Standard KD (`KL(teacher || student)`);
- HBCC-Medium với Standard KD;
- HBCC-Small với DKD;
- HBCC-Medium với DKD.

Có thể bật/tắt từng student và từng phương pháp bằng các switch `TRAIN_HBCC_*` và `RUN_*_KD`, đồng thời chỉnh độc lập `KD_EPOCHS`. Mỗi run được gắn định danh kiến trúc `hbcc_wide_stage4_v1` và kiểm tra chính xác `embed_dims` trước khi train, nên kết quả HBCC cũ không bị nhận nhầm là kết quả của model mới.

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
  --expected-teacher-epochs 300 `
  --data-root data `
  --output runs_kd_comparison `
  --epochs 300 `
  --methods standard dkd `
  --students hbcc_small hbcc_medium `
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
