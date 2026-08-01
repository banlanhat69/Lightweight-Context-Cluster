# Lightweight HBCC - two no-augmentation sessions

Repository được thu gọn chỉ còn pipeline cần thiết để chạy hai phiên trên CIFAR-10/CIFAR-100:

1. Baseline CE: ResNet-18, MobileNetV2, ShuffleNetV2, CoC baseline, HBCC-Small và HBCC-Medium.
2. HBCC KD: HBCC-Small và HBCC-Medium dùng ResNet-18 cùng dataset/seed làm teacher.

Không có augmentation: train/validation/test chỉ dùng ToTensor và Normalize. KD dùng alpha=0.5, T=4.0 và KL(student || teacher) theo công thức trong báo cáo HBCC.

## Chạy bằng notebook

Mở [notebooks/run_two_sessions.ipynb](notebooks/run_two_sessions.ipynb) và chạy từ trên xuống.

Notebook mặc định:

- chạy cả CIFAR-10 và CIFAR-100;
- dùng seed 42;
- chạy 6 model CE trước rồi 2 HBCC KD cho mỗi dataset;
- tự bỏ qua run đã hoàn tất và đúng metadata;
- tổng hợp kết quả vào runs_two_sessions/two_sessions_summary.csv.

Chỉnh `BASELINE_EPOCHS` và `KD_EPOCHS` trong ô cấu hình notebook để đặt riêng số epoch cho phiên baseline và phiên knowledge distillation. Đặt `SMOKE=True` để kiểm tra nhanh toàn bộ luồng bằng FakeData; chế độ smoke luôn dùng 1 epoch.

## Chạy bằng command line

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_two_sessions.py `
  --dataset cifar10 `
  --output runs_two_sessions
```

Đổi dataset thành cifar100 để chạy tập còn lại.

## Các file runtime chính

- configs/cifar_fair/model_catalog.yaml
- configs/cifar_fair/cifar10_no_augmentation.yaml
- configs/cifar_fair/cifar100_no_augmentation.yaml
- lightweight_hbcc/
- tools/train.py
- tools/run_two_sessions.py
- notebooks/run_two_sessions.ipynb
