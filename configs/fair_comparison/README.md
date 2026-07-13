# So sánh kiến trúc CIFAR với một protocol duy nhất

Tất cả config trong thư mục này kế thừa recipe
`configs/recipes/cifar_coc_paper_inspired.yaml`. Runner kiểm tra toàn bộ khối
`data`, `train` và metadata protocol trước khi train, nên P-HBCC-2M và mọi
baseline luôn nhận đúng cùng dữ liệu, augmentation và optimizer.

## Recipe paper-inspired

Context Cluster (ICLR 2023), Mục 4.1, nêu random horizontal flip, random pixel
erasing, MixUp, CutMix và label smoothing. Phiên bản CIFAR dùng:

- RandomCrop 32x32 với padding 4 (thích nghi chuẩn cho CIFAR);
- horizontal flip;
- Random Erasing, `p=0.25`, diện tích `0.02-1/3`, giá trị pixel ngẫu nhiên;
- MixUp, `alpha=0.8`;
- CutMix, `alpha=1.0`, xác suất chuyển từ MixUp sang CutMix là `0.5`;
- label smoothing `0.1`.

RandAugment được tắt vì không được liệt kê trong phần mô tả augmentation của
bài báo và không cần thêm một protocol ablation. Đây là recipe **paper-inspired**,
không phải tái lập ImageNet nguyên bản: pipeline dùng CIFAR, batch size 128,
200 epochs và không dùng EMA. Các thích nghi này được áp dụng giống hệt cho mọi
kiến trúc.

## Ma trận tiết kiệm tài nguyên

Mặc định runner chỉ huấn luyện năm mô hình cốt lõi:

- ResNet-18: baseline CNN phổ biến;
- MobileNetV2: baseline lightweight CNN;
- CoC baseline: kiến trúc gốc cùng họ Context Cluster;
- HBCC-Small: kiến trúc HBCC hiện tại;
- P-HBCC-2M: kiến trúc đề xuất.

Ba config mở rộng (ShuffleNetV2, HBCC-Small+ và HBCC-Medium) vẫn được giữ nhưng
không chạy mặc định. Với ba paired seeds `17, 29, 43`, ma trận chính có 15 runs
x 200 epochs = 3.000 epoch-runs, thay vì thiết kế mở rộng 40 x 300 = 12.000;
tổng ngân sách epoch giảm 75%.

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_fair_comparison.py `
  --dataset cifar10 `
  --benchmark
```

Đổi thành `--dataset cifar100` cho CIFAR-100. Có thể thêm mô hình mở rộng bằng
`--models`, nhưng phải chạy đủ cùng ba seed nếu đưa vào bảng chính.

## Quy tắc báo cáo

- Giữ `data.split_seed=42`; chỉ thay `train.seed` theo paired seeds.
- Chỉ dùng run có `protocol.name=cifar_coc_paper_inspired_200e_v1`,
  `canonical=true` và đủ 200 epochs.
- Nếu thiếu bất kỳ model/seed nào, không lập bảng xếp hạng chính.
- Báo cáo test accuracy của `best.pth` dưới dạng mean +/- SD và paired delta/CI.
- Ba seed là mức tối thiểu; không dùng CI rộng này để khẳng định non-inferiority mạnh.
- Benchmark chỉ chạy checkpoint của seed đầu tiên một lần cho mỗi kiến trúc.
- Khi chạy lại sau gián đoạn, runner tự bỏ qua artifact hoàn tất và khớp metadata;
  thư mục dở dang hoặc sai protocol vẫn bị từ chối để tránh trộn kết quả.

Smoke run không canonical:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_fair_comparison.py `
  --dataset cifar10 `
  --models phbcc_2m `
  --seeds 17 `
  --smoke
```
