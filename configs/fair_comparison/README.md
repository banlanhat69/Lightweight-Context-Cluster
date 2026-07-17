# So sánh kiến trúc CIFAR với một protocol duy nhất

Tất cả config trong thư mục này kế thừa recipe
`configs/recipes/cifar_coc_paper_inspired.yaml`. Runner kiểm tra toàn bộ khối
`data`, `train` và metadata protocol trước khi train. Artifact hoàn tất cũng chỉ
được tái sử dụng khi ba khối này khớp chính xác, nên HBCC-Small/Medium và mọi
baseline luôn nhận đúng cùng dữ liệu, augmentation, optimizer và seed.

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

Mặc định runner huấn luyện sáu mô hình đúng với bảng report cũ:

- ResNet-18: baseline CNN phổ biến;
- MobileNetV2: baseline lightweight CNN;
- ShuffleNetV2 x1.0: baseline nhẹ gần ngân sách tham số của HBCC-Small;
- CoC baseline: kiến trúc gốc cùng họ Context Cluster;
- HBCC-Small: cấu hình HBCC cũ hướng accuracy;
- HBCC-Medium: cấu hình HBCC cũ có capacity lớn hơn.

HBCC-Small+ và P-HBCC-2M vẫn được giữ để tái lập thử nghiệm cũ nhưng không thuộc
bảng chính. Với một shared seed `17`, ma trận chính có 6 runs × 200 epochs =
1.200 epoch-runs, giảm 90% so với thiết kế 8 × 5 × 300 = 12.000 epoch-runs.

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_fair_comparison.py `
  --dataset cifar10 `
  --benchmark
```

Đổi thành `--dataset cifar100` cho CIFAR-100. Có thể thêm mô hình mở rộng bằng
`--models`, nhưng phải dùng cùng seed 17 nếu đưa vào bảng chính.

## Quy tắc báo cáo

- Giữ `data.split_seed=42`; dùng `train.seed=17` và `data.loader_seed=17` cho mọi mô hình.
- Chỉ dùng run có `protocol.name=cifar_coc_paper_inspired_200e_v1`,
  `canonical=true` và đủ 200 epochs.
- Nếu thiếu bất kỳ model/seed nào, không lập bảng xếp hạng chính.
- Báo cáo test accuracy của `best.pth` và chênh lệch trực tiếp tại seed 17.
- Một seed không cho phép ước lượng standard deviation, confidence interval hoặc ý nghĩa thống kê.
- Benchmark chỉ chạy checkpoint của seed đầu tiên một lần cho mỗi kiến trúc.
- Khi chạy lại sau gián đoạn, runner tự bỏ qua artifact hoàn tất và khớp metadata;
  thư mục dở dang hoặc sai protocol vẫn bị từ chối để tránh trộn kết quả.

Smoke run không canonical:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_fair_comparison.py `
  --dataset cifar10 `
  --models hbcc_small `
  --seeds 17 `
  --smoke
```
