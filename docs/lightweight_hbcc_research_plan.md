# Research Plan: Lightweight HBCC cho Context Cluster

## 1. Mục tiêu 
Mục tiêu không chỉ là giảm số lượng tham số, mà là tìm được một hoặc nhiều cấu hình nằm trên Pareto frontier giữa:

- `Accuracy`: độ chính xác phân loại, ưu tiên `Top-1`.
- `Params`: số lượng tham số huấn luyện được và tổng tham số.
- `FLOPs/MACs`: chi phí số học thực trong các toán tử float/int.
- `BOPs`: chi phí lý thuyết của toán tử nhị phân, chỉ có ý nghĩa inference nếu có bitwise kernel thật.
- `Latency`: thời gian suy luận một batch, đặc biệt quan trọng ở batch 1.
- `Throughput`: ảnh/giây ở batch lớn hơn, ví dụ batch 16, 64, 128.
- `Memory`: peak GPU memory, activation memory, và footprint của checkpoint/model export.

Vấn đề hiện tại là mô hình HBCC đã giảm được số tham số so với ResNet-18, nhưng throughput lại giảm mạnh. Điều này không bất thường nếu nhìn vào execution path: CoC/HBCC có nhiều tensor reshape, permute, scatter, reduction, sigmoid, matmul nhỏ, branch nhỏ và kernel launch nhỏ. Các thao tác này có ít tham số nhưng có thể chậm hơn các convolution lớn được cuDNN tối ưu tốt.

Vì vậy hướng nghiên cứu cần chuyển từ:

```text
Giảm params/FLOPs trên giấy
```

sang:

```text
Tối ưu Pareto: accuracy / params / latency / memory trên phần cứng thật
```

## 2. Nguyên tắc nghiên cứu

### 2.1. Không đồng nhất params với tốc độ

Một model ít tham số có thể chậm hơn model nhiều tham số nếu:

- Dùng nhiều kernel nhỏ.
- Có nhiều branch nhỏ, concat, split, shuffle.
- Có nhiều `permute`, `reshape`, `contiguous`, `scatter_`, `topk`, `argmax`.
- Tạo nhiều activation trung gian.
- Dùng toán tử không được backend tối ưu.

Trong HBCC, các phần dễ gây latency gồm:

- Multi-head reshape trong cluster.
- Similarity matrix giữa centers và points.
- Hard assignment bằng `max` + `scatter_`.
- Aggregate/dispatch bằng broadcast + `sum`.
- Channel split/fusion và channel shuffle.
- PointShrink/LBPConv nếu tạo nhiều map trung gian.

### 2.2. BOPs chỉ là lợi thế nếu inference thật sự bitwise

Hamming similarity có thể thay cosine trong không gian nhị phân:

```math
x_b, y_b \in \{-1, +1\}^D
```

```math
x_b^\top y_b = D - 2 \cdot HD(x_b, y_b)
```

Và nếu dùng biểu diễn bit:

```math
x_b^\top y_b \propto \text{popcount}(\text{XNOR}(x_b, y_b))
```

Tuy nhiên, nếu implementation vẫn dùng `torch.sign()` và `torch.matmul()` trên tensor float, lợi thế bitwise chưa tồn tại trong runtime. Khi đó BOPs chỉ là thước đo lý thuyết, không được dùng để kết luận model nhanh hơn.

### 2.3. Token count quan trọng hơn channel count ở CoC

Context Cluster có chi phí phụ thuộc mạnh vào số điểm `N`, số center `M`, và chiều đặc trưng `D`:

```math
O(N \cdot M \cdot D)
```

Do đó, với ảnh CIFAR-10 `32x32`, để lightweight thật sự:

- Không nên để stage đầu cluster global trên `32x32` nếu không có lý do rất mạnh.
- Cần giảm số điểm sớm: `32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2`.
- Global clustering nên xuất hiện ở stage muộn khi `N` đã nhỏ.
- Stage sớm nên dùng local lightweight operation hoặc region-local cluster.

### 2.4. Mọi kỹ thuật phải qua ablation độc lập

Các kỹ thuật sau không nên bật đồng thời rồi kết luận tổng quát:

- Hamming similarity.
- PointShrink hoặc point reducer.
- LBPConv branch.
- Linear bottleneck.
- Channel shuffle.
- Structured pruning.
- Distillation.

Mỗi kỹ thuật phải có ablation riêng, đo cùng một protocol, và được giữ lại chỉ khi:

- Cải thiện accuracy ở cùng cost.
- Giảm latency/memory ở accuracy tương đương.
- Hoặc tạo trade-off rõ ràng trên Pareto frontier.

## 3. Cơ sở từ paper Context Cluster gốc

Paper `Image as Set of Points` xem ảnh như một tập hợp điểm thay vì lưới convolution hoặc chuỗi patch attention. Các thành phần quan trọng:

### 3.1. Coordinate augmentation

Ảnh đầu vào:

```math
I \in \mathbb{R}^{3 \times H \times W}
```

được augment thêm tọa độ 2D cho mỗi pixel:

```math
\left[\frac{i}{H} - 0.5, \frac{j}{W} - 0.5\right]
```

Mỗi điểm ban đầu có 5 chiều:

```math
p_i = [R_i, G_i, B_i, x_i, y_i]
```

Điểm này quan trọng vì CoC muốn xử lý ảnh như set of points. Nếu bỏ tọa độ, mô hình mất tín hiệu vị trí rõ ràng; paper gốc cũng cho thấy positional information là thành phần quan trọng.

### 3.2. Points reducer

Paper giảm số điểm bằng cách đặt anchors đều trong không gian, gom `k` điểm gần nhất quanh mỗi anchor, concatenate đặc trưng rồi dùng projection tuyến tính để sinh điểm mới.

Trong ảnh có trật tự lưới, reducer này có thể implement bằng convolution/pooling tương đương. Điểm quan trọng không phải tên module, mà là:

- Số điểm phải giảm thật.
- Spatial size phải giảm theo stage.
- Reducer phải tạo representation mới, không chỉ preprocess local feature rồi giữ nguyên `H x W`.

Với CIFAR-10, một lịch giảm hợp lý là:

```text
Input:  32 x 32
S1:     16 x 16
S2:      8 x 8
S3:      4 x 4
S4:      2 x 2
```

### 3.3. Context cluster operation

Một block context cluster gồm:

1. Project điểm sang similarity space.
2. Propose centers đều trong không gian.
3. Tính cosine similarity giữa centers và points.
4. Hard-assign mỗi point vào center gần nhất.
5. Aggregate value features trong từng cluster.
6. Dispatch feature cluster ngược lại từng point.
7. Multi-head, concat, projection.
8. MLP/channel communication.

Công thức aggregate:

```math
g =
\frac{
  v_c + \sum_{i=1}^{m} \sigma(\alpha s_i + \beta) v_i
}{
  1 + \sum_{i=1}^{m} \sigma(\alpha s_i + \beta)
}
```

Trong đó:

- `v_c`: center trong value space, giúp cluster rỗng vẫn có tín hiệu.
- `s_i`: similarity giữa point và center.
- `alpha`, `beta`: tham số học được.
- `sigmoid`: chuẩn hóa score về `(0, 1)`, không dùng softmax.

### 3.4. Vì sao region partition tồn tại

Nếu cluster trên toàn ảnh ở stage sớm, chi phí và memory tăng nhanh theo `N`. Region partition chia ảnh thành vùng nhỏ, cluster độc lập trong mỗi vùng để giảm chi phí. Nó hy sinh một phần global receptive field nhưng giúp model train/infer khả thi.

Do đó, đề xuất bỏ toàn bộ region partition chỉ nên được xem là hợp lý khi:

- `N` đã nhỏ, ví dụ stage muộn.
- Hoặc inference path đã có bitwise kernel thật.
- Hoặc benchmark chứng minh memory/latency vẫn tốt.

## 4. Chẩn đoán implementation hiện tại

Các điểm lệch cần được ghi nhận trước khi tối ưu:

| Current issue | Vì sao là vấn đề | Proposed fix | Metric to verify |
|---|---|---|---|
| Input vẫn là RGB 3 channel | Lệch với paper CoC dùng RGB + XY | Thêm coordinate augmentation tùy chọn | Accuracy, convergence, ablation with/without XY |
| HBCC stage đầu chạy `32x32` | Token count quá lớn ở block cluster đầu | Stem/reducer giảm sớm xuống `16x16` | Latency batch 1/128, FLOPs/MACs, accuracy |
| `PointShrink` hiện stride 1 | Không giảm số điểm, chỉ thêm local preprocessing | Tách rõ `LocalPreprocess` và `PointReducer` | Shape trace qua stage, latency |
| Hamming dùng float matmul | Chưa có lợi thế XNOR/popcount thật | Giai đoạn đầu chỉ xem là ablation accuracy; BitEdge làm sau | Operator profile, latency |
| Profiler chỉ đếm Conv/Linear | Thiếu matmul/scatter/reduction/reshape | Thêm operator-aware profiler hoặc torch profiler summary | FLOPs/MACs report đầy đủ hơn |
| LBPConv/pruning chưa bật trong config chính | Proposal chưa được kiểm chứng trong pipeline chính | Bật qua ablation riêng, không mặc định | Accuracy/latency từng ablation |
| Pruning mask chưa export model nhỏ thật | Soft mask không làm inference nhanh | Thêm export/prune/fine-tune | Latency model đã export |
| Benchmark batch 1 đơn lẻ | Không phản ánh paper và không đủ cho deployment | Đo batch 1/16/64/128 | Latency + throughput matrix |

## 5. Đánh giá các đề xuất hiện tại

### 5.1. Binarized Global Clustering

#### Ý tưởng

Thay cosine similarity bằng Hamming similarity trong không gian nhị phân. Nếu có bit-packed representation, similarity có thể dùng XNOR + popcount.

#### Điểm đúng

- Quan hệ giữa cosine và Hamming trong không gian `{ -1, +1 }` là rõ ràng.
- Có thể giảm chi phí lý thuyết của similarity path.
- Có thể giảm memory của activation similarity nếu pack bit đúng cách.

#### Rủi ro

- Nếu dùng PyTorch float matmul, không nhanh hơn đáng kể.
- Binarization có thể làm mất thông tin fine-grained, đặc biệt ở stage sớm.
- Bỏ region partition ngay từ stage đầu có thể tăng memory/latency.
- Hamming chỉ thay similarity path; aggregate/dispatch vẫn còn broadcast, scatter, sum.

#### Điều chỉnh đề xuất

Không gọi đây là `Binarized Global Clustering` mặc định cho toàn bộ mạng. Nên tách thành:

- `Binarized Similarity`: thay cosine bằng Hamming trong similarity path.
- `Late Global Cluster`: chỉ global ở stage muộn khi `N` nhỏ.
- `BitEdge Inference`: bitwise path thật, phát triển sau khi model float/latency đã ổn.

#### Tiêu chí giữ lại

Giữ Hamming nếu đạt ít nhất một điều kiện:

- Accuracy giảm không đáng kể và latency không tệ hơn cosine.
- Accuracy tốt hơn cosine trong cùng architecture.
- Khi có bitwise kernel, latency/memory giảm rõ ràng.

### 5.2. Straight-Through Estimator

#### Ý tưởng

Vì `sign()` có gradient gần như bằng 0, dùng STE để forward binarized nhưng backward truyền gradient xấp xỉ.

#### Điểm đúng

- Cần thiết cho training BNN.
- Hard-tanh STE trong khoảng `[-1, 1]` là lựa chọn phổ biến.

#### Rủi ro

- STE không đảm bảo accuracy.
- Binarization sớm có thể làm optimization khó.
- Cần scaling factor để bù mất biên độ.

#### Điều chỉnh đề xuất

Thêm các cơ chế sau vào roadmap:

- Learnable scale theo head hoặc channel:

  ```math
  \hat{x} = \gamma \cdot \text{sign}(x)
  ```

- Gradual binarization: train warmup bằng cosine/full precision rồi bật Hamming.
- Distillation từ teacher full precision.
- Theo dõi activation distribution trước và sau sign.

#### Tiêu chí giữ lại

STE path được xem là ổn nếu:

- Training ổn định, không collapse.
- Accuracy không giảm quá ngưỡng so với cosine.
- Activation binary có entropy đủ cao, không toàn `+1` hoặc `-1`.

### 5.3. Hybrid LBPConv Branch

#### Ý tưởng

Một nửa kênh đi qua global/binarized cluster, nửa còn lại đi qua branch local dùng fixed LBP filters để giữ high-frequency details.

#### Điểm đúng

- Có lý do rõ: binarization và global cluster có thể mất edge/texture.
- LBP filters cố định giảm tham số.
- Nhánh local giúp thêm inductive bias cho CIFAR và ảnh nhỏ.

#### Rủi ro

- LBPConv có thể giảm params nhưng chưa chắc giảm latency.
- Output `8C` trung gian có thể tăng memory.
- Sigmoid/QReLU trên nhiều map có thể tạo thêm kernel nhỏ.
- DWConv + PWConv có thể nhanh hơn trên GPU vì backend tối ưu hơn.

#### Điều chỉnh đề xuất

Không đặt LBPConv là default. So sánh ba local branch:

1. `Identity`.
2. `DWConv 3x3 + PWConv`.
3. `LBPConv fixed filters + activation + PWConv`.

#### Tiêu chí giữ lại

LBPConv chỉ nên vào model chính nếu:

- Accuracy tăng rõ so với DWConv ở cost tương đương.
- Hoặc latency không tăng đáng kể và params giảm.
- Hoặc trên edge target cụ thể, fixed sparse filters có kernel tối ưu.

### 5.4. Learnable Structured Pruning

#### Ý tưởng

Học mask channel/center để loại bỏ phần không quan trọng. Structured pruning tốt hơn unstructured pruning vì có thể export thành model nhỏ thật.

#### Điểm đúng

- Hướng phù hợp cho lightweight.
- Channel-level pruning có khả năng giảm compute thật nếu materialize model.
- Có thể dùng sau khi architecture đã train ổn.

#### Rủi ro

- Soft mask trong training không làm inference nhanh nếu không export.
- Crispness loss có thể đẩy mask về 0/1 nhưng không đảm bảo sparsity mong muốn.
- Nếu prune quá sớm, accuracy collapse.
- Pruning center trong cluster phức tạp hơn pruning channel vì ảnh hưởng assignment.

#### Điều chỉnh đề xuất

Quy trình pruning nên gồm:

1. Train base student ổn định.
2. Bật mask với sparsity regularization.
3. Chọn threshold.
4. Export model đã cắt channel/center thật.
5. Fine-tune exported model.
6. Benchmark lại exported model, không benchmark soft-mask model.

#### Tiêu chí giữ lại

Pruning được tính là thành công nếu exported model:

- Ít params hơn.
- Ít latency hơn.
- Accuracy recover sau fine-tune.

## 6. Chiến lược kiến trúc đề xuất

### 6.1. Hai hướng nghiên cứu

#### Track A: HBCC-Latency

Mục tiêu: nhanh thật trên PyTorch/GPU hoặc edge runtime thông thường.

Đặc điểm:

- Dùng toán tử backend-friendly.
- Giảm token count sớm.
- Giảm số branch nhỏ.
- Hạn chế scatter/topk/permute không cần thiết.
- Hamming chỉ là ablation nếu chưa có bitwise kernel.
- Ưu tiên latency batch 1 và throughput batch 128.

#### Track B: HBCC-BitEdge

Mục tiêu: inference nhị phân thật trên phần cứng hỗ trợ bit operations.

Đặc điểm:

- Bit-pack activation similarity path.
- XNOR + popcount kernel.
- Có export inference riêng.
- BOPs được báo kèm latency thật.
- Không dùng để kết luận tốc độ nếu chỉ chạy PyTorch float.

Track A phải hoàn thành trước Track B. Nếu Track A chưa tạo được architecture Pareto tốt, Track B có nguy cơ tối ưu kernel cho một kiến trúc chưa tốt.

### 6.2. Kiến trúc v1 cho CIFAR-10

Mục tiêu của v1 là sửa các sai lệch lớn nhất nhưng vẫn giữ scope hợp lý.

```text
Input: RGB, optional XY coordinate augmentation

Stem / Point Reducer:
  32x32 -> 16x16
  channels: 3 hoặc 5 -> 32

Stage 1:
  size: 16x16
  op: local lightweight op hoặc region-local cluster
  goal: preserve local detail, avoid expensive global cluster

Stage 2:
  size: 8x8
  op: small HBCC block
  goal: mix local + limited contextual information

Stage 3:
  size: 4x4
  op: global or semi-global HBCC
  goal: context aggregation when token count is small

Stage 4:
  size: 2x2
  op: global cluster or lightweight MLP/conv block
  goal: final semantic feature

Head:
  global average pooling
  layer norm or batch norm depending on best ablation
  linear classifier
```

### 6.3. Thông số khởi đầu

Config đề xuất cho `HBCC-Latency-Tiny`:

```python
embed_dims = [32, 48, 96, 160]
depths = [1, 1, 2, 1]
mlp_ratio = 2.0
heads = [1, 2, 2, 4]
head_dim = [16, 16, 16, 16]
centers = [(2, 2), (2, 2), (2, 2), (2, 2)]
```

Config đề xuất cho `HBCC-Latency-Small`:

```python
embed_dims = [32, 64, 128, 192]
depths = [1, 2, 2, 1]
mlp_ratio = 3.0
heads = [1, 2, 4, 4]
head_dim = [16, 16, 16, 16]
centers = [(2, 2), (2, 2), (2, 2), (2, 2)]
```

Quy tắc:

- Không dùng một cấu hình `heads/head_dim` cố định cho mọi stage nếu gây waste.
- Không tăng `mlp_ratio` lên 4 mặc định nếu mục tiêu là lightweight.
- Không bật bottleneck/local/global split nếu ablation chưa chứng minh lợi ích.

### 6.4. Local branch policy

Stage sớm nên ưu tiên local branch vì:

- Ảnh còn nhiều điểm.
- Local edge/texture quan trọng.
- Global cluster sớm đắt và dễ noisy.

Thứ tự thử:

1. `DWConv 3x3 + BN + GELU + PWConv`.
2. `PointShrink/PointReducer` stride 2 khi cần giảm spatial.
3. `LBPConv` nếu muốn kiểm chứng fixed binary filters.

### 6.5. Similarity path policy

Thứ tự thử:

1. Cosine full precision baseline.
2. Hamming simulated bằng sign + matmul để đo accuracy.
3. Hamming với learnable scale.
4. Bit-packed Hamming chỉ trong Track B.

Không dùng BOPs để claim tốc độ ở bước 2 và 3.

### 6.6. Distillation policy

Teacher:

- Ưu tiên `ResNet-18 CIFAR` nếu accuracy cao và ổn định.
- Có thể dùng `CoC baseline full precision` nếu muốn transfer behavior gần hơn.

Student:

- `HBCC-Latency-Tiny`.
- `HBCC-Latency-Small`.

Loss mặc định:

```math
L = L_{CE}(y, p_s) + \lambda_{KD} T^2 KL(p_t^T || p_s^T)
```

Trong đó:

- `T`: temperature, mặc định thử `T = 4`.
- `lambda_KD`: mặc định thử `0.5`.

Feature distillation chỉ bật nếu:

- CE + KL vẫn mất accuracy nhiều.
- Teacher/student có stage feature map tương thích.

## 7. Roadmap thực hiện

### Phase 0: Measurement và profiler đúng

Mục tiêu: mọi kết luận sau này dựa trên số đo đáng tin.

Việc cần làm:

- Benchmark batch `1, 16, 64, 128`.
- Đo latency và throughput riêng.
- Dùng `torch.profiler` để lấy top operators.
- Báo số kernel/operator count nếu có thể.
- Tách kết quả CPU/GPU.
- Dùng env benchmark mặc định: `ImgCaption`.

Deliverables:

- Script benchmark thống nhất.
- Bảng kết quả baseline hiện tại.
- Operator profile cho ResNet-18, CoC baseline, HBCC hiện tại.

Acceptance:

- Có thể tái chạy benchmark bằng một lệnh.
- Kết quả ghi rõ device, torch version, batch size, warmup/runs.

### Phase 1: Baseline matrix

Mục tiêu: so sánh HBCC với baseline đúng.

Models tối thiểu:

- `ResNet-18`: accuracy reference.
- `MobileNetV2` hoặc `ShuffleNetV2`: lightweight latency reference.
- `CoC baseline`: architecture reference.
- `HBCC current`: current proposal reference.

Metrics:

- Top-1 accuracy.
- Params.
- Operator-aware FLOPs/MACs.
- Latency batch 1.
- Throughput batch 16/64/128.
- Peak memory.

Acceptance:

- Có bảng baseline matrix.
- Mỗi baseline dùng cùng dataset, transform, device, benchmark protocol.

### Phase 2: Sửa nền tảng CoC/HBCC

Mục tiêu: đưa implementation về gần nền tảng paper hơn trước khi tối ưu sâu.

Việc cần làm:

- Thêm optional coordinate augmentation.
- Thêm reducer giảm sớm `32x32 -> 16x16`.
- Đảm bảo shape qua stage là `16 -> 8 -> 4 -> 2`.
- Tách rõ local preprocessing và point reduction.
- Đảm bảo config có thể bật/tắt từng thay đổi.

Acceptance:

- Shape trace đúng.
- Accuracy không collapse.
- Latency HBCC mới tốt hơn HBCC current ở batch 1 hoặc batch 128.

### Phase 3: Pareto architecture search nhẹ

Mục tiêu: tìm cấu hình width/depth/head/mlp tốt nhất trong ngân sách nhỏ.

Grid khởi đầu:

```text
embed_dims:
  A: [24, 48, 96, 160]
  B: [32, 48, 96, 160]
  C: [32, 64, 128, 192]

depths:
  A: [1, 1, 1, 1]
  B: [1, 1, 2, 1]
  C: [1, 2, 2, 1]

mlp_ratio:
  2.0, 3.0

heads/head_dim:
  small per-stage settings, avoid over-wide early stages
```

Không train full 200 epochs cho mọi điểm grid ban đầu. Dùng short training proxy:

- 20-50 epochs để lọc cấu hình yếu.
- Full training chỉ cho top candidates.

Acceptance:

- Có ít nhất 3 candidate nằm trên Pareto frontier.
- Loại bỏ cấu hình chậm/accuracy thấp bằng tiêu chí rõ ràng.

### Phase 4: Ablation từng kỹ thuật

Mục tiêu: kiểm chứng từng đề xuất độc lập.

Ablation groups:

| Group | Variants | Câu hỏi |
|---|---|---|
| Coordinate | RGB vs RGB+XY | XY có giúp CoC/HBCC không? |
| Reducer | stride1 vs stride2 | Giảm token sớm ảnh hưởng accuracy/latency thế nào? |
| Similarity | cosine vs Hamming simulated | Hamming giữ accuracy không? |
| Local branch | identity vs DWConv vs LBPConv | Local branch nào Pareto tốt nhất? |
| Bottleneck | off vs on | Bottleneck có đáng latency không? |
| Channel shuffle | off vs on | Shuffle có tăng accuracy đủ để giữ không? |
| Pruning mask | off vs soft mask | Mask có học được sparsity hữu ích không? |

Acceptance:

- Mỗi kỹ thuật có kết luận `keep`, `drop`, hoặc `defer`.
- Không giữ kỹ thuật chỉ vì hợp lý về lý thuyết.

### Phase 5: Distillation và fine-tune

Mục tiêu: recover accuracy cho student nhỏ.

Quy trình:

1. Train teacher hoặc dùng checkpoint teacher tốt nhất.
2. Train student bằng CE + KD.
3. So sánh với student CE-only.
4. Thử temperature và lambda nhỏ.
5. Fine-tune student tốt nhất bằng CE-only vài epoch nếu cần.

Acceptance:

- KD tăng accuracy student hoặc giúp convergence ổn định hơn.
- KD không làm latency thay đổi.

### Phase 6: Structured pruning export

Mục tiêu: biến sparsity học được thành model nhỏ và nhanh thật.

Quy trình:

1. Chọn student tốt nhất.
2. Thêm pruning mask cho channel hoặc center.
3. Train với sparsity/crispness regularization.
4. Chọn threshold.
5. Export model đã cắt thật.
6. Fine-tune exported model.
7. Benchmark lại exported model.

Acceptance:

- Exported model có params thấp hơn.
- Exported model latency thấp hơn soft-mask model.
- Accuracy recover trong ngưỡng chấp nhận.

### Phase 7: HBCC-BitEdge

Chỉ thực hiện khi Track A đã có architecture tốt.

Mục tiêu:

- Implement bit-packed Hamming inference.
- Tách training graph và inference graph.
- Đo latency thực trên target hardware hoặc CUDA extension.

Acceptance:

- Similarity path không còn dùng float matmul.
- BOPs report đi kèm latency thật.
- Có fallback path để kiểm chứng correctness với Hamming simulated.

## 8. Thiết kế thí nghiệm

### 8.1. Metrics bắt buộc

Mỗi model phải báo:

| Metric | Ý nghĩa | Ghi chú |
|---|---|---|
| `Top-1 Accuracy` | Hiệu quả model | CIFAR-10 validation/test |
| `Params` | Kích thước model | Total/trainable |
| `FLOPs/MACs` | Chi phí toán học float/int | Nên operator-aware |
| `BOPs` | Chi phí nhị phân lý thuyết | Chỉ dùng cho Hamming/bit path |
| `Latency b1` | Thời gian phản hồi | Quan trọng cho real-time |
| `Throughput b16/b64/b128` | Hiệu suất batch | So với paper và deployment batch |
| `Peak Memory` | Khả năng deploy/train | GPU memory hoặc activation estimate |
| `Operator Profile` | Bottleneck runtime | Top ops từ profiler |

### 8.2. Benchmark protocol

Mặc định:

```text
device: cuda nếu có
env: ImgCaption
input: 3x32x32 hoặc 5x32x32 nếu bật coordinate augmentation
warmup: >= 30 iterations
runs: đủ lớn để ổn định
batch sizes: 1, 16, 64, 128
model.eval()
torch.no_grad()
synchronize trước và sau timing
```

Không synchronize sau từng iteration trừ khi đang đo latency rất thận trọng, vì điều này có thể phạt kernel nhỏ và làm throughput batch lớn kém đại diện. Nên có hai chế độ:

- `latency_strict`: synchronize mỗi iteration.
- `throughput_stream`: synchronize sau toàn bộ loop.

### 8.3. Logging format

Mỗi experiment nên xuất một record dạng:

```json
{
  "model_name": "hbcc_latency_tiny",
  "config_id": "dims32_48_96_160_depth1121_mlp2",
  "dataset": "cifar10",
  "device": "cuda",
  "torch_version": "...",
  "params": 0,
  "flops": 0,
  "bops": 0,
  "acc1": 0.0,
  "latency_ms_b1": 0.0,
  "throughput_b16": 0.0,
  "throughput_b64": 0.0,
  "throughput_b128": 0.0,
  "peak_memory_mb": 0.0,
  "notes": ""
}
```

### 8.4. Quy tắc kết luận

Không được kết luận một model tốt hơn nếu:

- Chỉ ít params hơn nhưng latency tệ hơn nhiều.
- Chỉ FLOPs thấp hơn nhưng profiler cho thấy operator overhead cao.
- Chỉ BOPs thấp hơn nhưng inference vẫn dùng float matmul.
- Accuracy cao hơn nhưng params/latency vượt xa mục tiêu lightweight.

Một model được xem là Pareto tốt nếu không có model khác đồng thời:

- Accuracy cao hơn hoặc bằng.
- Params thấp hơn hoặc bằng.
- Latency thấp hơn hoặc bằng.
- Memory thấp hơn hoặc bằng.

và ít nhất một tiêu chí tốt hơn rõ ràng.

## 9. Acceptance criteria tổng thể

### 9.1. Research success criteria

Tối thiểu cần đạt:

- Có ít nhất một model nhỏ hơn ResNet-18 rõ rệt.
- Accuracy chỉ giảm trong ngưỡng chấp nhận được.
- Latency không tệ hơn lightweight baseline như MobileNetV2/ShuffleNetV2.
- HBCC-Latency mới nhanh hơn HBCC hiện tại ở batch 1 và batch 128.

Ngưỡng đề xuất ban đầu:

```text
Params: <= 40% ResNet-18
Accuracy drop: <= 2% absolute so với ResNet-18 CIFAR baseline
Latency b1: <= HBCC current và càng gần lightweight conv baseline càng tốt
Throughput b128: >= HBCC current
```

Các ngưỡng này có thể điều chỉnh sau khi baseline matrix đầy đủ.

### 9.2. Engineering acceptance criteria

- Có benchmark script tái lập được.
- Có config rõ ràng cho từng ablation.
- Có result table chuẩn hóa.
- Có profiler report cho candidate chính.
- Có shape trace cho architecture mới.
- Có checkpoint và config mapping cho model tốt nhất.

### 9.3. Scientific acceptance criteria

- Mỗi claim chính có số đo hỗ trợ.
- BOPs không bị dùng để thay thế latency.
- Hamming, LBPConv, pruning đều có ablation riêng.
- Nếu một kỹ thuật bị loại, lý do loại được ghi bằng metric.

## 10. Risk register

| Risk | Mức độ | Dấu hiệu | Giảm thiểu |
|---|---:|---|---|
| Hamming làm accuracy giảm | Cao | Accuracy drop lớn khi bật sign | Distillation, learnable scale, chỉ dùng stage muộn |
| LBPConv chậm hơn DWConv | Trung bình | Latency tăng dù params giảm | Giữ DWConv làm default, LBP là ablation |
| Pruning mask không tăng tốc | Cao | Soft-mask model params/runtime không đổi | Export model đã prune thật |
| Coordinate augmentation làm stem thay đổi phức tạp | Thấp | Shape/input mismatch | Flag `use_coord`, test shape |
| Profiler thiếu operator cost | Cao | FLOPs thấp nhưng latency cao | Dùng torch profiler/operator-aware summary |
| Search quá rộng tốn compute | Trung bình | Quá nhiều config full train | Short proxy training trước |
| BitEdge scope quá lớn | Cao | Cần custom kernel/export | Defer đến Phase 7 |

## 11. Implementation checklist

### Measurement

- [ ] Benchmark batch `1, 16, 64, 128`.
- [ ] Ghi device, env, torch version.
- [ ] Thêm profiler top operators.
- [ ] Tách latency strict và throughput stream.
- [ ] Lưu results dạng JSON/CSV chuẩn.

### Baselines

- [ ] ResNet-18 CIFAR.
- [ ] MobileNetV2 hoặc ShuffleNetV2.
- [ ] CoC baseline.
- [ ] HBCC current.

### Architecture

- [ ] Optional coordinate augmentation.
- [ ] Reducer giảm `32x32 -> 16x16`.
- [ ] Stage sizes `16 -> 8 -> 4 -> 2`.
- [ ] Config per-stage dims/depths/heads.
- [ ] Local branch variants.
- [ ] Similarity variants.

### Training

- [ ] Teacher checkpoint.
- [ ] CE-only student baseline.
- [ ] KD student.
- [ ] Feature distillation optional.

### Pruning

- [ ] Train mask.
- [ ] Choose threshold.
- [ ] Export pruned model.
- [ ] Fine-tune exported model.
- [ ] Benchmark exported model.

### Reporting

- [ ] Pareto table.
- [ ] Current issue -> fix -> metric table.
- [ ] Ablation summary.
- [ ] Final recommendation.

## 12. Đề xuất thứ tự ưu tiên

Thứ tự làm nên là:

1. Measurement/profiler.
2. Baseline matrix.
3. Stage size/reducer đúng.
4. Coordinate augmentation.
5. Architecture tiny/small sweep.
6. Distillation.
7. Hamming ablation.
8. Local branch ablation, gồm LBPConv.
9. Structured pruning export.
10. BitEdge kernel nếu vẫn cần.

Không nên bắt đầu bằng bitwise kernel hoặc pruning trước khi có architecture Pareto tốt. Nếu architecture nền chưa tốt, mọi tối ưu sau đó có thể chỉ cải thiện một thiết kế sai hướng.

## 13. Assumptions

- Dataset chính ban đầu là CIFAR-10.
- Env benchmark ưu tiên là `ImgCaption`.
- ResNet-18 là accuracy reference, không phải lightweight latency baseline.
- Lightweight latency baseline nên bổ sung MobileNetV2 hoặc ShuffleNetV2.
- Phase đầu chưa implement bitwise kernel.
- BOPs trong phase đầu chỉ là theoretical metric.
- Mục tiêu chính là tìm model Pareto tốt, không phải giữ mọi ý tưởng ban đầu.
- File này là tài liệu định hướng nghiên cứu và triển khai, chưa phải kết quả cuối cùng.

## 14. Kết luận

Hướng Lightweight HBCC có tiềm năng, nhưng cần điều chỉnh trọng tâm. Những đề xuất như Hamming, LBPConv và structured pruning đều có cơ sở, nhưng chỉ trở thành lợi thế thật nếu được triển khai và đo đúng.

Kế hoạch nghiên cứu nên bắt đầu từ những vấn đề chắc chắn nhất:

- Đo đúng latency và operator cost.
- Giảm token count sớm.
- Bám lại các thành phần quan trọng của CoC gốc như coordinate augmentation và point reduction.
- Tìm architecture Pareto nhỏ trước.
- Sau đó mới thêm Hamming, LBPConv, pruning và bitwise inference.

Một model lightweight tốt không phải là model có ít tham số nhất, mà là model đạt trade-off tốt nhất giữa accuracy, latency, memory và khả năng triển khai thật.
