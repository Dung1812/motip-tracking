# Hợp đồng dữ liệu giữa các module (interface.md)

Phiên bản: **1.0** (chốt ngày 25/11/2026; còn 2 mục chờ ở phần 8)
Người phụ trách: G4-TV5 Phạm Nguyễn Hữu Dũng

Mọi module trao đổi dữ liệu theo đúng định dạng dưới đây. Muốn đổi định dạng thì báo TV5 cập nhật file này trước, không tự đổi trong code.

## 1. Quy ước chung

| Mục | Quy ước |
|---|---|
| Frame ảnh | `numpy.ndarray`, shape `(H, W, 3)`, kiểu `uint8`, màu **BGR** (như OpenCV đọc ra) |
| Số thứ tự frame | `frame_idx`, **bắt đầu từ 1**, tăng dần (theo chuẩn MOT) |
| Toạ độ box | `[x1, y1, x2, y2]`, đơn vị **pixel của frame gốc**, kiểu `float`; (x1, y1) góc trên trái, (x2, y2) góc dưới phải |
| Điểm tin cậy | `score`, kiểu `float` trong khoảng 0 đến 1 |
| Số chiều đặc trưng | **C = 256** |
| ID hiển thị | `track_id`, số nguyên dương tăng dần từ 1, **không tái sử dụng** trong cùng một video |

## 2. TV1 - Detector (`detector_adapter.py`)

Mô hình: Deformable DETR của MOTIP chính thức (repo MCG-NJU/MOTIP), cấu hình `r50_deformable_detr_motip_dancetrack.yaml`.

```python
model, cfg, kind = load_official_motip(repo, config_path, checkpoint_path, device)
adapter = DetectorAdapter(threshold=0.30, max_detections=100, class_index=0, score_type="sigmoid")
raw = model(frames=nested_tensor, part="detr")
batch = adapter.convert(raw, image_size_hw=frame.shape[:2])   # -> DetectionBatch
```

Đầu ra `DetectionBatch` (một frame):

| Trường | Shape | Ý nghĩa |
|---|---|---|
| `boxes_xyxy` | `[N, 4]` | Box pixel gốc `[x1, y1, x2, y2]`, đã cắt trong khung hình |
| `scores` | `[N]` | `sigmoid(pred_logits)`, đã lọc ngưỡng **0.30** (`DET_THRESH`) |
| `labels` | `[N]` | Lớp (DanceTrack: 0 = người) |
| `embeddings` | `[N, 256]` | **Đặc trưng đối tượng** từ decoder DETR cuối (`hs[-1]`), *không phải* ID embedding |
| `query_indices` | `[N]` | Chỉ số query gốc, dùng để kiểm tra box, score, embedding cùng hàng |

N thay đổi theo từng frame. Box, score và embedding luôn lấy theo cùng `query_indices`.

## 3. TV3 - ID Decoder (`id_decoder.py`)

| Thông số | Giá trị |
|---|---|
| Lớp dùng cho tracker | `IDDecoderForTracker` |
| Số nhãn ID | **K = 50** (nhãn thường 0 đến 49) |
| Token newborn | chỉ số **50** (K) |
| Số chiều đặc trưng | C = 256 |
| Cách gọi | `decoder(batch, det_feats)`; thuộc tính `decoder.num_labels` |
| Đầu ra | ma trận xác suất **`(N, K)` = `(N, 50)`**, `float32` |

- Cột newborn **đã bị bỏ** khỏi đầu ra; xác suất newborn = 1 trừ tổng xác suất của 50 nhãn thường.
- `batch` (đặc trưng lịch sử, nhãn ID, `dt`, mask) do **TV2 dựng**; `det_feats` là `embeddings` của TV1, shape `(N, 256)`.
- Nhãn 0 đến 49 là "ô" trong từ điển ID, được tái sử dụng; **không** phải `track_id` hiển thị.

## 4. TV2 - Quản lý track (`tv2_interface.py`)

```python
tracker = TrackerTV2(decoder=<IDDecoderForTracker của TV3>,
                     include_lost=False, include_terminated=False, strict=False)
tracker.reset()                                  # trước mỗi video mới
tracks = tracker.update(detections, frame_idx)   # mỗi frame
tracker.finish()                                 # khi hết video
```

Đầu vào `detections`: danh sách, mỗi phần tử
```python
{"bbox": [x1, y1, x2, y2], "score": 0.92, "embedding": <tensor hoặc numpy shape (256,)>}
```

Đầu ra `tracks`: danh sách, mỗi phần tử
```python
{"track_id": 3, "bbox": [x1, y1, x2, y2], "score": 0.88, "state": "tracked"}
```

| state | Ý nghĩa | Mặc định có trả không |
|---|---|---|
| `new` | Track vừa xuất hiện ở frame này | Có |
| `tracked` | Đang theo dõi bình thường | Có |
| `lost` | Tạm mất (bị che), chưa kết thúc; bbox = vị trí cuối | Không (bật `include_lost=True`) |
| `terminated` | Kết thúc ở frame này | Không (bật `include_terminated=True`) |

- **TV2 là bên đổi nhãn 0 đến 49 thành `track_id`** (tăng dần, không tái sử dụng).
- Khi chưa có decoder của TV3, `decoder=None` sẽ dùng `AppearanceMockDecoder` (giả lập).
- `frame_idx < 1` hoặc không tăng dần sẽ báo `ValueError`.

## 5. TV5 - Pipeline và giao diện (`app.py`, `video_export.py`)

Giao diện không gọi riêng từng module, chỉ gọi một đối tượng `Pipeline` có 3 hàm:

```python
pipeline = Pipeline(checkpoint_path=..., device="cuda")  # tạo một lần
pipeline.reset()                                          # trước mỗi video
tracks = pipeline.process(frame, frame_idx)               # mỗi frame -> list[Track] (mục 4)
pipeline.finish()                                         # hết video
```

Bên trong `process`: TV1 phát hiện → **chuyển `DetectionBatch` thành danh sách dict** → TV2 `update` (TV2 gọi decoder TV3 bên trong).
Hàm chuyển đổi do TV5 viết trong `Pipeline`, không ai phải sửa code đã có:

```python
def batch_to_detections(b):
    return [{"bbox": b.boxes_xyxy[i].tolist(), "score": float(b.scores[i]),
             "embedding": b.embeddings[i]} for i in range(len(b.scores))]
```

Hiện giao diện chạy với `MockPipeline` (`mock_pipeline.py`), cùng 4 hàm trên.

## 6. File kết quả (đầu vào của TV4)

Theo quy chuẩn của TV4:

- Một file cho mỗi video, **trùng tên video**: `<sequence_name>.txt` (ví dụ `dancetrack0004.txt`).
- Mỗi dòng 10 cột chuẩn MOTChallenge, **không có dấu cách** sau dấu phẩy:

```
frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1
```

- `bb_left, bb_top` = góc trên trái; `bb_width = x2 - x1`, `bb_height = y2 - y1`.
- Chỉ ghi track có state `new` và `tracked`.
- Ví dụ: `1,3,433.00,216.00,80.00,90.00,0.80,-1,-1,-1`
- Video kết quả: `<sequence_name>_tracked.mp4`.

## 7. Môi trường chạy mô hình thật

| Phần | Môi trường |
|---|---|
| Giao diện, dữ liệu giả lập | Python 3.13, Windows, `requirements.txt` hiện tại |
| Mô hình MOTIP chính thức (TV1, demo thật) | Python 3.12, PyTorch 2.4.0, torchvision 0.19.0, cần biên dịch Deformable Attention; chạy trên **Linux có CUDA, WSL2 hoặc Google Colab** |

Checkpoint `.pth`, thư mục `external/`, `checkpoints/`, `outputs/` **không đẩy lên Git**.

## 8. Còn chờ xác nhận

- [x] TV1: số chiều embedding (256), ngưỡng score (0.30)
- [x] TV2: không trả `lost` mặc định; TV2 đổi nhãn thành `track_id`
- [x] TV3: tên lớp, K, C, cách gọi, đầu ra
- [x] TV4: định dạng và tên file kết quả
- [ ] TV1: tên file checkpoint đã tải và thư mục Drive chung của nhóm
- [ ] TV3: cú pháp `load_id_decoder` (tham số, trả về lớp nào) và checkpoint ID Decoder; hoặc nhóm dùng **full MOTIP checkpoint** chính thức (đã gồm ID Decoder)