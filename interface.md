# Hợp đồng dữ liệu giữa các module (interface.md)

Phiên bản: 0.1 (bản nháp, cả nhóm góp ý rồi chốt)
Người phụ trách: G4-TV5 Phạm Nguyễn Hữu Dũng

Mọi module trao đổi dữ liệu theo đúng định dạng dưới đây. Muốn đổi định dạng thì báo TV5 cập nhật file này trước, không tự đổi trong code.

## 1. Quy ước chung

| Mục | Quy ước |
|---|---|
| Frame ảnh | `numpy.ndarray`, shape `(H, W, 3)`, kiểu `uint8`, màu **BGR** (như OpenCV đọc ra) |
| Số thứ tự frame | `frame_idx`, **bắt đầu từ 1** (theo chuẩn MOT) |
| Toạ độ box | `[x1, y1, x2, y2]`, đơn vị **pixel** của frame gốc, kiểu `float`; (x1, y1) góc trên trái, (x2, y2) góc dưới phải |
| Điểm tin cậy | `score`, kiểu `float` trong khoảng 0 đến 1 |
| ID | `track_id`, số nguyên dương, không tái sử dụng ID đã kết thúc trong cùng một video |

## 2. Detection (đầu ra của TV1 - detector)

```python
{"bbox": [x1, y1, x2, y2], "score": 0.92, "embedding": <torch.Tensor shape (C,)>}
```

- `embedding`: đặc trưng vật thể từ DETR, dùng cho ID Decoder của TV3. Giá trị C ghi rõ khi TV1 chốt mô hình.

## 3. Track (đầu ra của TV2 + TV3, đầu vào của giao diện TV5)

```python
{"track_id": 3, "bbox": [x1, y1, x2, y2], "score": 0.88, "state": "tracked"}
```

`state` là một trong bốn giá trị (TV2 quản lý):

| state | Ý nghĩa |
|---|---|
| `new` | Track vừa xuất hiện ở frame này |
| `tracked` | Track đang được theo dõi bình thường |
| `lost` | Tạm mất (bị che khuất), chưa kết thúc |
| `terminated` | Đã kết thúc, không dùng lại ID |

Giao diện chỉ vẽ các track có state `new` và `tracked`.

## 4. Hàm mà giao diện gọi (Pipeline)

Giao diện không gọi riêng từng module. Mọi module được gói trong một đối tượng `Pipeline`:

```python
pipeline = Pipeline(checkpoint_path="weights/motip.pth")  # tạo một lần
pipeline.reset()                                           # trước mỗi video mới
tracks = pipeline.process(frame, frame_idx)                # mỗi frame, trả list[Track]
```

Bên trong `process`: TV1 phát hiện → TV3 dự đoán ID → TV2 cập nhật trạng thái track.
Hiện tại dùng `MockPipeline` trong `mock_pipeline.py` (dữ liệu giả lập) với cùng ba hàm này.

## 5. File kết quả tracking (đầu vào của TV4 - đánh giá)

Định dạng MOT, mỗi dòng một track ở một frame, phân tách bằng dấu phẩy:

```
frame,id,x,y,w,h,score,-1,-1,-1
```

- `x, y` = góc trên trái; `w = x2 - x1`, `h = y2 - y1` (đổi từ `bbox`).
- Ví dụ: `1,3,433.00,216.00,80.00,90.00,0.80,-1,-1,-1`

## 6. Việc cần cả nhóm xác nhận

- [ ] TV1: số chiều C của `embedding`, ngưỡng `score` tối thiểu
- [ ] TV2: có trả cả track `lost` cho giao diện hay không
- [ ] TV3: tên lớp và cách nạp checkpoint
- [ ] TV4: định dạng file kết quả trên đã đúng với công cụ đánh giá (TrackEval) chưa
