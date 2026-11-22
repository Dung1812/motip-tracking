# Phần G4-TV2: Track manager và inference

Theo hợp đồng `interface.md` v0.1. Chỉ cần `numpy` (và `opencv-python` nếu chạy với video thật; `pytest` để chạy test).

## File và việc của từng file
| File | Vai trò |
|---|---|
| `track_manager.py` | Vòng đời track (new, lost, duplicate, terminated), cấp và thu hồi nhãn ID |
| `track_memory.py` | Lịch sử ID: cửa sổ T frame cho decoder và lưu trữ quỹ đạo dài hạn |
| `tracking_inference.py` | Suy luận từng frame theo mục 3.4 của MOTIP; khai báo giao diện `IDDecoder` cho TV3 |
| `tv2_interface.py` | **Cổng giao tiếp theo hợp đồng**: `TrackerTV2.update(detections, frame_idx) -> tracks`, ghi file MOT |
| `tv1_bridge.py` | **Cầu nối với TV1**: kiểm tra detection, đọc `detections.csv`, detector giả lập, chuyển đổi định dạng |
| `run_tv2_pipeline.py` | Chạy thử TV1 -> TV3 -> TV2 (giả lập hoặc video thật), ghi file MOT |
| `test_tracking.py`, `test_interface.py` | 86 unit test (trạng thái track, nhãn, bộ nhớ, đúng hợp đồng) |

## Cách dùng trong Pipeline
```python
from tv2_interface import TrackerTV2
tracker = TrackerTV2(decoder=<ID Decoder của TV3>)   # tạo một lần
tracker.reset()                                       # trước mỗi video mới
tracks = tracker.update(detections, frame_idx)        # frame_idx bắt đầu từ 1
```
`detections`: `[{"bbox": [x1,y1,x2,y2], "score": 0.92, "embedding": <Tensor (C,)>}]`
`tracks`: `[{"track_id": 3, "bbox": [...], "score": 0.88, "state": "new"|"tracked"|"lost"|"terminated"}]`

Mặc định chỉ trả `new` và `tracked` (giao diện chỉ vẽ hai state này). Bật `include_lost=True` / `include_terminated=True` nếu cần.
`track_id` không bao giờ tái sử dụng trong một video.

## Chạy thử
```
pip install numpy pytest opencv-python
python -m pytest -q                                  # 86 test đạt
python run_tv2_pipeline.py --mock occlusion          # dữ liệu giả lập, ghi outputs/tv2_result.txt
python run_tv2_pipeline.py --video data/sample.mp4 --detections outputs/detections.csv
```

## Cần xác nhận với nhóm (hợp đồng mục 6)
- TV1: số chiều `C` của embedding, ngưỡng `score` tối thiểu. Dùng `tv1_bridge.report_detections(dets)` để in ra C và khoảng score.
- TV2: mặc định không trả track `lost`. Chốt với TV5 xem giao diện có cần không.
- TV3: tên lớp và cách nạp checkpoint. TV3 cần cung cấp đối tượng có `num_labels` và `__call__(batch, det_feats) -> (N, K)` xác suất; xem `IDDecoder` trong `tracking_inference.py`. `TorchIDDecoderAdapter` mới là khung, chưa kiểm thử với model thật.
- TV4: định dạng MOT đã đúng với TrackEval chưa (dòng mẫu `1,3,433.00,216.00,80.00,90.00,0.80,-1,-1,-1` đã có test).

## Giới hạn hiện tại
- Chưa chạy với detector, decoder thật. Khi thiếu decoder, `TrackerTV2` dùng `AppearanceMockDecoder` (giả lập, có cảnh báo).
- `detections.csv` của `detr_demo.py` không có embedding; `tv1_bridge.CsvDetector` dùng histogram màu thay thế.
