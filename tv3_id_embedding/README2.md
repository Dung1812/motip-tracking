# README2: Cách nạp checkpoint cho dự án

Trả lời yêu cầu của TV5 (`interface.md` mục 8, hai ô còn trống: checkpoint của TV1 và `load_id_decoder` của TV3).
Bổ sung cho `README1.md` (TV3). Đặt file này trong `tv3_id_embedding/` hoặc ở thư mục gốc đều được.

## 1. Dự án có 3 loại checkpoint, đừng nhầm

| Loại | Ai tạo / lấy ở đâu | Chứa gì | Nạp bằng | Trạng thái |
|---|---|---|---|---|
| **A. ID Decoder của nhóm** | TV3, `id_train.py` -> `save_id_decoder` | Chỉ ID Decoder + ID Dictionary, kèm `config` | `load_id_decoder(path, device)` (`id_decoder.py`) | Chạy đúng với tracker của TV2. Mới huấn luyện trên **dữ liệu tổng hợp** |
| **B. Full MOTIP chính thức** | MODEL_ZOO của MCG-NJU/MOTIP | Detector + trajectory modeling + ID Decoder chính thức | `load_official_motip(...)` (`detector_adapter.py`, TV1) | Dùng được cho **phần detector**. Phần ID Decoder chính thức **chưa** nối được với tracker của TV2 (xem mục 4) |
| **C. DETR pretrain (chỉ detector)** | MODEL_ZOO / COCO | Chỉ detector | `load_official_motip(...)`, tự nhận dạng qua khoá `bbox_embed.` | Dùng được cho detector |

Tóm lại: **detector** lấy từ B hoặc C, **ID Decoder** hiện lấy từ A. Hai checkpoint này độc lập với nhau, cần hai đường dẫn riêng.

## 2. Nạp từng loại

### 2.1. Loại A: ID Decoder của TV3

```python
from id_decoder import load_id_decoder

decoder = load_id_decoder("checkpoints/id_decoder_synth.pth", device="cuda")
print(decoder.num_labels)      # K, mặc định 50
```

- Trả về `IDDecoderForTracker`: có `.num_labels` và `__call__(batch, det_feats) -> np.ndarray (N, K)` float32, đã softmax.
- Cấu hình (K, C, d_model, số layer...) nằm **trong file**, không phải truyền lại.
- Nạp `strict=True`, `weights_only=True`: file thiếu hoặc thừa khoá, hoặc không phải do `save_id_decoder` tạo ra, sẽ báo lỗi chứ không nạp ngầm.
- Tạo checkpoint:

```bash
python id_train.py --steps 5000 --save checkpoints/id_decoder_synth.pth          # đúng cỡ bài báo, nên dùng GPU
python id_train.py --steps 1500 --C 32 --T 8 --layers 2 --d-model 128 --heads 4 \
                   --save checkpoints/id_decoder_small.pth                       # bản nhỏ để kiểm tra
```

**Ràng buộc:** `C` của decoder phải bằng `C` của detector (256). `InferenceConfig.num_labels` phải bằng `decoder.num_labels`. Nếu khác, `IDDecoderForTracker` hoặc `MOTIPTracker` báo `ValueError`.

### 2.2. Loại B và C: detector của TV1

```python
from pathlib import Path
import torch
from detector_adapter import load_official_motip

device = torch.device("cuda")
model, cfg, kind = load_official_motip(
    repo=Path("external/MOTIP"),
    config_path=Path("external/MOTIP/configs/r50_deformable_detr_motip_dancetrack.yaml"),
    checkpoint_path=Path("checkpoints/<TÊN_FILE_THẬT>.pth"),
    device=device,
)
print(kind)   # "full_MOTIP" hoặc "DETR_pretrain"
```

- `config` và `checkpoint` phải là **một cặp khớp nhau** (cùng bản trong MODEL_ZOO). Hàm nạp `strict=True` và không dùng `strict=False` để che lỗi.
- Hàm tự `chdir` vào repo MOTIP khi nạp (cần `external/MOTIP` đã clone và biên dịch Deformable Attention), rồi trả về thư mục cũ.
- Gọi detector cho một frame:

```python
from detector_adapter import DetectorAdapter, make_nested_frame

adapter = DetectorAdapter(threshold=0.30, max_detections=100, class_index=0, score_type="sigmoid")
nested = make_nested_frame(frame_bgr, device)            # frame BGR (H,W,3) uint8
with torch.inference_mode():
    raw = model(frames=nested, part="detr")              # pred_logits, pred_boxes, outputs
batch = adapter.convert(raw, image_size_hw=frame_bgr.shape[:2])   # DetectionBatch
```

- Môi trường: Python 3.12, PyTorch 2.4.0, torchvision 0.19.0, Linux có CUDA, WSL2 hoặc Colab (`interface.md` mục 7). Môi trường Windows Python 3.13 của giao diện **không** nạp được phần này.

### 2.3. Xem nội dung một checkpoint trước khi nạp

```python
from id_decoder import inspect_checkpoint
inspect_checkpoint("checkpoints/<file>.pth")              # in khoá + shape, bỏ qua phần "detr."
```

## 3. Ghép vào `Pipeline` (việc của TV5)

`interface.md` mục 5 hiện ghi `Pipeline(checkpoint_path=..., device=...)` với **một** đường dẫn. Vì detector và decoder là hai checkpoint riêng, **đề nghị TV5 đổi thành hai tham số** (cập nhật `interface.md` trước, đúng quy ước nhóm). Bản phác thảo, **chưa chạy thử**:

```python
# pipeline.py
import sys
from pathlib import Path
import torch

for d in ("tv1_detr", "tv2_mot_baseline", "tv3_id_embedding"):
    sys.path.insert(0, d)

from detector_adapter import DetectorAdapter, load_official_motip, make_nested_frame
from id_decoder import load_id_decoder
from tracking_inference import InferenceConfig
from tv2_interface import TrackerTV2


def batch_to_detections(b):
    return [{"bbox": b.boxes_xyxy[i].tolist(), "score": float(b.scores[i]),
             "embedding": b.embeddings[i]} for i in range(len(b.scores))]


class Pipeline:
    def __init__(self, detector_ckpt, decoder_ckpt, motip_repo="external/MOTIP",
                 motip_config="external/MOTIP/configs/r50_deformable_detr_motip_dancetrack.yaml",
                 device="cuda"):
        self.device = torch.device(device)
        self.model, self.cfg, self.kind = load_official_motip(
            Path(motip_repo), Path(motip_config), Path(detector_ckpt), self.device)
        self.adapter = DetectorAdapter(threshold=0.30, max_detections=100,
                                       class_index=0, score_type="sigmoid")
        self.decoder = load_id_decoder(decoder_ckpt, device=device)
        self.tracker = TrackerTV2(
            decoder=self.decoder,
            config=InferenceConfig(num_labels=self.decoder.num_labels),
            include_lost=False, include_terminated=False)

    def reset(self):
        self.tracker.reset()

    def process(self, frame, frame_idx):
        nested = make_nested_frame(frame, self.device)
        with torch.inference_mode():
            raw = self.model(frames=nested, part="detr")
        batch = self.adapter.convert(raw, image_size_hw=frame.shape[:2])
        return self.tracker.update(batch_to_detections(batch), frame_idx)

    def finish(self):
        return self.tracker.finish()
```

Lưu ý:
- `embeddings` là torch.Tensor; `TrackerTV2` tự đổi sang numpy (`to_numpy`), không cần `.numpy()` ở `Pipeline`.
- Nên nạp checkpoint **một lần** khi tạo `Pipeline`, không nạp trong `process` hoặc `reset`.
- Giao diện (Streamlit, Windows, Python 3.13) vẫn dùng `MockPipeline`. Chỉ khi chạy trên môi trường mục 2.2 mới dùng `Pipeline` thật.

## 4. Chưa làm được: dùng ID Decoder trong checkpoint MOTIP chính thức

Checkpoint loại B có ID Decoder riêng, nhưng **không nạp thẳng được** vào `MOTIPIDDecoder` (tên khoá, số layer, cách mã hoá vị trí có thể khác; xem README1 mục 4.5) và cũng không có `IDDecoderForTracker` cho nó. Hai hướng, cần nhóm chốt (đây là mục cuối còn chờ ở `interface.md` mục 8):

1. **Hướng hiện tại (A + B/C):** detector chính thức + ID Decoder của TV3. Chạy được về mặt giao diện dữ liệu, nhưng decoder mới học trên dữ liệu tổng hợp, nên **chưa thể kỳ vọng chất lượng tracking thật**. Cần huấn luyện trên DanceTrack (subset của TV4) với embedding thật của detector, việc này cần GPU.
2. **Dùng ID Decoder chính thức:** dùng `inspect_checkpoint` đối chiếu khoá với `models/motip/` của repo, viết bộ chuyển khoá hoặc một lớp bọc gọi thẳng module ID Decoder của repo, nạp `strict=True`, rồi so đầu ra với code chính thức trên cùng đầu vào. **Chưa có code cho việc này.**

## 5. Lỗi thường gặp

| Thông báo / triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Không phải checkpoint của TV3 (thiếu kind='tv3_id_decoder')` | Đưa checkpoint loại B/C hoặc file `{"model": state_dict}` trần vào `load_id_decoder` | Loại B/C nạp bằng `load_official_motip`, không dùng `load_id_decoder` |
| `Not a checkout of MCG-NJU/MOTIP` | `--repo` / `motip_repo` sai | Clone vào `external/MOTIP` |
| `Missing config or checkpoint` | Sai đường dẫn | Kiểm tra tên file `.pth` thật đã tải |
| `RuntimeError: Error(s) in loading state_dict` | Config và checkpoint không cùng bản | Tải đúng cặp từ MODEL_ZOO. **Không** đổi sang `strict=False` |
| `Unknown state_dict keys` | Checkpoint không phải MOTIP đầy đủ cũng không phải DETR pretrain | Dùng `inspect_checkpoint` xem khoá |
| `ValueError: decoder.num_labels=... khác config.num_labels=...` | K của decoder khác `InferenceConfig` | Đặt `InferenceConfig(num_labels=decoder.num_labels)` |
| `feature phải có 256 chiều` | C của decoder khác C của detector | Huấn luyện lại với `--C 256` |
| `ModuleNotFoundError: utils.nested_tensor` | Gọi `make_nested_frame` khi repo MOTIP chưa vào `sys.path` | Gọi `load_official_motip` trước (hàm này thêm repo vào `sys.path`) |
| Nạp trên CPU quá chậm hoặc không biên dịch được | Deformable Attention cần CUDA | Dùng WSL2 / Linux / Colab có GPU |

## 6. Quy ước lưu file

- Đặt checkpoint trong `checkpoints/`, ví dụ `checkpoints/detector_<tên>.pth`, `checkpoints/id_decoder_synth.pth`.
- **Không đẩy `.pth` lên Git.** `.gitignore` hiện tại ở thư mục gốc **chưa** có các dòng sau, cần thêm (TV5):

```
checkpoints/
outputs/
external/
*.pth
```

- Chia sẻ file nặng qua thư mục Drive chung. TV1 ghi tên file checkpoint đã tải và đường dẫn Drive vào `interface.md` mục 8.
- Chỉ nạp checkpoint từ nguồn tin cậy.

## 7. Trạng thái kiểm chứng

| Hạng mục | Tình trạng |
|---|---|
| `load_id_decoder`, `save_id_decoder` | Có test `test_checkpoint_roundtrip` trong `test_id_decoder.py`, nhưng theo README1 **chưa chạy pytest** (môi trường soạn không có torch). Cần chạy trên máy nhóm |
| `load_official_motip` | Chưa chạy với checkpoint thật (cần Linux/CUDA). Checklist nghiệm thu ở `tv1_detr/tensor_shapes.md` mục 9 vẫn mở |
| Đoạn `Pipeline` ở mục 3 | Phác thảo, **chưa chạy**. Chữ ký tham số là đề nghị, cần TV5 chốt |
| Chất lượng tracking với decoder tổng hợp | **Chưa đo.** Không kết luận gì về độ chính xác trên dữ liệu thật |

## 8. Việc còn lại

- [ ] TV3: chạy `pytest` và `id_train.py`, lưu `checkpoints/id_decoder_*.pth`, dán log vào báo cáo.
- [ ] TV1: ghi tên checkpoint + config đã tải và thư mục Drive vào `interface.md` mục 8; chạy 1 đến 3 frame thật.
- [ ] TV5: cập nhật `interface.md` mục 5 (hai tham số checkpoint), thêm dòng vào `.gitignore`, viết `Pipeline` thật.
- [ ] Cả nhóm: chốt hướng ở mục 4 (decoder huấn luyện lại hay dùng decoder chính thức).
