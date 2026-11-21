# MOTIP giai đoạn 2 – Bảng kích thước tensor và quy tắc bàn giao

> Phạm vi: **Detector / checkpoint / bbox / score / decoder object embedding**.  
> Không đánh đồng object embedding với learnable ID embedding hoặc ID do tracker dự đoán.

## 1. Kiến trúc được tham chiếu

Repository chính thức: https://github.com/MCG-NJU/MOTIP  
Cấu hình ví dụ: `configs/r50_deformable_detr_motip_dancetrack.yaml` (main, truy cập 2026-11-20)  
Thông số trong cấu hình đó: `BACKBONE=resnet50`, `DETR_FRAMEWORK=deformable_detr`, `DETR_NUM_QUERIES=300`, `DETR_HIDDEN_DIM=256`, `NUM_CLASSES=1`, `FEATURE_DIM=256`, `ID_DIM=256`.

**Chú ý:** Các con số là cho cấu hình ví dụ, **không phải** mọi checkpoint DETR/MOTIP. Luôn in shape thực tế sau `forward`.

## 2. Cách đọc ký hiệu

- `B`: batch size, số ảnh một lượt chạy (demo: B=1)
- `T`: số frame theo thời gian (để theo dõi); không có trong đầu ra detector một frame
- `Q`: số object query (cấu hình ví dụ Q=300)
- `C`: số lớp detector (DanceTrack C=1 / person)
- `D`: số chiều decoder object feature (cấu hình ví dụ D=256)
- `N`: số detection sau lọc, thay đổi theo frame; **không** bằng Q
- `H`, `W`: chiều cao/rộng frame gốc; `H'`, `W'`: kích thước tensor sau tiền xử lý

## 3. Tensor đầu vào/đầu ra

| Tên | Shape ví dụ | Ý nghĩa |
|---|---|---|
| `frame_bgr` | `[H,W,3]` | Frame OpenCV, BGR `uint8` |
| `image_tensor` | `[1,3,H',W']` | Ảnh RGB float, chuẩn hóa ImageNet, CHW |
| `padding_mask` | `[1,H',W']` | Bool, False = pixel hợp lệ |
| `pred_logits` | `[1,300,1]` | **Logit**, chưa phải xác suất |
| `pred_boxes` | `[1,300,4]` | BBox chuẩn hóa dạng `[cx,cy,w,h]` |
| `outputs` | `[1,300,256]` | Feature của **từng object query** từ decoder cuối |
| `scores` | `[N]` | `sigmoid(pred_logits)` rồi chọn class/query và lọc threshold |
| `labels` | `[N]` | Nhãn lớp của các detection đã chọn |
| `boxes_xyxy` | `[N,4]` | `[x1,y1,x2,y2]` pixel theo frame gốc |
| `embeddings` | `[N,256]` | Feature detector của đúng các query được chọn |
| `query_indices` | `[N]` | Chỉ số query gốc (cần để kiểm tra căn hàng) |

**Quan trọng:** MOTIP dùng đầu ra `outputs` từ decoder cuối (`hs[-1]`) làm đặc trưng đối tượng. Vanilla DETR/HuggingFace DETR có thể dùng API/keys khác và **không nhất thiết** xuất key `outputs` như trên.

Nguồn mã chính thức:
- `models/deformable_detr/deformable_detr.py`: `out = {'pred_logits', 'pred_boxes'}` và `out['outputs'] = hs[-1]`
- `train.py`: truy xuất `detr_outputs['outputs']` cùng chỉ số query và box.

## 4. Tại sao phải lấy cùng `query_indices`?

Giả sử có Q=300 query, sau lọc score giữ query `2`, `7`, `11`.

- `boxes_xyxy = boxes[[2,7,11]]` → shape `[3,4]`
- `scores = probabilities[[2,7,11]]` → shape `[3]`
- `embeddings = outputs[[2,7,11]]` → shape `[3,256]`

Nếu sắp xếp bbox theo score nhưng quên sắp xếp embedding tương ứng thì các vector bị gán nhầm đối tượng; toàn bộ đầu vào của ID Decoder trở nên sai.

## 5. BBox và hệ tọa độ

Đầu vào detector: `(cx,cy,bw,bh)` đã được chuẩn hóa theo chiều rộng và chiều cao.

```
x1 = (cx - bw/2)*W
x2 = (cx + bw/2)*W
y1 = (cy - bh/2)*H
y2 = (cy + bh/2)*H
```

Adapter giữ bbox ở **pixel ảnh gốc**, cắt về `[0,W]` và `[0,H]`, loại hộp suy biến. Các thành viên về sau phải biết chính xác hệ tọa độ này để tránh nhân kích thước lần nữa.

## 6. Score của MOTIP khác DETR gốc

MOTIP dùng Deformable DETR và sigmoid theo lớp:

```
scores = pred_logits.sigmoid()[0, :, 0]  # DanceTrack C=1
keep = scores >= DET_THRESH
```

Không lấy `pred_logits` thô làm score; cũng không dùng `softmax` với no-object cuối như cách hậu xử lý DETR gốc, trừ khi xác minh checkpoint có cấu hình đó.

Trong ví dụ DanceTrack tại thời điểm tra cứu: `DET_THRESH=0.3`, `NEWBORN_THRESH=0.6`, `ID_THRESH=0.2`. Hai ngưỡng sau liên quan đến quản lý trajectory/ID, không dùng để thay thế `DET_THRESH` cho adapter.

## 7. Các checkpoint cần phân biệt

1. **COCO detector pretrain** – trọng số detector ban đầu, có thể khác số lớp; cần hàm loader phù hợp.
2. **Dataset-specific DETR pretrain** – detector đã tiền huấn luyện trên DanceTrack/SportsMOT; thích hợp kiểm tra detector.
3. **Full MOTIP checkpoint** – bao gồm detector + trajectory modeling + ID Decoder. Tốt nhất khi cần tích hợp toàn tuyến.

Nạp phải cùng **repo version, config và loại checkpoint**. Đừng thay `strict=False` để che lỗi mismatch.

*Lưu ý quan trọng:* `DETR_NUM_CHECKPOINT_FRAMES` là **gradient activation checkpointing** để tiết kiệm bộ nhớ khi huấn luyện, **khác** file trọng số `.pth`.

## 8. Quy trình chạy và lưu sản phẩm

Đặt `detector_adapter.py` ở thư mục dự án hoặc `tv1_detr/` theo phân công nhóm.

**Bước A – kiểm thử độc lập (không cần tải model):**

```powershell
python detector_adapter.py --mode self-test
```

Sinh `outputs/stage2_detector/synthetic_only.json/.npz` (data **giả lập**, chỉ kiểm tra index và shape).

**Bước B – chuẩn bị MOTIP chính thức:**

```powershell
git clone https://github.com/MCG-NJU/MOTIP.git external/MOTIP
```

Cài và biên dịch đúng theo `external/MOTIP/docs/INSTALL.md`. Repo tác giả dùng Python 3.12, PyTorch 2.4.0, torchvision 0.19.0 và extension Deformable Attention. Môi trường Windows/Python 3.13 hiện tại có thể không tương thích; thường nên dùng WSL2 Ubuntu hoặc Linux + CUDA tương thích (kiểm tra driver trước). Mở VS Code vào WSL2 nếu cần.

Chọn **config + checkpoint khớp nhau** trong https://github.com/MCG-NJU/MOTIP/blob/main/docs/MODEL_ZOO.md

**Bước C – chạy với checkpoint thật:**

```powershell
python detector_adapter.py --mode motip `
    --repo external/MOTIP `
    --config external/MOTIP/configs/r50_deformable_detr_motip_dancetrack.yaml `
    --checkpoint checkpoints/MOTIP_MATCHING_DANCETRACK.pth `
    --source your_video.mp4 --limit 3 --device cuda
```

Thay `MOTIP_MATCHING_DANCETRACK.pth` bằng tên checkpoint **thật bạn đã tải**. `--source 0` là webcam, có thể dùng image `.jpg`.

Đầu ra mỗi frame:

- `.jpg`: vẽ bbox + class + score + query index (chưa có track ID)
- `.json`: `bbox`, `score`, `label`, `query_index`, raw tensor shapes
- `.npz`: **đầy đủ** mảng `boxes_xyxy [N,4]`, `scores [N]`, `labels [N]`, `embeddings [N,D]`, `query_indices [N]`

**Không chạy `--mode self-test` rồi báo rằng đã kiểm thử checkpoint thật.** Cần ảnh/nhật ký từ chế độ `--mode motip`.

## 9. Checklist nghiệm thu

- [ ] Chọn và ghi rõ tên/nguồn/phiên bản checkpoint, config, GPU/CPU.
- [ ] Checkpoint load đầy đủ, không báo lỗi thiếu hoặc sai kích thước tensor.
- [ ] Chạy 1-3 frame thực bằng detector chính thức.
- [ ] In được các shape `pred_logits`, `pred_boxes`, `outputs`.
- [ ] Kiểm tra score `[0,1]`, bbox hợp lệ trong frame gốc.
- [ ] Kiểm tra `embeddings.shape[0] == boxes.shape[0] == scores.shape[0]`.
- [ ] Lưu `.jpg`, `.json`, `.npz` và screenshot log làm minh chứng.
- [ ] Giải thích rõ object embedding khác ID embedding và track ID.
- [ ] Nộp `detector_adapter.py`, `tensor_shapes.md` và log kết quả chạy thật.

## 10. Chưa phải phạm vi giai đoạn 2

- Duy trì ID qua frame chưa có trong adapter.
- ID Decoder gốc dùng object feature + lịch sử/ID dictionary để dự đoán ID.
- Không kết luận chất lượng tracking từ screenshot detection riêng lẻ.
- Không push checkpoint lớn lên Git; dùng `.gitignore` cho `checkpoints/`, `outputs/`, `external/`.
