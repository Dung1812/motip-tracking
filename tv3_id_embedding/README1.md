# Phần G4-TV3: ID Dictionary, Tracklet và ID Decoder

Người phụ trách: Đào Trung Dũng. Cài đặt theo Sec 3.1–3.4 của MOTIP (Gao et al., CVPR 2025) và theo hợp đồng `interface.md` v0.1.

## 1. Trả lời yêu cầu của TV5 (interface.md mục 6)

> TV3: tên lớp và cách nạp checkpoint

| Câu hỏi | Trả lời |
|---|---|
| Tên lớp để TV2/TV5 dùng | `IDDecoderForTracker` (file `id_decoder.py`) |
| Lớp huấn luyện | `MOTIPIDDecoder` (nn.Module), đầu ra logit K+1 |
| Cách nạp checkpoint | `load_id_decoder("checkpoints/xxx.pth", device="cpu")` trả về `IDDecoderForTracker` |
| Hợp đồng với TV2 | `decoder.num_labels` (= K) và `decoder(batch, det_feats) -> np.ndarray (N, K)`, float32, đã softmax. Phần `1 - tổng` là xác suất newborn |
| Số chiều | K = 50, C = 256 (khớp `DETR_HIDDEN_DIM=256` của TV1 và bài báo) |
| Nhãn | 0..K-1 (như TV2). Token newborn ở chỉ số K |

Cách ghép vào pipeline (thay `AppearanceMockDecoder`):

```python
from id_decoder import load_id_decoder
from tracking_inference import InferenceConfig
from tv2_interface import TrackerTV2

decoder = load_id_decoder("checkpoints/id_decoder.pth", device="cuda")
tracker = TrackerTV2(decoder=decoder, config=InferenceConfig(num_labels=decoder.num_labels))
```

**Không dùng `TorchIDDecoderAdapter` của TV2** cho lớp này. Adapter đó gọi `model(hist_feats=..., ...)` rồi softmax ra `(N, K+1)`, trong khi TV2 kiểm tra shape `(N, K)` và sẽ báo `ValueError`. `IDDecoderForTracker` đã tự bỏ cột newborn.

## 2. File trong thư mục

| File | Vai trò |
|---|---|
| `id_decoder.py` | `IDDictionary`, `MOTIPIDDecoder`, `id_loss`, `IDDecoderForTracker`, `save_id_decoder` / `load_id_decoder` / `inspect_checkpoint` |
| `synthetic_data.py` | Dữ liệu tổng hợp in-context ID prediction + trajectory augmentation (chỉ cần numpy) |
| `id_train.py` | Huấn luyện thu nhỏ trên dữ liệu tổng hợp |
| `test_id_decoder.py` | 10 test (hợp đồng với TV2, mask, lịch sử rỗng, gradient, khả năng học, checkpoint) |
| `id_embedding_demo.py` | Bản demo cũ giai đoạn 1 (C=128). `IDDictionary` mới trong `id_decoder.py` thay thế và vẫn giữ các hàm `get_embedding`, `get_newborn_embedding`, `get_all_embeddings` |

## 3. Kiến trúc đã cài đặt

```
DecoderBatch (TV2): feats (M,T,C), labels (M,T), dt (M,T), mask (M,T)
  token lịch sử   tau = concat(f, i_k)        2C = 512 chiều  -> Linear -> d_model, cộng embedding(dt)
  token detection tau = concat(f, i_spec)     i_spec = từ thứ K+1 của ID dictionary (newborn)
  ID Decoder: nn.TransformerDecoder (self-attn giữa các detection hiện tại, cross-attn tới mọi token lịch sử)
  head Linear(d_model, K+1) -> softmax -> bỏ cột K -> (N, K)
```

Ánh xạ với bài báo:

| Mục bài báo | Cài đặt |
|---|---|
| Sec 3.2 ID Dictionary, K+1 từ học được | `IDDictionary` (`nn.Embedding(K+1, C)`) |
| Eq. (2) tracklet = concat(f, i_k) | `torch.cat([feat, id_embedding], -1)` |
| ID Decoder = transformer decoder, Key/Value là tracklet lịch sử | `nn.TransformerDecoder`, memory = mọi token lịch sử |
| Self-attention giữa các detection (Tab. 4) | Có (self-attn của decoder layer) |
| Sec 3.3, L_id = cross-entropy K+1 lớp | `id_loss` |
| Sec 3.3, trajectory augmentation λocc = λsw = 0.5 | `random_occlusion`, `random_switch` trong `synthetic_data.py` |
| Sec 3.4, bỏ qua token padding | `memory_key_padding_mask` |

## 4. Những chỗ KHÔNG bám sát bài báo (lựa chọn của nhóm)

1. **Mã hoá vị trí tương đối.** Bài báo dùng relative position encoding nhưng không mô tả chi tiết. Ở đây dùng embedding học được theo `dt` (cắt ở `max_dt=128`), cộng vào token lịch sử.
2. **Số layer, số head, d_model, FFN.** Bài báo không nêu trong phần chính. Mặc định 6 layer, 8 head, d_model = 256, FFN = 1024 (cần đối chiếu với `configs/` của repo chính thức).
3. **Chiếu 2C -> d_model** bằng một `Linear` dùng chung. Bài báo không nói rõ.
4. **Diễn giải augmentation.** Random occlusion: mỗi quỹ đạo, với xác suất λocc, bị bỏ token với tỉ lệ r~U(0,1), luôn giữ ít nhất 1 token. Random switch: với xác suất λsw, đổi nhãn của hai quỹ đạo tại 1–2 frame lịch sử, nhãn mục tiêu không đổi.
5. **Checkpoint chính thức MOTIP KHÔNG nạp thẳng được** vào `MOTIPIDDecoder`, vì tên khoá và cấu trúc layer có thể khác. `inspect_checkpoint(path)` in khoá và shape của phần không phải detector để đối chiếu. Viết bộ chuyển khoá cho checkpoint chính thức là việc chưa làm (xem mục 7).

## 5. Cách chạy

```bash
cd tv3_id_embedding
pip install torch numpy pytest
python -m pytest -q test_id_decoder.py

# huấn luyện thu nhỏ để kiểm tra decoder học được (CPU chạy được cấu hình nhỏ)
python id_train.py --steps 1500 --C 32 --T 8 --layers 2 --d-model 128 --heads 4 --save checkpoints/id_decoder_small.pth

# cấu hình đúng kích thước bài báo (C=256, T=29, 6 layer) nên chạy trên GPU
python id_train.py --steps 5000 --save checkpoints/id_decoder_synth.pth
```

Log mong đợi: dòng `[trước huấn luyện]` có acc gần ngẫu nhiên (~1/(K+1)), sau đó acc tăng dần qua các mốc `step`. Chụp log này làm minh chứng nghiệm thu.

## 6. Trạng thái kiểm chứng (đọc kỹ)

| Hạng mục | Tình trạng |
|---|---|
| `synthetic_data.py` | **Đã chạy thật** (200 mẫu): shape, `dt` trong [1, T], nhãn mục tiêu không trùng, `collate` đúng |
| `id_decoder.py`, `id_train.py`, `test_id_decoder.py` | Mới **kiểm tra cú pháp (`py_compile`)**. Môi trường soạn code không có torch và không có mạng, nên **chưa chạy pytest và chưa huấn luyện lần nào**. Cần chạy lệnh ở mục 5 trên máy nhóm và sửa nếu có lỗi |
| Test `test_model_learns_in_context_id_prediction` | Ngưỡng acc > 0.8 là **ước tính chưa kiểm chứng**. Nếu không đạt, tăng số bước hoặc giảm độ khó (`similarity`, `feat_noise`) rồi ghi lại số thật |
| Tích hợp với TV2 | Test `test_plugs_into_tv2_tracker_without_shape_errors` kiểm tra hợp đồng (shape, nhãn). Decoder chưa huấn luyện nên **không** kiểm tra chất lượng ID |
| Chất lượng trên dữ liệu thật | **Chưa có.** Dữ liệu tổng hợp chỉ chứng minh logic in-context ID prediction học được |

## 7. Việc còn lại / cần phối hợp

- [ ] Chạy pytest và huấn luyện thu nhỏ, dán log vào báo cáo.
- [ ] Đối chiếu số layer/head/cách mã hoá `dt` với `models/motip/` của repo MCG-NJU/MOTIP (cùng môi trường WSL2/Linux của TV1) và ghi lại các điểm khác.
- [ ] Nếu cần dùng trọng số chính thức: dùng `inspect_checkpoint`, viết bộ chuyển khoá, nạp `strict=True`, rồi so đầu ra với code chính thức trên cùng một đầu vào.
- [ ] Phối hợp TV1: xác nhận `C = 256` và `embeddings` (query feature `outputs`) đúng là đầu vào `det_feats` (không phải ID embedding).
- [ ] Phối hợp TV2: dùng `IDDecoderForTracker` trực tiếp, bỏ `TorchIDDecoderAdapter`. `InferenceConfig.num_labels` phải bằng `decoder.num_labels`.
- [ ] Phối hợp TV5: ghi `C=256`, `K=50` và tên lớp vào `interface.md` rồi tick mục 6. Đưa `checkpoints/` vào `.gitignore` (không đẩy file `.pth` lên repo).
- [ ] Huấn luyện thật trên DanceTrack (subset của TV4) chỉ khả thi khi có GPU. Nếu không có, nêu rõ trong báo cáo đây là giới hạn.
