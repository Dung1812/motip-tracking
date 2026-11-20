"""
tracking_inference.py
=====================
Suy luận tracking theo mục 3.4 của MOTIP ("Multiple Object Tracking as ID Prediction",
Gao et al., arXiv 2403.16848). Mỗi frame:

  1. Lọc detection theo λ_det.
  2. Dựng tracklet lịch sử (feature + ID label của T frame gần nhất) từ MemoryBank.
  3. ID Decoder dự đoán xác suất ID cho từng detection hiện tại.
  4. ID có xác suất cao nhất được nhận nếu p >= λ_id.
  5. Trùng ID: detection điểm cao hơn giữ ID, detection còn lại bị coi là newborn.
  6. Detection chưa có ID và score >= λ_new -> newborn, nhận nhãn mới từ pool.
  7. Track không được khớp -> LOST; mất quá max_lost frame -> ENDED, nhãn được thu hồi.

KHÔNG dùng Hungarian / IoU ở bước ghép (giống bài báo). Hungarian + Kalman chỉ nằm
trong sort_baseline.py để so sánh với phương pháp truyền thống.

Hai điểm KHÔNG được bài báo nêu rõ, đây là lựa chọn cài đặt của mình (đã đánh dấu "[LỰA CHỌN]"):
  [LỰA CHỌN 1] Chỉ xét các nhãn đang có quỹ đạo sống khi argmax (nhãn chưa cấp bị che về 0).
  [LỰA CHỌN 2] Lost-track: bài báo không có cơ chế riêng; ở đây track không được khớp
               vẫn giữ nhãn + lịch sử tối đa max_lost frame (<= T) để khớp lại sau che khuất.

Module chỉ phụ thuộc numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np

from track_manager import Event, Track, TrackManager, TrackState
from track_memory import DecoderBatch


# --------------------------------------------------------------------------- #
# Cấu hình (giá trị mặc định = cấu hình kết quả chính của bài báo, Sec 4.2/App. B)
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    num_labels: int = 50        # K  (50 mặc định; 200 cho MOT17)
    history_len: int = 29       # T  (DanceTrack 29, SportsMOT 59, BFT 19)
    lambda_det: float = 0.3     # lọc detection
    lambda_new: float = 0.6     # ngưỡng sinh newborn
    lambda_id: float = 0.2      # ngưỡng chấp nhận ID dự đoán
    max_lost: Optional[int] = None   # None -> = history_len
    confirm_hits: int = 2
    output_id_mode: str = "uid"      # "uid" | "label"


@dataclass
class TrackOutput:
    """Một đối tượng được xuất ra ở frame hiện tại."""
    uid: int
    label: int
    output_id: int
    box: np.ndarray
    score: float
    state: str
    det_index: int      # chỉ số detection GỐC (trước khi lọc λ_det) -> dùng để chấm điểm
    is_new: bool


# --------------------------------------------------------------------------- #
# Giao diện ID Decoder (thành viên 3 cài đặt)
# --------------------------------------------------------------------------- #
class IDDecoder(Protocol):
    """Hợp đồng giữa bộ quản lý quỹ đạo và ID Decoder.

    __call__(batch, det_feats) -> probs
        batch      : DecoderBatch  (M quỹ đạo, mỗi quỹ đạo tối đa T token)
        det_feats  : (N, C) float32 feature của N detection hiện tại
        probs      : (N, K) float32, xác suất mỗi detection thuộc từng ID label 0..K-1.
                     Phần xác suất còn lại (1 - sum) coi là "newborn".
    """
    num_labels: int

    def __call__(self, batch: DecoderBatch, det_feats: np.ndarray) -> np.ndarray: ...


class AppearanceMockDecoder:
    """Decoder GIẢ LẬP dùng cosine-similarity theo feature, để chạy/kiểm thử khi chưa có
    decoder thật. Không học gì và không đọc ID token: prototype của mỗi quỹ đạo = trung bình
    feature có trọng số giảm theo dt; softmax qua [các quỹ đạo, newborn].
    Chỉ nhằm mô phỏng *hình dạng* đầu ra (N, K) của decoder thật."""

    def __init__(self, num_labels: int, temperature: float = 0.1,
                 newborn_sim: float = 0.5, decay: float = 10.0):
        self.num_labels = num_labels
        self.temperature, self.newborn_sim, self.decay = temperature, newborn_sim, decay

    def __call__(self, batch: DecoderBatch, det_feats: np.ndarray) -> np.ndarray:
        w = np.exp(-batch.dt / self.decay) * batch.mask                       # (M,T)
        proto = (batch.feats * w[..., None]).sum(1)                           # (M,C)
        proto /= np.linalg.norm(proto, axis=1, keepdims=True) + 1e-9
        d = det_feats / (np.linalg.norm(det_feats, axis=1, keepdims=True) + 1e-9)
        sim = d @ proto.T                                                     # (N,M)
        logits = np.concatenate(
            [sim / self.temperature,
             np.full((len(d), 1), self.newborn_sim / self.temperature)], axis=1)
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        probs = np.zeros((len(d), self.num_labels), dtype=np.float32)
        probs[:, batch.track_labels] = p[:, :batch.num_tracks]
        return probs


class TorchIDDecoderAdapter:
    """Khung nối với ID Decoder PyTorch của thành viên 3.  *** CHƯA KIỂM THỬ ***
    (môi trường làm việc không có torch). Chữ ký forward của model dưới đây là GIẢ ĐỊNH --
    sửa lại cho khớp model thật. Việc cần giữ: đầu ra (N, K) xác suất, đã softmax."""

    def __init__(self, model, num_labels: int, device: str = "cpu"):
        self.model, self.num_labels, self.device = model, num_labels, device

    def __call__(self, batch: DecoderBatch, det_feats: np.ndarray) -> np.ndarray:
        import torch  # import muộn để module vẫn chạy khi không có torch
        t = lambda a, dtype=None: torch.as_tensor(a, dtype=dtype, device=self.device)
        with torch.no_grad():
            logits = self.model(
                hist_feats=t(batch.feats), hist_labels=t(batch.labels), hist_dt=t(batch.dt),
                hist_mask=t(batch.mask), det_feats=t(det_feats, torch.float32))
            return torch.softmax(logits, dim=-1).cpu().numpy()


# --------------------------------------------------------------------------- #
# Bộ suy luận
# --------------------------------------------------------------------------- #
class MOTIPTracker:
    def __init__(self, decoder: IDDecoder, config: Optional[InferenceConfig] = None):
        self.cfg = config or InferenceConfig()
        if decoder.num_labels != self.cfg.num_labels:
            raise ValueError(f"decoder.num_labels={decoder.num_labels} "
                             f"khác config.num_labels={self.cfg.num_labels}")
        self.decoder = decoder
        self.mgr = TrackManager(self.cfg.num_labels, self.cfg.history_len, self.cfg.max_lost,
                                self.cfg.confirm_hits, self.cfg.output_id_mode)
        self._last_frame = -1

    @property
    def events(self) -> List[Event]:
        return self.mgr.events

    def step(self, frame: int, boxes, scores, feats) -> List[TrackOutput]:
        """Xử lý một frame. boxes (N,4) xyxy, scores (N,), feats (N,C)."""
        if frame <= self._last_frame:
            raise ValueError("frame phải tăng dần")
        self._last_frame = frame
        cfg, mgr = self.cfg, self.mgr

        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        feats = np.asarray(feats, np.float32)
        # frame không có detection nào (hay gặp khi bị che hết): reshape (0, -1) sẽ lỗi
        feats = feats.reshape(len(scores), -1) if len(scores) else np.zeros((0, 1), np.float32)

        # 1) lọc λ_det
        orig = np.flatnonzero(scores >= cfg.lambda_det)
        b, s, f = boxes[orig], scores[orig], feats[orig]
        n = len(orig)

        # 2-4) dự đoán ID
        pred = np.full(n, -1, dtype=int)
        batch = mgr.memory.build_batch(frame, mgr.live_tracks()) if n else None
        if batch is not None:
            probs = np.asarray(self.decoder(batch, f), dtype=np.float64)
            if probs.shape != (n, cfg.num_labels):
                raise ValueError(f"decoder trả về {probs.shape}, cần {(n, cfg.num_labels)}")
            allowed = np.zeros(cfg.num_labels, bool)
            allowed[batch.track_labels] = True                    # [LỰA CHỌN 1]
            probs = np.where(allowed[None, :], probs, 0.0)
            best = probs.argmax(1)
            p_best = probs[np.arange(n), best]
            pred = np.where(p_best >= cfg.lambda_id, best, -1)

        # 5) trùng ID
        winners, _losers = mgr.resolve_duplicate_labels(frame, pred, s)

        outputs: List[TrackOutput] = []
        matched_dets, matched_uids = set(), set()
        by_label = mgr.by_label()
        for lab, i in winners.items():
            tr = by_label[lab]
            mgr.update_track(tr, frame, b[i], f[i], s[i])
            mgr.record_output(frame, tr, b[i], s[i])
            outputs.append(self._out(tr, b[i], s[i], orig[i], False))
            matched_dets.add(i)
            matched_uids.add(tr.uid)

        # 7) mất dấu / kết thúc (trước khi sinh newborn để nhãn vừa trả về dùng lại được)
        for tr in mgr.live_tracks():
            if tr.uid not in matched_uids:
                mgr.mark_missed(tr, frame)

        # 6) newborn -- detection điểm cao xử lý trước (quan trọng khi pool gần cạn)
        order = sorted((i for i in range(n) if i not in matched_dets and s[i] >= cfg.lambda_new),
                       key=lambda i: -s[i])
        for i in order:
            tr = mgr.create_track(frame, b[i], f[i], s[i])
            if tr is None:
                continue
            mgr.record_output(frame, tr, b[i], s[i])
            outputs.append(self._out(tr, b[i], s[i], orig[i], True))

        outputs.sort(key=lambda o: o.det_index)
        return outputs

    def finish(self, frame: int) -> None:
        """Kết thúc video: đóng mọi quỹ đạo còn sống."""
        self.mgr.end_all(frame, "video_end")

    def _out(self, tr: Track, box, score, det_index, is_new) -> TrackOutput:
        return TrackOutput(tr.uid, tr.label, self.mgr.output_id(tr), np.asarray(box, float),
                           float(score), tr.state.value, int(det_index), is_new)
