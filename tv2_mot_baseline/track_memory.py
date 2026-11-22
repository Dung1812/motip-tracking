"""
track_memory.py
===============
Bộ nhớ lịch sử đối tượng cho tracking kiểu MOTIP (in-context ID prediction).

Có HAI loại "lịch sử", được tách bạch có chủ đích:

1. ``TrackMemory`` / ``MemoryBank``  -- bộ nhớ NGẮN HẠN, cửa sổ T frame gần nhất của
   từng quỹ đạo. Đây chính là "historical tracklets" (key/value) mà ID Decoder
   đọc ở mục 3.2/3.4 của bài báo: mỗi token = (feature, ID label, thời điểm).
   Chỉ giữ T frame gần nhất  ->  kích thước bị chặn, đưa thẳng vào decoder.

2. ``TrajectoryArchive``             -- lịch sử DÀI HẠN, không bao giờ cắt bớt,
   lưu toàn bộ (frame, uid, label, box, state) để xuất kết quả MOT, vẽ demo,
   thống kê vòng đời. Decoder KHÔNG đọc cái này.

Quy ước box: xyxy (x1, y1, x2, y2), đơn vị pixel.
Module chỉ phụ thuộc numpy -> chạy được mà không cần torch. Khi nối với ID Decoder
(PyTorch) của thành viên 3, chỉ cần chuyển ``DecoderBatch`` sang tensor
(xem ``TorchIDDecoderAdapter`` trong tracking_inference.py).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Bộ nhớ ngắn hạn (đầu vào của ID Decoder)
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    """Một lần quỹ đạo được quan sát (được gán detection) tại một frame."""
    frame: int
    box: np.ndarray        # (4,) xyxy
    feature: np.ndarray    # (C,) object embedding từ detector
    score: float


class TrackMemory:
    """Cửa sổ T frame gần nhất của MỘT quỹ đạo.

    Lưu ý: frame bị che khuất / mất detection thì KHÔNG có observation
    (không điền giả). Decoder nhìn thấy "lỗ hổng" qua ``dt`` -- giống việc
    random-occlusion augmentation ở huấn luyện (mục 3.3) bỏ token ra khỏi tracklet.
    """

    def __init__(self, history_len: int):
        if history_len < 1:
            raise ValueError("history_len phải >= 1")
        self.history_len = history_len
        self._obs: Deque[Observation] = deque(maxlen=history_len)
        self.total_obs = 0  # tổng số lần quan sát từ lúc sinh, không bị cắt

    def append(self, obs: Observation) -> None:
        if self._obs and obs.frame <= self._obs[-1].frame:
            raise ValueError(
                f"Observation phải tăng dần theo frame (có {self._obs[-1].frame} -> {obs.frame})")
        self._obs.append(obs)
        self.total_obs += 1

    def window(self, current_frame: int) -> List[Observation]:
        """Các observation nằm trong [current_frame - T, current_frame - 1]."""
        lo = current_frame - self.history_len
        return [o for o in self._obs if lo <= o.frame < current_frame]

    @property
    def last(self) -> Optional[Observation]:
        return self._obs[-1] if self._obs else None

    def clear(self) -> None:
        """Gọi khi quỹ đạo KẾT THÚC: xoá sạch để nhãn được tái sử dụng không
        mang theo token của danh tính cũ vào decoder."""
        self._obs.clear()

    def __len__(self) -> int:
        return len(self._obs)


@dataclass
class DecoderBatch:
    """Đầu vào đã pad cho ID Decoder (M quỹ đạo ứng viên, T token mỗi quỹ đạo).

    feats  : (M, T, C) float32  feature của từng token
    labels : (M, T)    int64    ID label (0..K-1) gắn với token -> tra ID dictionary
    dt     : (M, T)    int64    current_frame - frame_của_token (>=1); 0 ở chỗ pad
    mask   : (M, T)    bool     True = token thật, False = pad
    track_uids   : (M,)  uid nội bộ của quỹ đạo
    track_labels : (M,)  ID label hiện tại của quỹ đạo
    """
    feats: np.ndarray
    labels: np.ndarray
    dt: np.ndarray
    mask: np.ndarray
    track_uids: np.ndarray
    track_labels: np.ndarray

    @property
    def num_tracks(self) -> int:
        return int(self.track_uids.shape[0])


class MemoryBank:
    """uid -> TrackMemory, kèm hàm dựng ``DecoderBatch``."""

    def __init__(self, history_len: int, feat_dim: Optional[int] = None):
        self.history_len = history_len
        self.feat_dim = feat_dim
        self._mem: Dict[int, TrackMemory] = {}

    def create(self, uid: int) -> TrackMemory:
        if uid in self._mem:
            raise KeyError(f"uid {uid} đã có bộ nhớ")
        self._mem[uid] = TrackMemory(self.history_len)
        return self._mem[uid]

    def get(self, uid: int) -> TrackMemory:
        return self._mem[uid]

    def drop(self, uid: int) -> None:
        mem = self._mem.pop(uid, None)
        if mem is not None:
            mem.clear()

    def __contains__(self, uid: int) -> bool:
        return uid in self._mem

    def build_batch(self, current_frame: int, tracks: Iterable) -> Optional[DecoderBatch]:
        """Dựng batch từ các track (đối tượng có .uid và .label).

        Track nào cửa sổ rỗng (không có token trong T frame gần nhất) bị loại:
        decoder không có gì để "đọc" -> không thể được dự đoán ID.
        Trả về None nếu không còn ứng viên nào.
        """
        rows = []
        for t in tracks:
            win = self._mem[t.uid].window(current_frame)
            if win:
                rows.append((t, win))
        if not rows:
            return None

        M, T = len(rows), self.history_len
        C = rows[0][1][0].feature.shape[0]
        feats = np.zeros((M, T, C), dtype=np.float32)
        labels = np.zeros((M, T), dtype=np.int64)
        dt = np.zeros((M, T), dtype=np.int64)
        mask = np.zeros((M, T), dtype=bool)
        uids = np.zeros(M, dtype=np.int64)
        tlabels = np.zeros(M, dtype=np.int64)
        for m, (t, win) in enumerate(rows):
            n = len(win)
            feats[m, :n] = np.stack([o.feature for o in win])
            labels[m, :n] = t.label
            dt[m, :n] = [current_frame - o.frame for o in win]
            mask[m, :n] = True
            uids[m], tlabels[m] = t.uid, t.label
        return DecoderBatch(feats, labels, dt, mask, uids, tlabels)


# --------------------------------------------------------------------------- #
# 2. Lịch sử dài hạn (xuất kết quả, thống kê, demo)
# --------------------------------------------------------------------------- #
@dataclass
class ArchiveRecord:
    frame: int
    uid: int
    label: int
    output_id: int
    box: np.ndarray
    score: float
    state: str


@dataclass
class TrajectoryLife:
    """Tóm tắt vòng đời của một quỹ đạo (một uid)."""
    uid: int
    label: int
    birth_frame: int
    end_frame: Optional[int] = None
    end_reason: Optional[str] = None
    n_observed: int = 0
    n_lost_frames: int = 0
    n_recoveries: int = 0   # số lần LOST -> ACTIVE


class TrajectoryArchive:
    """Lưu toàn bộ lịch sử, không cắt. Mỗi uid một ``TrajectoryLife``."""

    def __init__(self):
        self.records: List[ArchiveRecord] = []
        self.lives: Dict[int, TrajectoryLife] = {}

    # -- ghi -------------------------------------------------------------- #
    def open_life(self, uid: int, label: int, frame: int) -> None:
        self.lives[uid] = TrajectoryLife(uid=uid, label=label, birth_frame=frame)

    def record(self, rec: ArchiveRecord) -> None:
        self.records.append(rec)
        self.lives[rec.uid].n_observed += 1

    def close_life(self, uid: int, frame: int, reason: str) -> None:
        life = self.lives[uid]
        life.end_frame, life.end_reason = frame, reason

    # -- đọc -------------------------------------------------------------- #
    def trajectory(self, uid: int) -> List[ArchiveRecord]:
        return [r for r in self.records if r.uid == uid]

    def frame_ids(self) -> Dict[int, Dict[int, int]]:
        """frame -> {uid: label}"""
        out: Dict[int, Dict[int, int]] = {}
        for r in self.records:
            out.setdefault(r.frame, {})[r.uid] = r.label
        return out

    def to_mot_txt(self) -> str:
        """Định dạng MOTChallenge: frame,id,x,y,w,h,conf,-1,-1,-1 (frame bắt đầu từ 1)."""
        lines = []
        for r in sorted(self.records, key=lambda r: (r.frame, r.output_id)):
            x1, y1, x2, y2 = r.box
            lines.append(f"{r.frame + 1},{r.output_id},{x1:.2f},{y1:.2f},"
                         f"{x2 - x1:.2f},{y2 - y1:.2f},{r.score:.3f},-1,-1,-1")
        return "\n".join(lines)

    def summary(self) -> str:
        rows = ["uid label birth  end  reason            observed lost_fr recov"]
        for life in sorted(self.lives.values(), key=lambda l: l.uid):
            end = "-" if life.end_frame is None else str(life.end_frame)
            rows.append(f"{life.uid:>3} {life.label:>5} {life.birth_frame:>5} {end:>4}  "
                        f"{(life.end_reason or 'alive'):<16} {life.n_observed:>8} "
                        f"{life.n_lost_frames:>7} {life.n_recoveries:>5}")
        return "\n".join(rows)
