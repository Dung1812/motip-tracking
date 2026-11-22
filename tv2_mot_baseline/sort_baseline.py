"""
sort_baseline.py
================
Ví dụ tracking-by-detection TRUYỀN THỐNG: SORT (Bewley et al., 2016) =
Kalman Filter (hằng vận tốc) + ma trận IoU + thuật toán Hungarian.
Cờ ``use_byte=True`` thêm bước ghép lần 2 với detection điểm thấp (ý tưởng của ByteTrack).

Dùng để so sánh với MOTIP (ghép bằng ID prediction) trong demo/test.
Chỉ dùng numpy + scipy (scipy.optimize.linear_sum_assignment = Hungarian).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------- chuyển đổi box <-> trạng thái Kalman ---------------- #
def xyxy_to_z(b) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)]).reshape(4, 1)


def x_to_xyxy(x) -> np.ndarray:
    s, r = max(float(x[2, 0]), 1e-6), max(float(x[3, 0]), 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([x[0, 0] - w / 2, x[1, 0] - h / 2, x[0, 0] + w / 2, x[1, 0] + h / 2])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU (len(a), len(b)) giữa hai tập box xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class KalmanBoxTrack:
    """Kalman hằng vận tốc, trạng thái [cx, cy, s, r, vcx, vcy, vs]; ma trận như SORT gốc."""
    _next_id = 1

    def __init__(self, box):
        self.x = np.zeros((7, 1)); self.x[:4] = xyxy_to_z(box)
        self.F = np.eye(7); self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.R = np.diag([1., 1., 10., 10.])
        self.P = np.diag([10., 10., 10., 10., 1e4, 1e4, 1e4])
        self.Q = np.diag([1., 1., 1., 1., 0.01, 0.01, 1e-4])
        self.id = KalmanBoxTrack._next_id; KalmanBoxTrack._next_id += 1
        self.hits, self.since_update, self.age = 1, 0, 0

    def predict(self) -> np.ndarray:
        if self.x[6, 0] + self.x[2, 0] <= 0:      # tránh diện tích âm
            self.x[6, 0] = 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.since_update += 1
        return x_to_xyxy(self.x)

    def update(self, box) -> None:
        y = xyxy_to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.hits += 1
        self.since_update = 0


class SortTracker:
    def __init__(self, max_age: int = 10, min_hits: int = 2, iou_thr: float = 0.3,
                 det_thr: float = 0.3, use_byte: bool = False, high_thr: float = 0.6,
                 byte_iou_thr: float = 0.5):
        self.max_age, self.min_hits, self.iou_thr = max_age, min_hits, iou_thr
        self.det_thr, self.use_byte, self.high_thr, self.byte_iou_thr = det_thr, use_byte, high_thr, byte_iou_thr
        self.tracks: List[KalmanBoxTrack] = []
        self.frame = 0
        KalmanBoxTrack._next_id = 1   # ID bắt đầu từ 1 cho mỗi video

    @staticmethod
    def _hungarian(tracks_boxes, det_boxes, thr) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        iou = iou_matrix(tracks_boxes, det_boxes)
        matches = []
        if iou.size:
            r, c = linear_sum_assignment(-iou)
            matches = [(i, j) for i, j in zip(r, c) if iou[i, j] >= thr]
        mt = {i for i, _ in matches}; md = {j for _, j in matches}
        return (matches,
                [i for i in range(len(tracks_boxes)) if i not in mt],
                [j for j in range(len(det_boxes)) if j not in md])

    def step(self, boxes, scores) -> List[Tuple[int, np.ndarray, int]]:
        """Trả về list (track_id, box, det_index gốc) cho track đang được quan sát."""
        self.frame += 1
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        preds = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)

        keep = np.flatnonzero(scores >= self.det_thr)
        if self.use_byte:
            hi = [i for i in keep if scores[i] >= self.high_thr]
            lo = [i for i in keep if scores[i] < self.high_thr]
        else:
            hi, lo = list(keep), []

        matched = {}   # track idx -> det idx gốc
        m, un_t, un_d = self._hungarian(preds, boxes[hi], self.iou_thr)
        for ti, dj in m:
            matched[ti] = hi[dj]
        new_det = [hi[j] for j in un_d]

        if self.use_byte and lo and un_t:                     # vòng 2: detection điểm thấp
            m2, _, _ = self._hungarian(preds[un_t], boxes[lo], self.byte_iou_thr)
            for a, dj in m2:
                matched[un_t[a]] = lo[dj]

        out = []
        for ti, di in matched.items():
            self.tracks[ti].update(boxes[di])
        for di in new_det:
            if not self.use_byte or scores[di] >= self.high_thr:
                self.tracks.append(KalmanBoxTrack(boxes[di]))
                matched[len(self.tracks) - 1] = di
        for ti, di in matched.items():
            t = self.tracks[ti]
            if t.hits >= self.min_hits or self.frame <= self.min_hits:
                out.append((t.id, boxes[di].copy(), int(di)))
        self.tracks = [t for t in self.tracks if t.since_update <= self.max_age]
        return sorted(out, key=lambda o: o[2])
