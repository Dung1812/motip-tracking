"""
tv2_interface.py  -  Cổng giao tiếp của TV2 (quản lý track) theo hợp đồng interface.md v0.1
=================================================================================
Người phụ trách: G4-TV2 Lương Thị Mai Phương.

Vị trí trong Pipeline (hợp đồng mục 4):  TV1 phát hiện  ->  TV3 dự đoán ID  ->  TV2 cập nhật trạng thái track.

    tracker = TrackerTV2(decoder=<ID Decoder của TV3>)     # tạo một lần
    tracker.reset()                                         # trước mỗi video mới
    tracks = tracker.update(detections, frame_idx)          # mỗi frame; frame_idx bắt đầu từ 1

Đầu vào  `detections` (hợp đồng mục 2), mỗi phần tử:
    {"bbox": [x1, y1, x2, y2], "score": 0.92, "embedding": <torch.Tensor hoặc numpy, shape (C,)>}
Đầu ra   `tracks` (hợp đồng mục 3), mỗi phần tử:
    {"track_id": 3, "bbox": [x1, y1, x2, y2], "score": 0.88, "state": "new" | "tracked" | "lost" | "terminated"}

Ý nghĩa state (ánh xạ từ vòng đời nội bộ trong track_manager.py):
    new         track vừa xuất hiện ở frame này (nội bộ: vừa sinh, NEWBORN)
    tracked     track đang được theo dõi bình thường (nội bộ: ACTIVE, hoặc NEWBORN đang chờ xác nhận)
    lost        tạm mất (bị che khuất), chưa kết thúc; bbox = vị trí quan sát cuối   [trả khi include_lost=True, mặc định True]
    terminated  kết thúc ở frame này, không dùng lại ID; bbox = vị trí quan sát cuối [trả khi include_terminated=True, mặc định True]

Câu hỏi mở của hợp đồng mục 6 ("TV2 có trả cả track lost cho giao diện hay không"):
    TRẢ LỜI: CÓ. Mặc định trả đủ bốn state (include_lost = include_terminated = True), đúng bảng state ở mục 3.
    Giao diện / file MOT PHẢI tự lọc: chỉ vẽ và chỉ ghi các track có state new hoặc tracked
    (xem DRAWN_STATES và hàm only_drawn bên dưới). Muốn quay lại hành vi cũ: include_lost=False, include_terminated=False.

`track_id` = uid nội bộ (số nguyên dương, tăng dần từ 1, KHÔNG BAO GIỜ tái sử dụng trong một video),
đúng yêu cầu "không tái sử dụng ID đã kết thúc" của hợp đồng. Nhãn ID (label) của decoder thì có tái sử dụng,
nhưng không lộ ra ngoài.

Module chỉ phụ thuộc numpy (torch là tuỳ chọn: embedding dạng torch.Tensor được tự chuyển sang numpy).
"""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tracking_inference import (AppearanceMockDecoder, IDDecoder, InferenceConfig,
                                MOTIPTracker)

STATE_NEW, STATE_TRACKED, STATE_LOST, STATE_TERMINATED = "new", "tracked", "lost", "terminated"
DRAWN_STATES = (STATE_NEW, STATE_TRACKED)     # giao diện chỉ vẽ hai state này
Track = Dict[str, object]
Detection = Dict[str, object]


def only_drawn(tracks: Iterable[Track]) -> List[Track]:
    """Giữ lại track new/tracked: dùng cho giao diện (vẽ) và file MOT; bỏ lost/terminated."""
    return [t for t in tracks if t["state"] in DRAWN_STATES]


# --------------------------------------------------------------------------- #
# Chuyển đổi và kiểm tra detection (đầu ra của TV1)
# --------------------------------------------------------------------------- #
def to_numpy(x) -> np.ndarray:
    """torch.Tensor (có thể nằm trên GPU, có gradient) hoặc list/numpy -> numpy float32."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32)


def check_detection(d: Detection) -> Optional[str]:
    """Trả về None nếu detection đúng hợp đồng, ngược lại trả về lý do (chuỗi tiếng Việt)."""
    for key in ("bbox", "score", "embedding"):
        if key not in d:
            return f"thiếu khoá '{key}'"
    try:
        bbox = [float(v) for v in d["bbox"]]
    except (TypeError, ValueError):
        return "bbox không phải dãy số"
    if len(bbox) != 4:
        return f"bbox phải có 4 số [x1,y1,x2,y2], đang có {len(bbox)}"
    if not all(np.isfinite(bbox)):
        return "bbox chứa NaN/inf"
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return f"bbox không hợp lệ (cần x2>x1 và y2>y1): {bbox}"
    try:
        score = float(d["score"])
    except (TypeError, ValueError):
        return "score không phải số"
    if not (0.0 <= score <= 1.0):
        return f"score phải trong [0,1], đang là {score}"
    emb = to_numpy(d["embedding"])
    if emb.ndim != 1 or emb.size == 0:
        return f"embedding phải có shape (C,), đang là {tuple(emb.shape)}"
    if not np.all(np.isfinite(emb)):
        return "embedding chứa NaN/inf"
    return None


def parse_detections(dets: Sequence[Detection], feat_dim: Optional[int] = None,
                     strict: bool = False):
    """list[Detection] -> (boxes (N,4), scores (N,), feats (N,C), số detection bị bỏ).

    strict=True : detection sai hợp đồng -> ValueError (nên bật khi ghép với TV1 để phát hiện lỗi sớm).
    strict=False: bỏ detection sai và cảnh báo (không làm sập pipeline giữa chừng).
    """
    boxes, scores, feats, dropped = [], [], [], 0
    for i, d in enumerate(dets):
        err = check_detection(d)
        if err is None and feat_dim is not None and to_numpy(d["embedding"]).size != feat_dim:
            err = f"embedding có {to_numpy(d['embedding']).size} chiều, cần {feat_dim} (cùng C với các detection trước)"
        if err is not None:
            if strict:
                raise ValueError(f"detection #{i} sai hợp đồng: {err}")
            warnings.warn(f"bỏ detection #{i}: {err}", stacklevel=3)
            dropped += 1
            continue
        boxes.append([float(v) for v in d["bbox"]])
        scores.append(float(d["score"]))
        feats.append(to_numpy(d["embedding"]))
    if not boxes:
        return (np.zeros((0, 4)), np.zeros(0),
                np.zeros((0, feat_dim or 1), np.float32), dropped)
    return np.asarray(boxes, float), np.asarray(scores, float), np.stack(feats), dropped


# --------------------------------------------------------------------------- #
# Bộ theo dõi theo hợp đồng
# --------------------------------------------------------------------------- #
class TrackerTV2:
    """Bọc MOTIPTracker (track_manager + track_memory + tracking_inference) theo giao diện của hợp đồng."""

    def __init__(self, decoder: Optional[IDDecoder] = None, config: Optional[InferenceConfig] = None,
                 include_lost: bool = True, include_terminated: bool = True, strict: bool = False):
        self.cfg = config or InferenceConfig()
        if decoder is None:
            warnings.warn("Chưa có ID Decoder của TV3 -> dùng AppearanceMockDecoder (GIẢ LẬP, chỉ để chạy thử).",
                          stacklevel=2)
            decoder = AppearanceMockDecoder(self.cfg.num_labels)
        self.decoder = decoder
        self.include_lost, self.include_terminated, self.strict = include_lost, include_terminated, strict
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Gọi trước mỗi video mới: xoá toàn bộ track, bộ nhớ; uid bắt đầu lại từ 1."""
        self._trk = MOTIPTracker(self.decoder, self.cfg)
        self._feat_dim: Optional[int] = None
        self._last_frame = 0
        self._ev_ptr = 0
        self._last_seen: Dict[int, Tuple[np.ndarray, float]] = {}   # uid -> (bbox, score) cuối cùng
        self._out_id: Dict[int, int] = {}                            # uid -> track_id đã xuất
        self.n_dropped = 0

    @property
    def events(self):
        return self._trk.events

    @property
    def archive(self):
        return self._trk.mgr.archive

    # ------------------------------------------------------------------ #
    def update(self, detections: Sequence[Detection], frame_idx: int) -> List[Track]:
        """Xử lý một frame, trả list[Track] theo hợp đồng mục 3."""
        if frame_idx < 1:
            raise ValueError(f"frame_idx bắt đầu từ 1 (hợp đồng mục 1), nhận được {frame_idx}")
        if frame_idx <= self._last_frame:
            raise ValueError(f"frame_idx phải tăng dần: {self._last_frame} -> {frame_idx}. "
                             f"Gọi reset() trước video mới.")
        self._last_frame = frame_idx

        boxes, scores, feats, dropped = parse_detections(detections, self._feat_dim, self.strict)
        self.n_dropped += dropped
        if len(scores) and self._feat_dim is None:
            self._feat_dim = feats.shape[1]

        outs = self._trk.step(frame_idx, boxes, scores, feats)

        tracks: List[Track] = []
        for o in outs:
            self._last_seen[o.uid] = (o.box.copy(), o.score)
            self._out_id[o.uid] = o.output_id
            tracks.append(self._track(o.output_id, o.box, o.score,
                                      STATE_NEW if o.is_new else STATE_TRACKED))

        if self.include_lost:
            for t in self._trk.mgr.live_tracks():
                if t.state.value == "lost" and t.uid in self._last_seen:
                    box, sc = self._last_seen[t.uid]
                    tracks.append(self._track(self._out_id[t.uid], box, sc, STATE_LOST))

        ended = self._collect_ended(frame_idx)
        if self.include_terminated:
            tracks.extend(ended)
        return tracks

    def finish(self, frame_idx: Optional[int] = None) -> List[Track]:
        """Hết video: đóng mọi track còn sống. Trả các track terminated (nếu include_terminated)."""
        frame_idx = frame_idx if frame_idx is not None else self._last_frame + 1
        self._trk.finish(frame_idx)
        ended = self._collect_ended(frame_idx)
        return ended if self.include_terminated else []

    # ------------------------------------------------------------------ #
    def _collect_ended(self, frame_idx: int) -> List[Track]:
        ended: List[Track] = []
        evs = self._trk.events
        for e in evs[self._ev_ptr:]:
            if e.kind == "end" and e.uid in self._last_seen:
                box, sc = self._last_seen[e.uid]
                ended.append(self._track(self._out_id[e.uid], box, sc, STATE_TERMINATED))
        self._ev_ptr = len(evs)
        return ended

    @staticmethod
    def _track(track_id: int, box, score: float, state: str) -> Track:
        return {"track_id": int(track_id), "bbox": [float(v) for v in box],
                "score": float(score), "state": state}


# --------------------------------------------------------------------------- #
# File kết quả MOT (hợp đồng mục 5, đầu vào của TV4)
# --------------------------------------------------------------------------- #
def tracks_to_mot_lines(frame_idx: int, tracks: Iterable[Track]) -> List[str]:
    """Mỗi track new/tracked -> một dòng  frame,id,x,y,w,h,score,-1,-1,-1  (x,y = góc trên trái)."""
    lines = []
    for t in tracks:
        if t["state"] not in DRAWN_STATES:
            continue
        x1, y1, x2, y2 = t["bbox"]
        lines.append(f"{frame_idx},{t['track_id']},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                     f"{t['score']:.2f},-1,-1,-1")
    return lines


def write_mot(path: str, results: Dict[int, List[Track]]) -> int:
    """results[frame_idx] = list[Track]  ->  file .txt định dạng MOT. Trả về số dòng đã ghi."""
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        for f in sorted(results):
            for line in tracks_to_mot_lines(f, results[f]):
                fp.write(line + "\n")
                n += 1
    return n
