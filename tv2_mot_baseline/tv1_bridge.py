"""
tv1_bridge.py  -  Cầu nối giữa TV1 (detector) và TV2 (track manager)
====================================================================
Mục đích: chạy và kiểm thử phần TV2 TRƯỚC KHI detector thật của TV1 sẵn sàng, rồi thay bằng detector thật
mà không phải sửa code TV2.  Mọi "detector" ở đây đều có cùng chữ ký:

    detector(frame, frame_idx) -> list[Detection]        # Detection theo hợp đồng mục 2

Ba nguồn detection:
  MockDetector      detection giả lập từ scenarios.py (có embedding giả), không cần video
  CsvDetector       đọc detections.csv do detr_demo.py của TV1 sinh ra (frame_index,time_s,label,score,x1,y1,x2,y2)
                    CSV đó KHÔNG có embedding -> tạm dùng HistogramEmbedder (histogram màu của vùng ảnh)
                    làm "embedding giả". Khi TV1 đưa embedding thật, dùng detector thật của TV1 thay vào đây.
  adapt_tv1_output  chuyển đầu ra thật của TV1 (nếu khác hợp đồng) sang đúng hợp đồng
"""
from __future__ import annotations

import csv
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from scenarios import Scenario
from tv2_interface import Detection, check_detection, to_numpy


# --------------------------------------------------------------------------- #
class HistogramEmbedder:
    """Embedding GIẢ từ ảnh: histogram màu HSV của vùng bbox, chuẩn hoá L2.
    Chỉ để thay thế embedding của DETR khi chưa có; không thay thế được embedding học sâu."""

    def __init__(self, bins=(8, 4, 4)):
        self.bins = bins

    def __call__(self, frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        import cv2
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(w, int(np.ceil(bbox[2]))), min(h, int(np.ceil(bbox[3])))
        if x2 <= x1 or y2 <= y1:
            return np.ones(int(np.prod(self.bins)), np.float32) / np.sqrt(np.prod(self.bins))
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256]).flatten()
        hist = hist.astype(np.float32)
        return hist / (np.linalg.norm(hist) + 1e-9)


# --------------------------------------------------------------------------- #
class CsvDetector:
    """Đọc detections.csv của detr_demo.py.

    frame_offset: detr_demo.py ghi frame_index tính từ 0, còn hợp đồng yêu cầu frame_idx từ 1 -> offset = 1.
    labels      : chỉ giữ các lớp này (mặc định chỉ 'person'); None = giữ tất cả.
    """

    def __init__(self, csv_path: str, labels: Optional[Iterable[str]] = ("person",),
                 min_score: float = 0.0, embedder: Optional[Callable] = None, frame_offset: int = 1):
        self.embedder = embedder or HistogramEmbedder()
        self.table: Dict[int, List[dict]] = defaultdict(list)
        keep = set(labels) if labels else None
        with open(csv_path, newline="", encoding="utf-8") as fp:
            for r in csv.DictReader(fp):
                if keep and r["label"] not in keep:
                    continue
                score = float(r["score"])
                if score < min_score:
                    continue
                self.table[int(r["frame_index"]) + frame_offset].append(
                    {"bbox": [float(r[k]) for k in ("x1", "y1", "x2", "y2")], "score": score})

    def __call__(self, frame: np.ndarray, frame_idx: int) -> List[Detection]:
        return [{"bbox": d["bbox"], "score": d["score"], "embedding": self.embedder(frame, d["bbox"])}
                for d in self.table.get(frame_idx, [])]


# --------------------------------------------------------------------------- #
class MockDetector:
    """Detection giả lập (kèm embedding giả) từ một Scenario; frame_idx = 1 ứng với frame đầu của kịch bản."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    def __call__(self, frame, frame_idx: int) -> List[Detection]:
        i = frame_idx - 1
        if not (0 <= i < len(self.scenario.frames)):
            return []
        fr = self.scenario.frames[i]
        return [{"bbox": [float(v) for v in b], "score": float(s), "embedding": f.astype(np.float32)}
                for b, s, f in zip(fr.boxes, fr.scores, fr.feats)]

    def gt_ids(self, frame_idx: int):
        return self.scenario.frames[frame_idx - 1].gt_ids


# --------------------------------------------------------------------------- #
def adapt_tv1_output(raw: Iterable[dict], bbox_key="bbox", score_key="score", emb_key="embedding",
                     xywh: bool = False) -> List[Detection]:
    """Chuyển đầu ra của TV1 sang hợp đồng, nếu TV1 dùng tên khoá khác hoặc box dạng [x, y, w, h].
    Sửa tham số khi biết định dạng thật trong detector_adapter.py của TV1."""
    out = []
    for r in raw:
        b = [float(v) for v in r[bbox_key]]
        if xywh:
            b = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
        out.append({"bbox": b, "score": float(r[score_key]), "embedding": to_numpy(r[emb_key])})
    return out


def report_detections(dets: Sequence[Detection]) -> str:
    """Báo cáo ngắn về một lô detection: dùng khi đối chiếu với TV1 (kích thước C, điểm, lỗi hợp đồng)."""
    if not dets:
        return "0 detection"
    errs = [(i, check_detection(d)) for i, d in enumerate(dets)]
    bad = [(i, e) for i, e in errs if e]
    dims = sorted({int(to_numpy(d['embedding']).size) for d in dets if 'embedding' in d})
    scores = [float(d["score"]) for d in dets if "score" in d]
    s = (f"{len(dets)} detection | C (số chiều embedding) = {dims} | "
         f"score từ {min(scores):.2f} đến {max(scores):.2f} | lỗi hợp đồng: {len(bad)}")
    for i, e in bad[:5]:
        s += f"\n  #{i}: {e}"
    return s
