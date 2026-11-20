"""
run_tv2_pipeline.py  -  Chạy TV1 (detector) -> TV3 (decoder) -> TV2 (track manager) theo hợp đồng interface.md
================================================================================================
Hai cách chạy:

  1) Không cần video, không cần model (kiểm tra logic TV2 bằng dữ liệu giả lập):
         python run_tv2_pipeline.py --mock occlusion
         python run_tv2_pipeline.py --mock demo --include-lost
     Kịch bản: parallel, crossing, bounce, occlusion, enter_exit, sequential, demo

  2) Video thật + detections.csv do detr_demo.py của TV1 tạo (cần opencv-python):
         python run_tv2_pipeline.py --video data/sample.mp4 --detections outputs/detections.csv
     CSV không có embedding nên dùng histogram màu thay thế (xem tv1_bridge.py).

Kết quả: file MOT (hợp đồng mục 5) ở outputs/tv2_result.txt; với --video còn có outputs/tv2_tracked.mp4.
Khi TV1/TV3 xong: thay `detector` và `decoder` ở class PipelineTV1TV2 bằng đối tượng thật, giữ nguyên phần còn lại.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional

import numpy as np

import scenarios as S
from tracking_inference import InferenceConfig
from tv1_bridge import CsvDetector, MockDetector
from tv2_interface import DRAWN_STATES, TrackerTV2, write_mot


class PipelineTV1TV2:
    """Phiên bản rút gọn của `Pipeline` (hợp đồng mục 4), chỉ gồm TV1 + TV2 (+ decoder của TV3).

        pipeline.reset()                              # trước mỗi video
        tracks = pipeline.process(frame, frame_idx)   # mỗi frame -> list[Track]
    """

    def __init__(self, detector: Callable, decoder=None, config: Optional[InferenceConfig] = None,
                 include_lost: bool = False, include_terminated: bool = False, strict: bool = True):
        self.detector = detector
        self.tracker = TrackerTV2(decoder, config, include_lost, include_terminated, strict)

    def reset(self) -> None:
        self.tracker.reset()

    def process(self, frame, frame_idx: int) -> List[dict]:
        return self.tracker.update(self.detector(frame, frame_idx), frame_idx)

    def finish(self, frame_idx: Optional[int] = None) -> List[dict]:
        return self.tracker.finish(frame_idx)


# --------------------------------------------------------------------------- #
def run_mock(name: str, out_dir: str, include_lost: bool) -> int:
    sc = S.ALL[name]()
    det = MockDetector(sc)
    pipe = PipelineTV1TV2(det, include_lost=include_lost, include_terminated=include_lost)
    pipe.reset()
    results: Dict[int, List[dict]] = {}
    assign = []
    print(f"Kịch bản '{name}': {sc.num_frames} frame")
    for f in range(1, sc.num_frames + 1):
        tracks = pipe.process(None, f)
        results[f] = tracks
        shown = [t for t in tracks if t["state"] in DRAWN_STATES]
        # đối chiếu với ground-truth: track nào ứng với đối tượng thật nào (bằng IoU lớn nhất)
        gt = sc.frames[f - 1]
        row = {}
        for t in shown:
            ious = [_iou(t["bbox"], b) for b in gt.boxes]
            if ious and max(ious) > 0.5:
                row[int(gt.gt_ids[int(np.argmax(ious))])] = t["track_id"]
        assign.append(row)
        marks = " ".join(f"{t['track_id']}:{t['state']}" for t in tracks)
        print(f"  frame {f:3d}: {marks}")
    results[sc.num_frames + 1] = pipe.finish(sc.num_frames + 1)
    m = S.evaluate_ids(assign)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tv2_result.txt")
    n = write_mot(path, results)
    print(f"\nID switch = {m.id_switches} | ID theo từng đối tượng thật: {m.ids_per_gt}")
    print(f"Đã ghi {n} dòng MOT vào {path}")
    return 0


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def run_video(video: str, csv_path: str, out_dir: str, include_lost: bool, labels, min_score: float) -> int:
    import cv2
    det = CsvDetector(csv_path, labels=labels, min_score=min_score)
    pipe = PipelineTV1TV2(det, include_lost=include_lost, include_terminated=include_lost, strict=False)
    pipe.reset()
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"Không mở được video: {video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(out_dir, exist_ok=True)
    vw = cv2.VideoWriter(os.path.join(out_dir, "tv2_tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    pal = [(60, 76, 231), (219, 152, 52), (113, 204, 46), (15, 196, 241), (182, 89, 155), (34, 126, 230)]
    results: Dict[int, List[dict]] = {}
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f += 1                                        # hợp đồng: frame_idx bắt đầu từ 1
        tracks = pipe.process(frame, f)
        results[f] = tracks
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            if t["state"] in DRAWN_STATES:
                c = pal[(t["track_id"] - 1) % len(pal)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                cv2.putText(frame, f"ID {t['track_id']}", (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            elif t["state"] == "lost":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 140, 140), 1)
        vw.write(frame)
    cap.release(); vw.release()
    results[f + 1] = pipe.finish(f + 1)
    path = os.path.join(out_dir, "tv2_result.txt")
    n = write_mot(path, results)
    arch = pipe.tracker.archive
    print(f"Đã xử lý {f} frame, {len(arch.lives)} quỹ đạo, ghi {n} dòng MOT vào {path}")
    print("Video:", os.path.join(out_dir, "tv2_tracked.mp4"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", choices=sorted(S.ALL), help="chạy bằng dữ liệu giả lập")
    ap.add_argument("--video", help="video thật")
    ap.add_argument("--detections", help="detections.csv của detr_demo.py (TV1)")
    ap.add_argument("--labels", nargs="*", default=["person"], help="lớp cần theo dõi (mặc định: person)")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--include-lost", action="store_true", help="trả cả track lost/terminated")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    if a.mock:
        return run_mock(a.mock, a.out, a.include_lost)
    if a.video and a.detections:
        return run_video(a.video, a.detections, a.out, a.include_lost, a.labels or None, a.min_score)
    ap.error("cần --mock <kịch bản>  hoặc  --video <file> --detections <csv>")


if __name__ == "__main__":
    sys.exit(main())
