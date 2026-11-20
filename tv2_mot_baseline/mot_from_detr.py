"""
mot_from_detr.py  -  phần của Lương Thị Mai Phương: nhận bounding box của bạn làm phần phát hiện -> theo dõi ID.

Đầu vào (lấy từ code của bạn đi trước):
    video_reader.py   -> VideoReader: đọc video / webcam thành ma trận ảnh BGR (cao x rộng x 3)
    detr_demo.py      -> outputs/detections.csv  (frame_index,time_s,label,score,x1,y1,x2,y2)
Đầu ra:
    outputs/tracked.mp4   video có hộp + ID
    outputs/tracks.csv    frame,id,x1,y1,x2,y2,score  (mỗi dòng = 1 đối tượng ở 1 frame)

Cách dùng (đặt file này CÙNG THƯ MỤC với video_reader.py và mot_baseline_demo.py):
    # B1: bạn kia (hoặc chính bạn) tạo detections.csv:
    python detr_demo.py --input data/sample.mp4 --limit 300
    # B2: theo dõi ID từ file csv đó
    python mot_from_detr.py --input data/sample.mp4 --detections outputs/detections.csv
    # hoặc chạy DETR trực tiếp trong lúc theo dõi (cần torch + transformers + internet lần đầu):
    python mot_from_detr.py --input data/sample.mp4 --live
Tuỳ chọn: --labels person car   --min-score 0.7   --every 1   --limit 300   --velocity/--no-velocity   --preview
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys

import cv2
import numpy as np

import mot_baseline_demo as mb

try:
    from video_reader import VideoReader
except ImportError:
    # Trong repo nhóm, video_reader.py (TV1) nằm ở thư mục anh em tv1_detr/ -> thêm vào đường dẫn tìm module
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tv1_detr"))
    from video_reader import VideoReader

Det = Tuple[str, float, List[float]]          # (nhãn lớp, điểm, [x1,y1,x2,y2])


def load_detections_csv(path: str) -> Dict[int, List[Det]]:
    """Đọc detections.csv của detr_demo.py -> {frame_index: [(label, score, box), ...]}."""
    out: Dict[int, List[Det]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            out[int(r["frame_index"])].append(
                (r["label"], float(r["score"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
    return out


def filter_dets(dets: Sequence[Det], labels, min_score) -> List[Det]:
    return [d for d in dets if d[1] >= min_score and (not labels or d[0] in labels)]


def make_live_detector(threshold: float, size: int):
    """Chạy DETR trực tiếp (giống detr_demo.py). Trả hàm rgb -> [(label, score, box)]."""
    import torch
    from PIL import Image
    from transformers import pipeline
    detector = pipeline("object-detection", model="facebook/detr-resnet-50",
                        device=0 if torch.cuda.is_available() else -1)
    try:
        detector.image_processor.size = {"shortest_edge": size, "longest_edge": int(size * 1.67)}
    except Exception as e:
        print("Không đổi được cỡ ảnh vào DETR, dùng mặc định:", e)

    def run(rgb):
        res = detector(Image.fromarray(rgb), threshold=threshold)
        return [(str(d["label"]), float(d["score"]),
                 [float(d["box"][k]) for k in ("xmin", "ymin", "xmax", "ymax")]) for d in res]
    return run


PAL = [(60, 76, 231), (219, 152, 52), (113, 204, 46), (15, 196, 241), (182, 89, 155), (34, 126, 230)]


def draw(img, rows, trk, k, vel, title):
    for tid, box, label, score in rows:
        c = PAL[(tid - 1) % len(PAL)]
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, f"ID {tid} {label} {score:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    for t in trk.tracks:
        if t.state == mb.TrackState.LOST:
            p = t.predict(k, vel).astype(int)
            cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (140, 140, 140), 1)
            cv2.putText(img, f"ID {t.track_id} LOST", (p[0], p[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="đường dẫn video hoặc chỉ số webcam")
    p.add_argument("--detections", help="detections.csv của detr_demo.py (không có thì phải dùng --live)")
    p.add_argument("--live", action="store_true", help="chạy DETR trực tiếp thay vì đọc csv")
    p.add_argument("--labels", nargs="*", default=["person"], help="chỉ theo dõi các lớp này; để trống = mọi lớp")
    p.add_argument("--min-score", type=float, default=0.7)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--size", type=int, default=400, help="cỡ ảnh vào DETR khi --live")
    p.add_argument("--no-velocity", action="store_true", help="tắt ngoại suy vận tốc (IoU thuần)")
    p.add_argument("--iou-thr", type=float, default=0.3)
    p.add_argument("--max-age", type=int, default=5)
    p.add_argument("--output", default="outputs/tracked.mp4")
    p.add_argument("--tracks-csv", default="outputs/tracks.csv")
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()

    if not a.live and not a.detections:
        p.error("cần --detections <csv> hoặc --live")
    source = int(a.input) if a.input.isdigit() else a.input
    vel = not a.no_velocity
    table = None if a.live else load_detections_csv(a.detections)
    live = make_live_detector(a.min_score, a.size) if a.live else None
    labels = set(a.labels or [])

    trk = mb.IoUTracker(iou_thr=a.iou_thr, max_age=a.max_age, use_velocity=vel)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.tracks_csv).parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    with VideoReader(source) as reader, open(a.tracks_csv, "w", newline="", encoding="utf-8") as fp:
        cw = csv.writer(fp)
        cw.writerow(["frame", "id", "x1", "y1", "x2", "y2", "score"])
        fps = reader.info.fps / a.every if reader.info.fps > 0 else 25.0
        try:
            for k, f in enumerate(reader.frames(every=a.every, limit=a.limit)):
                dets = live(f.rgb) if live else table.get(f.index, [])
                dets = filter_dets(dets, labels, a.min_score)
                boxes = np.array([d[2] for d in dets], float).reshape(-1, 4)   # <- ma trận (số det, 4) đưa vào tracker
                outs = trk.step(k, boxes)                  # k = số thứ tự frame ĐÃ XỬ LÝ (đều đặn, kể cả khi --every > 1)
                rows = [(tid, b, dets[d][0], dets[d][1]) for tid, b, d, _ in outs]
                for tid, b, _, sc in rows:
                    cw.writerow([f.index, tid, *[round(float(v), 1) for v in b], round(sc, 4)])
                vis = draw(f.bgr.copy(), rows, trk, k, vel, f"frame {f.index}  track: {len(trk.tracks)}")
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("Không khởi tạo được bộ ghi MP4")
                writer.write(vis)
                n += 1
                if a.preview:
                    cv2.imshow("MOT - nhan q de thoat", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if writer is not None:
                writer.release()
            if a.preview:
                cv2.destroyAllWindows()
    print(f"Đã xử lý {n} frame, tạo {trk._next_id - 1} track.")
    print("Video:", Path(a.output).resolve())
    print("Bảng ID:", Path(a.tracks_csv).resolve())
    for e in trk.events[:20]:
        print("  ", e)


if __name__ == "__main__":
    main()
