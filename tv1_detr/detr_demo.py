"""Optional DETR detection demo (NOT tracking, does not output persistent IDs).

Install extra dependencies: pip install -r requirements-detr.txt
Run: python detr_demo.py --input data/sample.mp4 --limit 30
First run downloads model weights from Hugging Face (internet required).
"""
import argparse
import csv
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import pipeline

from video_reader import VideoReader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/sample.mp4")
    p.add_argument("--output", default="outputs/detr_detection.mp4")
    p.add_argument("--csv", default="outputs/detections.csv")
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--limit", type=int, default=30)
    a = p.parse_args()

    # Model DETR ResNet-50 baseline. Change model here for future experiments.
    device = 0 if torch.cuda.is_available() else -1
    detector = pipeline("object-detection", model="facebook/detr-resnet-50", device=device)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(a.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with VideoReader(a.input) as reader:
        writer = None
        fps = reader.info.fps / a.every if reader.info.fps > 0 else 25.0
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(["frame_index", "time_s", "label", "score", "x1", "y1", "x2", "y2"])
            try:
                for f in reader.frames(every=a.every, limit=a.limit):
                    if writer is None:
                        h, w = f.bgr.shape[:2]
                        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError("Failed to initialize MP4 writer; check local codecs")
                    image = Image.fromarray(f.rgb)
                    predictions = detector(image, threshold=a.threshold)
                    vis = f.bgr.copy()
                    for d in predictions:
                        box = d["box"]
                        x1, y1, x2, y2 = [int(box[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                        score = float(d["score"])
                        label = str(d["label"])
                        csv_writer.writerow([f.index, round(f.timestamp_s, 3), label, round(score, 5), x1, y1, x2, y2])
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis, f"{label} {score:.2f}", (x1, max(20, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    writer.write(vis)
                    print(f"Frame {f.index}: {len(predictions)} detections")
            finally:
                if writer is not None:
                    writer.release()
    print("Detection video:", out.resolve())
    print("Detection CSV:", csv_path.resolve())
    print("Note: detections have no persistent IDs. MOT tracker is a separate module.")


if __name__ == "__main__":
    main()
