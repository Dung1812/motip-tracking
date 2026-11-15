"""DETR detection on live webcam, smooth preview (NOT tracking, no persistent IDs).

Detection runs in a background thread, so the preview never freezes.

Run: python detr_webcam.py
     python detr_webcam.py --size 320            # faster, less accurate
     python detr_webcam.py --input 1 --threshold 0.8
Press q in the preview window to quit.
"""
import argparse
import os
import threading
import time

import cv2
import torch
from transformers import pipeline

from video_reader import VideoReader


class AsyncDetector:
    """Runs the detector in a worker thread on the newest frame only."""

    def __init__(self, detector, threshold):
        self.detector = detector
        self.threshold = threshold
        self.lock = threading.Lock()
        self.pending = None
        self.busy = False
        self.predictions = []
        self.fps = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, rgb):
        with self.lock:
            if self.busy or self.pending is not None:
                return
            self.pending = rgb

    def _run(self):
        from PIL import Image
        while self.running:
            with self.lock:
                rgb, self.pending = self.pending, None
                self.busy = rgb is not None
            if rgb is None:
                time.sleep(0.005)
                continue
            t0 = time.time()
            preds = self.detector(Image.fromarray(rgb), threshold=self.threshold)
            with self.lock:
                self.predictions = preds
                self.fps = 1.0 / max(time.time() - t0, 1e-6)
                self.busy = False

    def result(self):
        with self.lock:
            return list(self.predictions), self.fps

    def stop(self):
        self.running = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="0", help="webcam index (0, 1...) or video path")
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--size", type=int, default=400,
                   help="DETR input size (shortest side). Smaller = faster. Default model uses 800")
    a = p.parse_args()

    source = int(a.input) if a.input.isdigit() else a.input

    # Leave a couple of CPU cores free so the preview window stays smooth
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

    device = 0 if torch.cuda.is_available() else -1
    detector = pipeline("object-detection", model="facebook/detr-resnet-50", device=device)
    try:
        detector.image_processor.size = {"shortest_edge": a.size, "longest_edge": int(a.size * 1.67)}
    except Exception as e:
        print("Could not change DETR input size, using default:", e)

    det = AsyncDetector(detector, a.threshold)

    try:
        with VideoReader(source) as reader:
            for f in reader.frames():
                det.submit(f.rgb)
                predictions, det_fps = det.result()

                vis = f.bgr.copy()
                for d in predictions:
                    b = d["box"]
                    x1, y1, x2, y2 = int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"])
                    label = f'{d["label"]} {float(d["score"]):.2f}'
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, label, (x1, max(20, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(vis, f"DETR: {det_fps:.1f} fps", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("DETR webcam - press q to quit", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        det.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
cls