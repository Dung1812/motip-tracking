"""Video reader for a Multi-Object Tracking (MOT) project.

Run:
    python video_reader.py --input data/sample.mp4 --preview
    python video_reader.py --input data/sample.mp4 --save-frames --every 10
    python video_reader.py --input 0 --preview   # webcam

The VideoReader class can also be imported by a detector/tracker module.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass
class FrameData:
    index: int           # 0-based frame index
    timestamp_s: float   # video timestamp in seconds (approximate fallback)
    bgr: np.ndarray      # H x W x 3, uint8; OpenCV uses BGR

    @property
    def rgb(self) -> np.ndarray:
        """Return RGB copy for models expecting RGB input."""
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)


class VideoReader:
    def __init__(self, source: Union[str, int, Path]):
        if isinstance(source, Path):
            source = str(source)
        if isinstance(source, str) and source.isdigit() and not Path(source).is_file():
            source = int(source)
        if isinstance(source, str) and not Path(source).exists():
            raise FileNotFoundError(f"Video does not exist: {source}")
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Cannot open video/camera: {source}")
        self.info = VideoInfo(
            fps=float(self.cap.get(cv2.CAP_PROP_FPS)),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        self._closed = False

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def frames(self, every: int = 1, limit: int | None = None) -> Iterator[FrameData]:
        if every < 1:
            raise ValueError("every must be >= 1")
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")
        index = 0
        yielded = 0
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                break
            if index % every == 0:
                timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = (timestamp_ms / 1000.0) if timestamp_ms > 0 else (
                    index / self.info.fps if self.info.fps > 0 else 0.0
                )
                yield FrameData(index=index, timestamp_s=timestamp, bgr=bgr)
                yielded += 1
                if limit is not None and yielded >= limit:
                    break
            index += 1

    def close(self) -> None:
        if not self._closed:
            self.cap.release()
            self._closed = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Read, inspect and export video frames.")
    parser.add_argument("--input", default="data/sample.mp4", help="Video file path or camera index, e.g. 0")
    parser.add_argument("--preview", action="store_true", help="Display frames (press q to quit)")
    parser.add_argument("--save-frames", action="store_true", help="Save extracted frames to disk")
    parser.add_argument("--output", default="outputs/frames", help="Directory for saved frames")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--limit", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    if not args.preview and not args.save_frames:
        args.limit = 1 if args.limit is None else args.limit
    output = Path(args.output)
    if args.save_frames:
        output.mkdir(parents=True, exist_ok=True)

    with VideoReader(args.input) as reader:
        print("Video source:", reader.source)
        print("FPS:", reader.info.fps)
        print("Resolution:", f"{reader.info.width} x {reader.info.height}")
        print("Reported frame count:", reader.info.frame_count)
        processed = 0
        try:
            for f in reader.frames(every=args.every, limit=args.limit):
                processed += 1
                if processed == 1:
                    print("First frame shape (H, W, C):", f.bgr.shape)
                    print("First frame dtype:", f.bgr.dtype)
                if args.save_frames:
                    path = output / f"frame_{f.index:06d}.jpg"
                    if not cv2.imwrite(str(path), f.bgr):
                        raise OSError(f"Failed to save frame: {path}")
                if args.preview:
                    image = f.bgr.copy()
                    cv2.putText(image, f"Frame: {f.index}   t={f.timestamp_s:.2f}s", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("MOT Video Reader - press q to quit", image)
                    delay_ms = max(1, round(1000 * args.every / reader.info.fps)) if reader.info.fps > 0 else 30
                    if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                        break
        finally:
            if args.preview:
                cv2.destroyAllWindows()
        print("Frames processed:", processed)
        if args.save_frames:
            print("Frames saved in:", output.resolve())


if __name__ == "__main__":
    main()
