"""Vẽ kết quả tracking lên frame, ghi video kết quả và file tracking (định dạng MOT).

Theo interface.md v1.0 mục 6: file kết quả trùng tên video (<sequence_name>.txt),
video kết quả là <sequence_name>_tracked.mp4.
"""

import os

import cv2

DRAWN_STATES = ("new", "tracked")   # chỉ vẽ và ghi các state này


def output_paths(video_name, out_dir="."):
    """Từ tên video (vd 'dancetrack0004.mp4') -> (đường dẫn video kết quả, đường dẫn file txt)."""
    seq = os.path.splitext(os.path.basename(video_name))[0]
    return (os.path.join(out_dir, f"{seq}_tracked.mp4"),
            os.path.join(out_dir, f"{seq}.txt"))


def color_for_id(track_id):
    """Mỗi ID một màu cố định, để cùng một vật thể luôn cùng màu."""
    return ((37 * track_id) % 255, (17 * track_id) % 255, (97 * track_id) % 255)


def draw_tracks(frame, tracks, lost_tracks=()):
    """Vẽ box, ID và số đối tượng lên một bản sao của frame.
    lost_tracks (nếu có) được vẽ màu xám, nét mảnh, ghi chữ 'mat' (đang tạm mất)."""
    out = frame.copy()
    for t in lost_tracks:
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)
        cv2.putText(out, f"ID {t['track_id']} (mat)", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        color = color_for_id(t["track_id"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID {t['track_id']}"
        cv2.putText(out, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(out, f"So doi tuong: {len(tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out


def track_to_mot_line(frame_idx, t):
    """Đổi 1 track sang 1 dòng MOT: frame,id,x,y,w,h,score,-1,-1,-1"""
    x1, y1, x2, y2 = t["bbox"]
    return (f"{frame_idx},{t['track_id']},{x1:.2f},{y1:.2f},"
            f"{x2 - x1:.2f},{y2 - y1:.2f},{t['score']:.2f},-1,-1,-1\n")


class VideoExporter:
    """Mở file video và file txt đầu ra, ghi từng frame một."""

    def __init__(self, out_video_path, out_txt_path, fps, width, height):
        # Thử H.264 ("avc1") trước vì trình duyệt phát được; máy không hỗ trợ thì dùng "mp4v"
        self.writer = None
        self.codec = None
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*codec),
                                     fps, (width, height))
            if writer.isOpened():
                self.writer, self.codec = writer, codec
                break
            writer.release()
        if self.writer is None:
            raise IOError(f"Không tạo được file video: {out_video_path}")
        self.txt = open(out_txt_path, "w", encoding="utf-8")

    def write(self, frame_idx, frame, tracks):
        """Vẽ kết quả lên frame, ghi frame vào video và ghi các dòng MOT."""
        active = [t for t in tracks if t.get("state", "tracked") in DRAWN_STATES]
        lost = [t for t in tracks if t.get("state") == "lost"]
        drawn = draw_tracks(frame, active, lost)
        self.writer.write(drawn)
        for t in active:                       # file MOT chỉ ghi track new/tracked
            self.txt.write(track_to_mot_line(frame_idx, t))
        return drawn  # trả lại để giao diện hiển thị

    def close(self):
        self.writer.release()
        self.txt.close()


def process_video(input_path, out_video_path, out_txt_path, pipeline, progress_callback=None):
    """Xử lý cả video không cần giao diện (dùng để kiểm thử nhanh)."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    exporter = VideoExporter(out_video_path, out_txt_path, fps, w, h)
    pipeline.reset()
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1                       # MOT đánh số frame từ 1
            tracks = pipeline.process(frame, frame_idx)
            exporter.write(frame_idx, frame, tracks)
            if progress_callback:
                progress_callback(frame_idx, total)
        pipeline.finish()
    finally:
        cap.release()
        exporter.close()
    return frame_idx


if __name__ == "__main__":
    # Chạy thử: python video_export.py video_vao.mp4
    import sys
    from mock_pipeline import MockPipeline

    out_video, out_txt = output_paths(sys.argv[1], "outputs")
    os.makedirs("outputs", exist_ok=True)
    n = process_video(sys.argv[1], out_video, out_txt, MockPipeline(),
                      lambda i, total: print(f"\r{i}/{total}", end=""))
    print(f"\nĐã xử lý {n} frame -> {out_video}, {out_txt}")