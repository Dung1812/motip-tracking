"""Pipeline giả lập: trả kết quả tracking đúng định dạng interface.md.

Dùng thay cho module thật của TV1 (detector), TV2 (track manager), TV3 (ID decoder)
cho đến khi các module đó hoàn thành.
"""


class MockPipeline:
    def __init__(self, checkpoint_path=None, num_objects=3):
        self.checkpoint_path = checkpoint_path  # bản giả không dùng, chỉ giữ cho đúng hợp đồng
        self.num_objects = num_objects

    def reset(self):
        """Gọi trước khi xử lý một video mới."""
        pass

    def process(self, frame, frame_idx):
        """Nhận 1 frame (BGR, HxWx3) và số thứ tự frame (bắt đầu từ 1).
        Trả về danh sách track theo định dạng trong interface.md."""
        h, w = frame.shape[:2]
        box_w, box_h = w // 8, h // 4
        tracks = []
        for i in range(self.num_objects):
            # Mỗi vật thể chạy ngang màn hình ở một độ cao khác nhau, tốc độ khác nhau
            speed = 3 + 2 * i
            x1 = (frame_idx * speed + i * w // 3) % (w - box_w)
            y1 = (i + 1) * h // (self.num_objects + 2)
            tracks.append({
                "track_id": i + 1,
                "bbox": [float(x1), float(y1), float(x1 + box_w), float(y1 + box_h)],
                "score": 0.90 - 0.05 * i,
                "state": "tracked",
            })
        # Vật thể thứ 4 chỉ xuất hiện từ frame 60 để thử trường hợp ID mới
        if frame_idx >= 60:
            tracks.append({
                "track_id": 4,
                "bbox": [w * 0.7, h * 0.1, w * 0.7 + box_w, h * 0.1 + box_h],
                "score": 0.75,
                "state": "new" if frame_idx == 60 else "tracked",
            })
        return tracks
