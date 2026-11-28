"""Pipeline giả lập: trả kết quả tracking đúng định dạng interface.md.

Dùng thay cho module thật của TV1 (detector), TV2 (track manager), TV3 (ID decoder)
cho đến khi các module đó hoàn thành.

Để thử các thông số tracking, vật thể ID 2 bị "che khuất" từ frame 30 đến 44 (15 frame):
- Nếu miss_tolerance >= 15: track được giữ, khi hiện lại vẫn là ID 2.
- Nếu miss_tolerance < 15: track bị kết thúc, khi hiện lại nhận ID mới.
- Nếu show_lost=True: trong lúc bị che, track được trả về với state "lost".
"""

HIDE_START, HIDE_END = 30, 44     # vật thể thứ 2 biến mất trong khoảng này


class MockPipeline:
    def __init__(self, checkpoint_path=None, num_objects=3, miss_tolerance=30, show_lost=False):
        self.checkpoint_path = checkpoint_path  # bản giả không dùng, chỉ giữ cho đúng hợp đồng
        self.num_objects = num_objects
        self.miss_tolerance = miss_tolerance
        self.show_lost = show_lost
        self.reset()

    def reset(self):
        """Gọi trước khi xử lý một video mới."""
        self.ids = {i: i + 1 for i in range(self.num_objects)}   # vật thể i -> track_id
        self.next_id = self.num_objects + 1
        self.missed = {}                                          # vật thể -> số frame đã mất
        self.last_box = {}
        self.extra_id = None                                      # ID của vật thể xuất hiện ở frame 60

    def finish(self):
        """Gọi khi hết video (giống TrackerTV2.finish). Bản giả không cần làm gì."""
        return []

    def _new_id(self):
        new = self.next_id
        self.next_id += 1
        return new

    def process(self, frame, frame_idx):
        """Nhận 1 frame (BGR, HxWx3) và số thứ tự frame (bắt đầu từ 1).
        Trả về danh sách track theo định dạng trong interface.md."""
        h, w = frame.shape[:2]
        box_w, box_h = w // 8, h // 4
        tracks = []
        for i in range(self.num_objects):
            speed = 3 + 2 * i
            x1 = (frame_idx * speed + i * w // 3) % (w - box_w)
            y1 = (i + 1) * h // (self.num_objects + 2)
            box = [float(x1), float(y1), float(x1 + box_w), float(y1 + box_h)]

            hidden = (i == 1 and HIDE_START <= frame_idx <= HIDE_END)
            if hidden:
                self.missed[i] = self.missed.get(i, 0) + 1
                if self.missed[i] <= self.miss_tolerance and self.show_lost and i in self.last_box:
                    tracks.append({"track_id": self.ids[i], "bbox": self.last_box[i],
                                   "score": 0.0, "state": "lost"})
                continue

            state = "tracked"
            if self.missed.get(i, 0) > self.miss_tolerance:     # mất quá lâu -> ID mới
                self.ids[i] = self._new_id()
                state = "new"
            elif frame_idx == 1:
                state = "new"
            self.missed[i] = 0
            self.last_box[i] = box
            tracks.append({"track_id": self.ids[i], "bbox": box,
                           "score": 0.90 - 0.05 * i, "state": state})

        # Thêm một vật thể mới xuất hiện từ frame 60 để thử trường hợp ID mới
        if frame_idx >= 60:
            state = "tracked"
            if self.extra_id is None:
                self.extra_id = self._new_id()
                state = "new"
            tracks.append({"track_id": self.extra_id,
                           "bbox": [w * 0.7, h * 0.1, w * 0.7 + box_w, h * 0.1 + box_h],
                           "score": 0.75, "state": state})
        return tracks