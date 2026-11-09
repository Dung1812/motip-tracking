import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk


class App:
    def __init__(self, root):
        self.root = root
        root.title("MOTIP Tracking - Khung giao diện")
        self.cap = None           # đối tượng đọc video
        self.video_path = None    # đường dẫn video đang mở
        self.playing = False      # đang chạy hay đang dừng

        # Vùng hiển thị video
        self.label = tk.Label(root, text="Chưa mở video", width=80, height=24,
                              bg="black", fg="white")
        self.label.pack(padx=10, pady=10)

        # Hàng 3 nút
        frame = tk.Frame(root)
        frame.pack(pady=(0, 10))
        tk.Button(frame, text="Mở video", command=self.open_video).pack(side="left", padx=5)
        self.btn_play = tk.Button(frame, text="Chạy", command=self.toggle_play)
        self.btn_play.pack(side="left", padx=5)
        tk.Button(frame, text="Xuất video", command=self.export_video).pack(side="left", padx=5)

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")])
        if not path:                      # người dùng bấm Hủy
            return
        if self.cap is not None:
            self.cap.release()            # đóng video cũ
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được video")
            self.cap = None
            return
        self.video_path = path
        self.playing = False
        self.btn_play.config(text="Chạy")
        self.show_next_frame()            # hiện frame đầu tiên

    def toggle_play(self):
        if self.cap is None:
            messagebox.showinfo("Thông báo", "Hãy mở video trước")
            return
        self.playing = not self.playing   # đảo trạng thái
        self.btn_play.config(text="Dừng" if self.playing else "Chạy")
        if self.playing:
            self.play_loop()

    def play_loop(self):
        if not self.playing:
            return
        ok = self.show_next_frame()
        if not ok:                        # hết video
            self.playing = False
            self.btn_play.config(text="Chạy")
            return
        self.root.after(30, self.play_loop)   # 30 ms sau gọi lại chính nó

    def show_next_frame(self):
        ok, frame = self.cap.read()       # đọc 1 frame
        if not ok:
            return False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV dùng BGR, Tkinter cần RGB
        frame = cv2.resize(frame, (640, 360))
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.label.config(image=img, text="", width=640, height=360)
        self.label.image = img            # giữ tham chiếu, nếu không ảnh sẽ biến mất
        return True

    def export_video(self):
        if self.video_path is None:
            messagebox.showinfo("Thông báo", "Hãy mở video trước")
            return
        out_path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                                filetypes=[("MP4", "*.mp4")])
        if not out_path:
            return
        cap = cv2.VideoCapture(self.video_path)   # mở bản đọc riêng, không ảnh hưởng phần đang xem
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)           # ghi frame ra file mới
            count += 1
        cap.release()
        writer.release()
        messagebox.showinfo("Xong", f"Đã xuất {count} frame")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()