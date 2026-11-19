"""Giao diện MOTIP Tracking bằng Streamlit.

Chạy: streamlit run app.py
Hiện dùng MockPipeline (dữ liệu giả lập). Khi module thật xong, chỉ cần đổi
dòng tạo pipeline trong hàm get_pipeline().
"""

import os
import tempfile

import cv2
import streamlit as st

from mock_pipeline import MockPipeline
from video_export import VideoExporter

st.set_page_config(page_title="MOTIP Tracking", layout="wide")
st.title("MOTIP Tracking - Demo theo dõi đa đối tượng")


def get_pipeline(checkpoint_path, num_objects):
    # Sau này thay bằng: return Pipeline(checkpoint_path=checkpoint_path)
    return MockPipeline(checkpoint_path=checkpoint_path, num_objects=num_objects)


# ---------- Thanh bên: cấu hình ----------
st.sidebar.header("Cấu hình")
checkpoint = st.sidebar.text_input("Đường dẫn checkpoint", "weights/motip.pth",
                                   help="Bản giả lập chưa dùng tới, chỉ để sẵn chỗ")
num_objects = st.sidebar.slider("Số đối tượng giả lập", 1, 6, 3)
st.sidebar.info("Đang chạy với dữ liệu giả lập (MockPipeline).")

# ---------- 1. Tải video lên ----------
uploaded = st.file_uploader("1. Tải video lên", type=["mp4", "avi", "mov"])

if uploaded is not None:
    # Lưu file tải lên ra thư mục tạm, vì OpenCV cần đường dẫn file
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, uploaded.name)
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    c1, c2, c3 = st.columns(3)
    c1.metric("Kích thước", f"{width}x{height}")
    c2.metric("FPS", f"{fps:.0f}")
    c3.metric("Số frame", total)

    # ---------- 2. Xử lý ----------
    if st.button("2. Chạy tracking", type="primary"):
        out_video = os.path.join(tmp_dir, "ket_qua.mp4")
        out_txt = os.path.join(tmp_dir, "ket_qua.txt")

        pipeline = get_pipeline(checkpoint, num_objects)
        pipeline.reset()
        exporter = VideoExporter(out_video, out_txt, fps, width, height)

        progress = st.progress(0, text="Đang xử lý...")
        preview = st.empty()          # chỗ hiển thị frame đang xử lý
        count_box = st.empty()        # chỗ hiển thị số đối tượng

        cap = cv2.VideoCapture(input_path)
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1                                  # MOT đánh số từ 1
                tracks = pipeline.process(frame, frame_idx)
                drawn = exporter.write(frame_idx, frame, tracks)

                if total > 0:
                    progress.progress(min(frame_idx / total, 1.0),
                                      text=f"Frame {frame_idx}/{total}")
                if frame_idx % 5 == 0:                          # 5 frame cập nhật 1 lần cho nhẹ
                    preview.image(drawn, channels="BGR", caption=f"Frame {frame_idx}")
                    count_box.write(f"Số đối tượng ở frame này: **{len(tracks)}**")
        finally:
            cap.release()
            exporter.close()

        progress.progress(1.0, text="Xong")
        # Lưu kết quả vào session_state để không mất khi trang chạy lại
        st.session_state["result"] = {
            "video": out_video, "txt": out_txt,
            "frames": frame_idx, "codec": exporter.codec,
        }

# ---------- 3. Kết quả ----------
result = st.session_state.get("result")
if result:
    st.subheader("3. Kết quả")
    st.success(f"Đã xử lý {result['frames']} frame.")
    if result["codec"] == "avc1":
        st.video(result["video"])
    else:
        st.warning("Máy chưa hỗ trợ mã H.264 nên trình duyệt không phát được video. "
                   "Hãy tải video về để xem.")

    with open(result["video"], "rb") as f:
        st.download_button("Tải video kết quả", f, file_name="ket_qua.mp4", mime="video/mp4")
    with open(result["txt"], "rb") as f:
        st.download_button("Tải file tracking (MOT)", f, file_name="ket_qua.txt",
                           mime="text/plain")
