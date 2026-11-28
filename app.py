"""Giao diện MOTIP Tracking bằng Streamlit, có đăng nhập hai vai trò.

Chạy: streamlit run app.py
- Người dùng (user): chỉ tải video lên, chạy tracking, tải kết quả.
- Quản trị (admin): thêm trang Quản trị để chỉnh cấu hình và quản lý tài khoản.
Tạo admin đầu tiên: python tao_tai_khoan.py admin --role admin
"""

import os
import tempfile

import cv2
import streamlit as st

from app_settings import TRACKER_TYPES, load_settings, save_settings
from auth import ROLES, add_user, delete_user, load_users, register_user, verify_login
from mock_pipeline import MockPipeline
from video_export import VideoExporter, output_paths

st.set_page_config(page_title="MOTIP Tracking", page_icon="🎯", layout="wide")


def get_pipeline(settings):
    """Tạo pipeline theo cấu hình của admin.
    Sau này thêm nhánh "motip": return Pipeline(settings)  (detector TV1 + tracker TV2 + ID Decoder TV3)"""
    if settings["tracker_type"] == "mock":
        return MockPipeline(checkpoint_path=settings["det_checkpoint"],
                            num_objects=settings["num_objects"],
                            miss_tolerance=settings["miss_tolerance"],
                            show_lost=settings["show_lost"])
    raise RuntimeError("Mô hình MOTIP thật chưa được tích hợp. Admin hãy chọn chế độ Giả lập.")


# =====================================================================
# Trang đăng nhập
# =====================================================================
def login_page():
    st.title("MOTIP Tracking")
    st.caption("Hệ thống theo dõi đa đối tượng trong video")

    if not load_users():
        st.warning("Hệ thống chưa có tài khoản nào. Người quản trị cần chạy lệnh:\n\n"
                   "`python tao_tai_khoan.py admin --role admin`")
        return

    if load_settings()["allow_register"]:
        tab_login, tab_register = st.tabs(["Đăng nhập", "Đăng ký"])
    else:
        tab_login, tab_register = st.container(), None

    with tab_login:
        with st.form("login"):
            username = st.text_input("Tên đăng nhập")
            password = st.text_input("Mật khẩu", type="password")
            submitted = st.form_submit_button("Đăng nhập", type="primary")
        if submitted:
            role = verify_login(username, password)
            if role is None:
                st.error("Sai tên đăng nhập hoặc mật khẩu.")
            else:
                st.session_state["user"] = {"username": username.strip(), "role": role}
                st.rerun()                   # chạy lại trang để vào giao diện chính

    if tab_register is not None:
        with tab_register:
            with st.form("register", clear_on_submit=True):
                new_name = st.text_input("Chọn tên đăng nhập",
                                         help="3-30 ký tự: chữ không dấu, số, dấu _ . -")
                new_pass = st.text_input("Mật khẩu (ít nhất 6 ký tự)", type="password")
                new_pass2 = st.text_input("Nhập lại mật khẩu", type="password")
                registered = st.form_submit_button("Tạo tài khoản", type="primary")
            if registered:
                try:
                    register_user(new_name, new_pass, new_pass2)
                    st.success(f"Đã tạo tài khoản '{new_name.strip()}'. "
                               "Chuyển sang tab Đăng nhập để vào hệ thống.")
                except ValueError as e:
                    st.error(str(e))


# =====================================================================
# Trang xử lý video (cả user và admin đều thấy)
# =====================================================================
def process_page(settings):
    st.header("Theo dõi đối tượng trong video")
    uploaded = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov"])

    if uploaded is not None:
        tmp_dir = tempfile.mkdtemp()
        input_path = os.path.join(tmp_dir, uploaded.name)
        with open(input_path, "wb") as f:
            f.write(uploaded.getbuffer())

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Không đọc được video. Hãy thử một file khác.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        c1, c2, c3 = st.columns(3)
        c1.metric("Kích thước", f"{width}x{height}")
        c2.metric("FPS", f"{fps:.0f}")
        c3.metric("Số frame", total)

        if st.button("Chạy tracking", type="primary"):
            run_tracking(input_path, uploaded.name, tmp_dir, fps, width, height, total, settings)

    show_result()


def run_tracking(input_path, video_name, tmp_dir, fps, width, height, total, settings):
    out_video, out_txt = output_paths(video_name, tmp_dir)
    try:
        pipeline = get_pipeline(settings)
    except Exception as e:
        st.error(f"Không khởi tạo được mô hình: {e}")
        return

    pipeline.reset()
    exporter = VideoExporter(out_video, out_txt, fps, width, height)
    progress = st.progress(0, text="Đang xử lý...")
    preview = st.empty()
    count_box = st.empty()

    cap = cv2.VideoCapture(input_path)
    frame_idx = 0
    all_ids, max_count, new_count = set(), 0, 0      # thống kê tracking
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            tracks = pipeline.process(frame, frame_idx)
            drawn = exporter.write(frame_idx, frame, tracks)

            active = [t for t in tracks if t["state"] in ("new", "tracked")]
            all_ids.update(t["track_id"] for t in active)
            max_count = max(max_count, len(active))
            new_count += sum(1 for t in active if t["state"] == "new")

            if total > 0:
                progress.progress(min(frame_idx / total, 1.0), text=f"Frame {frame_idx}/{total}")
            if frame_idx % 5 == 0:
                preview.image(drawn, channels="BGR", caption=f"Frame {frame_idx}")
                count_box.write(f"Đang theo dõi **{len(active)}** đối tượng ở frame này")
        pipeline.finish()
    except Exception as e:
        st.error(f"Lỗi khi xử lý ở frame {frame_idx}: {e}")
        return
    finally:
        cap.release()
        exporter.close()

    progress.progress(1.0, text="Xong")
    st.session_state["result"] = {"video": out_video, "txt": out_txt,
                                  "frames": frame_idx, "codec": exporter.codec,
                                  "unique_ids": len(all_ids), "max_count": max_count,
                                  "new_count": new_count}


def show_result():
    result = st.session_state.get("result")
    if not result:
        return
    st.subheader("Kết quả tracking")
    st.success(f"Đã xử lý {result['frames']} frame.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng số ID khác nhau", result["unique_ids"],
              help="Số đối tượng riêng biệt hệ thống đã theo dõi trong cả video")
    m2.metric("Nhiều nhất trong 1 frame", result["max_count"])
    m3.metric("Số lần tạo ID mới", result["new_count"],
              help="Lớn hơn số người thật trong video nghĩa là có người bị gán lại ID mới")
    if result["codec"] == "avc1":
        st.video(result["video"])
    else:
        st.info("Trình duyệt không phát trực tiếp được video này. Hãy tải về để xem.")
    with open(result["video"], "rb") as f:
        st.download_button("Tải video kết quả", f, file_name=os.path.basename(result["video"]),
                           mime="video/mp4")
    with open(result["txt"], "rb") as f:
        st.download_button("Tải file tracking (MOT)", f, file_name=os.path.basename(result["txt"]),
                           mime="text/plain")


# =====================================================================
# Trang quản trị (chỉ admin thấy)
# =====================================================================
def admin_page(settings):
    st.header("Cấu hình hệ thống")
    with st.form("settings"):
        st.subheader("1. Detector (TV1) - phát hiện người trong từng frame")
        det_checkpoint = st.text_input("Checkpoint detector / full MOTIP", settings["det_checkpoint"])
        c1, c2 = st.columns(2)
        det_thresh = c1.slider("Ngưỡng phát hiện (DET_THRESH)", 0.0, 1.0,
                               float(settings["det_thresh"]), 0.05,
                               help="Bỏ các khung có độ tự tin thấp hơn ngưỡng này")
        max_detections = c2.number_input("Số khung tối đa mỗi frame", 1, 300,
                                         int(settings["max_detections"]))
        c3, c4 = st.columns(2)
        max_long_side = c3.number_input("Cạnh dài ảnh đầu vào (pixel)", 320, 1920,
                                        int(settings["max_long_side"]), 80,
                                        help="Giảm xuống để chạy nhanh hơn trên CPU")
        device = c4.selectbox("Thiết bị chạy mô hình", ["cpu", "cuda"],
                              index=["cpu", "cuda"].index(settings["device"]))

        st.subheader("2. Tracking (TV2 + TV3) - gán và giữ ID qua các frame")
        tracker_type = st.selectbox("Bộ tracking", list(TRACKER_TYPES),
                                    format_func=TRACKER_TYPES.get,
                                    index=list(TRACKER_TYPES).index(settings["tracker_type"]))
        id_checkpoint = st.text_input("Checkpoint ID Decoder (để trống nếu dùng full MOTIP)",
                                      settings["id_checkpoint"])
        c5, c6 = st.columns(2)
        id_thresh = c5.slider("Ngưỡng gán ID cũ (ID_THRESH)", 0.0, 1.0,
                              float(settings["id_thresh"]), 0.05,
                              help="Xác suất tối thiểu để coi một phát hiện là người đã có ID")
        newborn_thresh = c6.slider("Ngưỡng tạo ID mới (NEWBORN_THRESH)", 0.0, 1.0,
                                   float(settings["newborn_thresh"]), 0.05,
                                   help="Điểm tối thiểu để tạo một ID mới cho người mới xuất hiện")
        miss_tolerance = st.number_input("Số frame giữ track khi bị mất", 0, 300,
                                         int(settings["miss_tolerance"]),
                                         help="Bị che lâu hơn số frame này thì kết thúc track; "
                                              "khi hiện lại sẽ nhận ID mới")
        show_lost = st.checkbox("Vẽ cả track đang tạm mất (khung xám)", settings["show_lost"])

        st.subheader("3. Hệ thống")
        num_objects = st.slider("Số đối tượng giả lập", 1, 6, int(settings["num_objects"]))
        allow_register = st.checkbox("Cho phép người dùng tự đăng ký tài khoản",
                                     settings["allow_register"])

        if st.form_submit_button("Lưu cấu hình", type="primary"):
            save_settings({
                "det_checkpoint": det_checkpoint, "det_thresh": det_thresh,
                "max_detections": int(max_detections), "max_long_side": int(max_long_side),
                "device": device, "tracker_type": tracker_type, "id_checkpoint": id_checkpoint,
                "id_thresh": id_thresh, "newborn_thresh": newborn_thresh,
                "miss_tolerance": int(miss_tolerance), "show_lost": show_lost,
                "num_objects": num_objects, "allow_register": allow_register,
            })
            st.success("Đã lưu cấu hình.")
    st.caption("Ở chế độ Giả lập, chỉ 'Số frame giữ track', 'Vẽ track tạm mất' và "
               "'Số đối tượng giả lập' có tác dụng. Các thông số khác dùng khi ghép mô hình thật.")

    st.header("Quản lý tài khoản")
    users = load_users()
    st.table([{"Tên đăng nhập": name, "Vai trò": info["role"]} for name, info in users.items()])

    with st.form("add_user", clear_on_submit=True):
        st.write("**Thêm tài khoản / đổi mật khẩu**")
        new_name = st.text_input("Tên đăng nhập")
        new_pass = st.text_input("Mật khẩu (ít nhất 6 ký tự)", type="password")
        new_role = st.selectbox("Vai trò", ROLES)
        if st.form_submit_button("Lưu tài khoản"):
            try:
                add_user(new_name, new_pass, new_role)
                st.success(f"Đã lưu tài khoản '{new_name.strip()}'.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    me = st.session_state["user"]["username"]
    others = [name for name in users if name != me]   # không cho tự xoá tài khoản đang dùng
    if others:
        with st.form("delete_user"):
            st.write("**Xoá tài khoản**")
            target = st.selectbox("Chọn tài khoản", others)
            if st.form_submit_button("Xoá"):
                delete_user(target)
                st.success(f"Đã xoá '{target}'.")
                st.rerun()


# =====================================================================
# Chương trình chính
# =====================================================================
user = st.session_state.get("user")
if user is None:
    login_page()
    st.stop()                                # chưa đăng nhập thì dừng ở đây

with st.sidebar:
    st.write(f"Xin chào, **{user['username']}**")
    st.caption("Quản trị viên" if user["role"] == "admin" else "Người dùng")
    if st.button("Đăng xuất"):
        st.session_state.clear()             # xoá thông tin đăng nhập và kết quả cũ
        st.rerun()

settings = load_settings()
if user["role"] == "admin":
    tab_process, tab_admin = st.tabs(["Xử lý video", "Quản trị"])
    with tab_process:
        process_page(settings)
    with tab_admin:
        admin_page(settings)
else:
    process_page(settings)