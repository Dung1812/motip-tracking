"""Cấu hình hệ thống do admin chỉnh, người dùng thường không nhìn thấy.

Chia thành 3 nhóm:
- Detector (TV1): phát hiện người trong từng frame.
- Tracking (TV2 + TV3): nối các phát hiện qua nhiều frame, gán và giữ ID.
- Hệ thống: chế độ chạy, đăng ký tài khoản.
"""

import json
import os

SETTINGS_FILE = "settings.json"

DEFAULTS = {
    # ---- Detector (TV1) ----
    "det_checkpoint": "checkpoints/motip_dancetrack.pth",  # file trọng số detector / full MOTIP
    "det_thresh": 0.30,            # bỏ khung có score thấp hơn (DET_THRESH)
    "max_detections": 100,         # số khung tối đa mỗi frame
    "max_long_side": 1440,         # cạnh dài ảnh đưa vào mô hình; nhỏ hơn thì nhanh hơn trên CPU
    "device": "cpu",               # "cpu" hoặc "cuda"

    # ---- Tracking (TV2 + TV3) ----
    "tracker_type": "mock",        # "mock" = giả lập, "motip" = ID Decoder MOTIP (TV3)
    "id_checkpoint": "",           # checkpoint ID Decoder; để trống nếu dùng full MOTIP checkpoint
    "id_thresh": 0.20,             # xác suất tối thiểu để gán vào một ID cũ (ID_THRESH)
    "newborn_thresh": 0.60,        # điểm tối thiểu để tạo ID mới (NEWBORN_THRESH)
    "miss_tolerance": 30,          # số frame giữ một track bị mất trước khi kết thúc
    "show_lost": False,            # vẽ cả track đang tạm mất (màu xám)

    # ---- Hệ thống ----
    "num_objects": 3,              # số đối tượng giả lập
    "allow_register": True,        # cho phép người dùng tự đăng ký
}

TRACKER_TYPES = {"mock": "Giả lập (MockPipeline)", "motip": "MOTIP ID Decoder (TV3)"}


def load_settings(path=SETTINGS_FILE):
    """Đọc cấu hình; thiếu mục nào thì lấy giá trị mặc định."""
    settings = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        settings.update({k: v for k, v in saved.items() if k in DEFAULTS})
    return settings


def save_settings(settings, path=SETTINGS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)