"""Cấu hình hệ thống do admin chỉnh, người dùng thường không nhìn thấy."""

import json
import os

SETTINGS_FILE = "settings.json"

DEFAULTS = {
    "checkpoint_path": "weights/motip.pth",   # đường dẫn file trọng số mô hình
    "device": "cpu",                           # "cpu" hoặc "cuda"
    "use_mock": True,                          # True = chạy dữ liệu giả lập
    "num_objects": 3,                          # số đối tượng giả lập
    "allow_register": True,                    # cho phép người dùng tự đăng ký
}


def load_settings(path=SETTINGS_FILE):
    """Đọc cấu hình; thiếu mục nào thì lấy giá trị mặc định."""
    settings = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            settings.update(json.load(f))
    return settings


def save_settings(settings, path=SETTINGS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)