"""Quản lý tài khoản đăng nhập: lưu, kiểm tra mật khẩu, thêm và xoá tài khoản.

Mật khẩu KHÔNG lưu dạng chữ thường. Mỗi tài khoản lưu một chuỗi "muối" (salt) ngẫu nhiên
và kết quả băm PBKDF2-SHA256 của mật khẩu, nên đọc file users.json cũng không biết mật khẩu.
"""

import hashlib
import hmac
import json
import os
import re
import secrets

USERS_FILE = "users.json"
ROLES = ("user", "admin")
_ITERATIONS = 200_000          # số vòng băm; càng nhiều càng khó dò mật khẩu


def hash_password(password, salt=None):
    """Băm mật khẩu. Trả về (salt, hash) dạng chuỗi hex."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(salt), _ITERATIONS)
    return salt, digest.hex()


def load_users(path=USERS_FILE):
    """Đọc danh sách tài khoản. Chưa có file thì trả về dict rỗng."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users, path=USERS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def add_user(username, password, role="user", path=USERS_FILE):
    """Thêm hoặc đổi mật khẩu một tài khoản. Báo lỗi nếu dữ liệu không hợp lệ."""
    username = username.strip()
    if not username:
        raise ValueError("Tên đăng nhập không được để trống")
    if len(password) < 6:
        raise ValueError("Mật khẩu phải có ít nhất 6 ký tự")
    if role not in ROLES:
        raise ValueError(f"Vai trò phải là một trong {ROLES}")
    users = load_users(path)
    salt, digest = hash_password(password)
    users[username] = {"role": role, "salt": salt, "hash": digest}
    save_users(users, path)


def register_user(username, password, password_again, path=USERS_FILE):
    """Người dùng tự đăng ký. Luôn tạo vai trò 'user', không cho trùng tên đã có."""
    username = username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,30}", username):
        raise ValueError("Tên đăng nhập dài 3-30 ký tự, chỉ gồm chữ không dấu, số, dấu _ . -")
    if username.lower() in (name.lower() for name in load_users(path)):
        raise ValueError("Tên đăng nhập này đã có người dùng, hãy chọn tên khác")
    if password != password_again:
        raise ValueError("Hai lần nhập mật khẩu không khớp")
    add_user(username, password, "user", path)


def delete_user(username, path=USERS_FILE):
    users = load_users(path)
    users.pop(username, None)
    save_users(users, path)


def verify_login(username, password, path=USERS_FILE):
    """Đúng tên và mật khẩu thì trả về vai trò ('user' / 'admin'), sai thì trả về None."""
    user = load_users(path).get(username.strip())
    if user is None:
        return None
    _, digest = hash_password(password, user["salt"])
    # compare_digest so sánh an toàn, không để lộ thông tin qua thời gian so sánh
    if hmac.compare_digest(digest, user["hash"]):
        return user["role"]
    return None