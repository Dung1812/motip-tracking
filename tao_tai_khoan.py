"""Tạo tài khoản từ terminal (dùng để tạo tài khoản admin đầu tiên).

Cách dùng:
    python tao_tai_khoan.py admin --role admin
    python tao_tai_khoan.py nguoidung1
Chương trình sẽ hỏi mật khẩu (gõ không hiện chữ là bình thường).
"""

import argparse
import getpass

from auth import ROLES, add_user

parser = argparse.ArgumentParser(description="Tạo hoặc đổi mật khẩu tài khoản")
parser.add_argument("username", help="Tên đăng nhập")
parser.add_argument("--role", choices=ROLES, default="user", help="user hoặc admin")
args = parser.parse_args()

password = getpass.getpass("Mật khẩu: ")
again = getpass.getpass("Nhập lại mật khẩu: ")
if password != again:
    print("Hai lần nhập mật khẩu không khớp.")
else:
    try:
        add_user(args.username, password, args.role)
        print(f"Đã lưu tài khoản '{args.username}' với vai trò {args.role}.")
    except ValueError as e:
        print(f"Lỗi: {e}")