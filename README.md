# MOTIP Tracking - Giai đoạn 1: Nghiên cứu

Dự án nghiên cứu theo dõi nhiều đối tượng (Multi-Object Tracking)
dựa trên bài báo MOTIP (CVPR 2025).

## Thành viên và thư mục
| Mã | Thành viên | Thư mục |
|---|---|---|
| G1-TV1 | Đặng Quang Vũ | `tv1_detr/` |
| G1-TV2 | Lương Thị Mai Phương | `tv2_mot_baseline/` |
| G1-TV3 | Đào Trung Dũng | `tv3_id_embedding/` |
| G1-TV4 | Nguyễn Phú Dương | `tv4_dataset_eval/` |
| G1-TV5 | Phạm Nguyễn Hữu Dũng | Git, môi trường, giao diện |

## Cài đặt môi trường
Yêu cầu: Python 3.13
```bash

git clone https://github.com/Dung1812/motip-tracking.git
cd motip-tracking
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy demo
```bash
python ui_skeleton.py
```

## Quy ước làm việc
- Mỗi người làm trên nhánh `feature/...` riêng.
- Chỉ sửa file trong thư mục của mình.
- Không đẩy video và file model nặng lên repo.

