# Chú ý đường dẫn có thể sai

import json
import os
import shutil
from collections import defaultdict

# Cấu hình ánh xạ ID: COCO gốc -> YOLO (0-based)
# 0: person | 1: vehicle (gộp bicycle, car, motorcycle, bus, truck)
ID_MAPPING = {
    1: 0,
    2: 1,
    3: 1,
    4: 1,
    6: 1,
    8: 1,
}

CLASS_NAMES = {
    0: "person",
    1: "vehicle",
}


def convert_split(coco_split_dir, yolo_root_dir, split_name, skip_crowd=True):
    json_path = os.path.join(coco_split_dir, "labels.json")
    src_img_dir = os.path.join(coco_split_dir, "data")

    if not os.path.exists(json_path):
        print(f"[BỎ QUA] Không tìm thấy: {json_path}")
        return

    out_img_dir = os.path.join(yolo_root_dir, "images", split_name)
    out_lbl_dir = os.path.join(yolo_root_dir, "labels", split_name)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    print(f"\n--- Đang chuyển đổi tập {split_name.upper()} ---")
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Gom annotations theo image_id để tra cứu nhanh O(1)
    anns_by_img = defaultdict(list)
    skipped_crowd = 0
    for ann in coco["annotations"]:
        if skip_crowd and ann.get("iscrowd", 0) == 1:
            skipped_crowd += 1
            continue
        cid = ann["category_id"]
        if cid in ID_MAPPING:
            anns_by_img[ann["image_id"]].append(ann)

    total_imgs = 0
    total_boxes = 0
    class_counts = defaultdict(int)

    for img in coco["images"]:
        img_id = img["id"]
        fname = img["file_name"]
        img_w = img["width"]
        img_h = img["height"]

        src_img_path = os.path.join(src_img_dir, fname)
        if not os.path.exists(src_img_path):
            continue

        # 1. Liên kết ảnh (hardlink không tốn dung lượng ổ cứng)
        dst_img_path = os.path.join(out_img_dir, fname)
        if not os.path.exists(dst_img_path):
            try:
                os.link(src_img_path, dst_img_path)
            except OSError:
                shutil.copy2(src_img_path, dst_img_path)

        # 2. Ghi file nhãn .txt trùng tên ảnh
        txt_name = os.path.splitext(fname)[0] + ".txt"
        txt_path = os.path.join(out_lbl_dir, txt_name)

        lines = []
        for ann in anns_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
                continue

            # Chuẩn hóa về [0, 1] và giới hạn tránh tràn biên
            x_center = max(0.0, min(1.0, (x + w / 2.0) / img_w))
            y_center = max(0.0, min(1.0, (y + h / 2.0) / img_h))
            w_norm = max(0.0, min(1.0, w / img_w))
            h_norm = max(0.0, min(1.0, h / img_h))

            yolo_cid = ID_MAPPING[ann["category_id"]]
            lines.append(f"{yolo_cid} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            class_counts[yolo_cid] += 1
            total_boxes += 1

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        total_imgs += 1

    print(f"HOÀN TẤT {split_name.upper()}: {total_imgs} ảnh | {total_boxes} boxes (đã bỏ {skipped_crowd} box iscrowd)")
    for cid, name in CLASS_NAMES.items():
        print(f"  Class {cid} ({name}): {class_counts[cid]} boxes")


def create_data_yaml(yolo_root_dir):
    yaml_path = os.path.join(yolo_root_dir, "data.yaml")
    abs_path = os.path.abspath(yolo_root_dir).replace("\\", "/")

    lines = [
        f"path: {abs_path}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for cid, name in CLASS_NAMES.items():
        lines.append(f"  {cid}: {name}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nĐã tạo file cấu hình: {yaml_path}")


if __name__ == "__main__":
    coco_root = "Dataset/coco_dataset"
    yolo_root = "Dataset/yolo_dataset"

    convert_split(os.path.join(coco_root, "train"), yolo_root, "train", skip_crowd=True)
    convert_split(os.path.join(coco_root, "val"), yolo_root, "val", skip_crowd=True)
    create_data_yaml(yolo_root)
