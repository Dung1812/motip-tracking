# Chú ý: Đường dẫn có thể sai

import json
import os
import random
import shutil

def create_single_subset(base_dir="Dataset/coco_dataset", seed=42):
    # 1. Nhập tên tập cần lấy
    split_name = input("Nhập tên tập cần lấy (ví dụ: train, val, test): ").strip()
    src_dir = os.path.join(base_dir, split_name)
    src_json = os.path.join(src_dir, "labels.json")
    src_img_dir = os.path.join(src_dir, "data")

    if not os.path.exists(src_json):
        print(f"Lỗi: Không tìm thấy file nhãn tại '{src_json}'")
        return

    # 2. Đọc dữ liệu gốc
    with open(src_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    total_imgs = len(coco["images"])
    print(f"Tìm thấy {total_imgs} ảnh trong tập '{split_name}'.")

    # 3. Nhập số lượng ảnh muốn lấy
    try:
        num_samples = int(input(f"Nhập số lượng ảnh muốn lấy ngẫu nhiên (1 - {total_imgs}): ").strip())
        if num_samples <= 0 or num_samples > total_imgs:
            print("Lỗi: Số lượng không hợp lệ.")
            return
    except ValueError:
        print("Lỗi: Vui lòng nhập một số nguyên.")
        return

    # 4. Tự động tạo thư mục con chứa subset
    out_dir = os.path.join(base_dir, f"{split_name}_subset_{num_samples}")
    out_img_dir = os.path.join(out_dir, "data")
    out_json = os.path.join(out_dir, "labels.json")
    os.makedirs(out_img_dir, exist_ok=True)

    # 5. Lấy ngẫu nhiên ảnh và lọc annotations tương ứng
    random.seed(seed)
    selected_images = random.sample(coco["images"], num_samples)
    selected_ids = {img["id"] for img in selected_images}
    selected_anns = [ann for ann in coco["annotations"] if ann["image_id"] in selected_ids]

    subset_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": selected_images,
        "annotations": selected_anns,
        "categories": coco["categories"]
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(subset_coco, f)

    # 6. Copy ảnh sang thư mục con
    copied = 0
    for img in selected_images:
        src = os.path.join(src_img_dir, img["file_name"])
        dst = os.path.join(out_img_dir, img["file_name"])
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1

    # 7. Thống kê nhanh
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    counts = {name: 0 for name in cat_map.values()}
    for ann in selected_anns:
        name = cat_map.get(ann["category_id"])
        if name in counts:
            counts[name] += 1

    print(f"\nHoàn tất! Đã tạo subset tại: {out_dir}")
    print(f"- Số ảnh đã lấy : {copied}")
    print(f"- Tổng số box    : {len(selected_anns)}")
    print(f"- Chi tiết lớp   : {counts}")


if __name__ == "__main__":
    create_single_subset()
