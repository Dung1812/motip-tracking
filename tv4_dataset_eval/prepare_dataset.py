#Tạo subnet, kiểm tra nhãn

import os
import shutil
import argparse
import pandas as pd

def prepare_subset(src_root, dst_root, split, num_seqs):
    """Trích xuất subset từ dataset gốc và tạo lại file seqmap."""
    src_split_dir = os.path.join(src_root, split)
    dst_split_dir = os.path.join(dst_root, split)

    if not os.path.exists(src_split_dir):
        print(f"[Lỗi] Không tìm thấy thư mục gốc: {src_split_dir}")
        return []

    os.makedirs(dst_split_dir, exist_ok=True)
    all_seqs = sorted([d for d in os.listdir(src_split_dir) if os.path.isdir(os.path.join(src_split_dir, d))])

    if len(all_seqs) == 0:
        print(f"[Lỗi] Không có sequence nào trong {src_split_dir}")
        return []

    subset_seqs = all_seqs[:num_seqs]
    print(f"[*] Đang tạo subset với {len(subset_seqs)} sequences: {subset_seqs}")

    for seq in subset_seqs:
        src_seq_dir = os.path.join(src_split_dir, seq)
        dst_seq_dir = os.path.join(dst_split_dir, seq)

        if os.path.exists(dst_seq_dir):
            shutil.rmtree(dst_seq_dir)

        print(f"    -> Đang sao chép {seq}...")
        shutil.copytree(src_seq_dir, dst_seq_dir)

    seqmap_path = os.path.join(dst_root, f"{split}_seqmap.txt")
    with open(seqmap_path, "w") as f:
        f.write("name\n")
        for seq in subset_seqs:
            f.write(f"{seq}\n")

    print(f"[*] Đã lưu seqmap mới tại: {seqmap_path}")
    return subset_seqs

def verify_labels_advanced(dst_root, split, seq_names):
    """Kiểm tra nhãn chuyên sâu cho bài toán MOTIP (Context-based)."""
    print(f"\n[*] ĐANG KIỂM TRA NHÃN CHUYÊN SÂU TẬP SUBSET...")
    cols = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']

    for seq in seq_names:
        gt_path = os.path.join(dst_root, split, seq, 'gt', 'gt.txt')
        if not os.path.exists(gt_path):
            print(f"    [Lỗi] Không tìm thấy file nhãn: {gt_path}")
            continue

        try:
            df = pd.read_csv(gt_path, names=cols)
            issues_found = False

            dupes = df[df.duplicated(subset=['frame', 'id'], keep=False)]
            if not dupes.empty:
                print(f"    [CẢNH BÁO CAO] {seq}: Phát hiện {len(dupes)} dòng có ID bị trùng trong cùng 1 frame!")
                issues_found = True

            invalid_boxes = df[(df['w'] <= 0) | (df['h'] <= 0) | (df['x'] < 0) | (df['y'] < 0)]
            if not invalid_boxes.empty:
                print(f"    [Cảnh báo] {seq}: Có {len(invalid_boxes)} bounding box bị âm hoặc x/y tràn viền.")
                issues_found = True

            track_lengths = df.groupby('id')['frame'].count()
            short_tracks = track_lengths[track_lengths < 3]
            if not short_tracks.empty:
                print(f"    [Chú ý] {seq}: Có {len(short_tracks)} track IDs quá ngắn (< 3 frames), có thể gây nhiễu context của MOTIP.")

            frames = df['frame'].sort_values().unique()
            expected_frames = list(range(frames.min(), frames.max() + 1))
            missing_frames = set(expected_frames) - set(frames)
            if missing_frames:
                print(f"    [Cảnh báo] {seq}: Bị đứt gãy {len(missing_frames)} frames ở giữa chuỗi (Ví dụ: thiếu {list(missing_frames)[:3]}).")
                issues_found = True

            if not issues_found:
                print(f"    [OK] Sequence {seq}: Nhãn sạch sẽ. (Frames: {len(frames)} | IDs: {df['id'].nunique()})")

        except Exception as e:
            print(f"    [Lỗi hệ thống] Không thể phân tích {gt_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tạo Subset và kiểm tra nhãn cho bài toán MOTIP")
    parser.add_argument("--src", type=str, required=True, help="Đường dẫn đến dataset gốc đã format (vd: ./datasets/BFT)")
    parser.add_argument("--dst", type=str, required=True, help="Đường dẫn lưu subset (vd: ./datasets/BFT_subset)")
    parser.add_argument("--split", type=str, default="train", help="Tập dữ liệu (train, val, test)")
    parser.add_argument("--num_seqs", type=int, default=2, help="Số lượng sequence muốn cắt làm subset")

    args = parser.parse_args()

    subset_list = prepare_subset(args.src, args.dst, args.split, args.num_seqs)

    if subset_list:
        verify_labels_advanced(args.dst, args.split, subset_list)
        print("\n[*] HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH.")
