"""
make_diagrams.py
================
Vẽ hai sơ đồ cho giai đoạn 1 (cần: pip install matplotlib):
    out/methods_relationship.png (+ .svg)   quan hệ giữa các phương pháp MOT
    out/track_lifecycle_baseline.png (+ .svg)  vòng đời track của phương pháp truyền thống

Chạy:  python make_diagrams.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
plt.rcParams["font.family"] = "DejaVu Sans"
INK = "#2c3e50"
C = dict(task="#d6eaf8", block="#fdebd0", sort="#abebc6", deep="#a9cce3", byte="#f9e79f",
         motip="#d7bde2", manage="#eaecee", newborn="#f4d03f", active="#58d68d", lost="#f5b041", ended="#ec7063")


def box(ax, x, y, w, h, title, sub="", color="#fff", fs=12, subfs=8.8, bold=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=color, ec=INK, lw=1.8))
    ax.text(x, y + (h * 0.18 if sub else 0), title, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK)
    if sub:
        ax.text(x, y - h * 0.2, sub, ha="center", va="center", fontsize=subfs, color=INK, linespacing=1.35)


def arrow(ax, p, q, text="", rad=0.0, ls="-", color=INK, tx=None, ty=None, fs=8.8, lw=1.7, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=lw, ls=ls, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        tx = (p[0] + q[0]) / 2 if tx is None else tx
        ty = (p[1] + q[1]) / 2 if ty is None else ty
        ax.text(tx, ty, text, ha=ha, va="center", fontsize=fs, color=color, linespacing=1.3,
                bbox=dict(fc="white", ec="none", alpha=0.92, pad=1.5))


def methods_relationship():
    fig, ax = plt.subplots(figsize=(17, 11))
    ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")
    ax.text(50, 64, "Quan hệ giữa các phương pháp theo dõi đa đối tượng (MOT)", ha="center", fontsize=18,
            fontweight="bold", color=INK)

    # --- hàng 1: bài toán và khung tracking-by-detection
    box(ax, 50, 57, 30, 5, "Bài toán MOT", "gán ID nhất quán cho nhiều đối tượng theo thời gian", C["task"], fs=13)
    box(ax, 50, 47.5, 56, 6.2, "Tracking-by-Detection",
        "① Detector cho hộp mỗi frame   →   ② Data association (ghép với track)   →   ③ Quản lý track",
        C["task"], fs=13)
    arrow(ax, (50, 54.4), (50, 50.7))

    # --- cột trái: thành phần cổ điển
    ax.text(10, 43.2, "Thành phần cổ điển", ha="center", fontsize=11, fontweight="bold", color=INK)
    box(ax, 10, 38.5, 16, 5.2, "Kalman Filter", "dự đoán chuyển động", C["block"])
    box(ax, 10, 31.5, 16, 5.2, "IoU", "độ chồng lấp hộp", C["block"])
    box(ax, 10, 24.5, 16, 5.2, "Hungarian", "gán một-một tối ưu", C["block"])

    # --- SORT (gốc)
    box(ax, 50, 33, 24, 8.5, "SORT (2016)", "Kalman + IoU + Hungarian\nnhanh, chỉ dùng chuyển động", C["sort"], fs=14)
    arrow(ax, (18.2, 38.5), (37.8, 35.5), "dự đoán hộp", rad=0.0, tx=27, ty=38.5)
    arrow(ax, (18.2, 31.5), (37.8, 33), "", rad=0.0)
    arrow(ax, (18.2, 24.5), (37.8, 30.5), "ghép", rad=0.0, tx=27, ty=26.6)
    arrow(ax, (50, 44.2), (50, 37.4), "cài đặt bước ② kiểu\ntruyền thống", tx=50, ty=41.0)

    # --- nhánh mở rộng
    box(ax, 20, 12, 26, 9.5, "DeepSORT (2017)", "thêm embedding ngoại hình (CNN re-ID)\n+ Mahalanobis + matching cascade", C["deep"], fs=13)
    box(ax, 50, 12, 26, 9.5, "ByteTrack (2021)", "giữ cả detection điểm thấp\nghép hai vòng (cao rồi thấp)", C["byte"], fs=13)
    box(ax, 81, 12, 30, 9.5, "MOTIP (CVPR 2025)", "thay bước ② bằng dự đoán ID:\nTransformer đọc lịch sử tracklet,\nkhông dùng Hungarian", C["motip"], fs=13)
    arrow(ax, (43, 28.6), (26, 16.9), "+ ngoại hình", rad=0.0, tx=32.5, ty=22.5, color="#1f618d")
    arrow(ax, (50, 28.6), (50, 16.9), "+ detection điểm thấp", tx=50, ty=22.5, color="#9a7d0a")
    arrow(ax, (60.5, 29.5), (76, 16.9), "thay thế cách ghép\n(học từ dữ liệu)", rad=0.0, tx=71, ty=24.5, color="#6c3483", ls="--")

    # --- Tracklet
    box(ax, 86, 33, 22, 7.2, "Tracklet", "lịch sử T frame của một track\n(feature + ID) làm ngữ cảnh", C["block"])
    arrow(ax, (86, 29.4), (84.5, 16.9), "đầu vào của\nID Decoder", tx=91.5, ty=23.2, color="#6c3483")

    # --- dải quản lý track
    ax.add_patch(FancyBboxPatch((3, 1.2), 94, 4.4, boxstyle="round,pad=0.3,rounding_size=1",
                                fc=C["manage"], ec=INK, lw=1.4))
    ax.text(50, 3.4, "Bước ③ Quản lý track dùng chung cho mọi phương pháp:   NEWBORN → ACTIVE ⇄ LOST → ENDED     "
                     "(ID nội bộ, ID xuất ra video, tái sử dụng nhãn)", ha="center", va="center", fontsize=10.5, color=INK)
    for x in (20, 50, 81):
        arrow(ax, (x, 7.1), (x, 5.8), "", lw=1.2)

    fig.savefig(os.path.join(OUT, "methods_relationship.png"), dpi=130, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, "methods_relationship.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def lifecycle_baseline():
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 54); ax.axis("off")
    ax.text(50, 52, "Vòng đời track trong theo dõi truyền thống (ghép IoU + Hungarian)", ha="center",
            fontsize=17, fontweight="bold", color=INK)

    box(ax, 14, 30, 17, 9, "NEWBORN", "mới sinh, chưa tin cậy\n(SORT/DeepSORT: tentative)", C["newborn"], fs=13)
    box(ax, 44, 30, 17, 9, "ACTIVE", "đang quan sát\n(DeepSORT: confirmed)", C["active"], fs=13)
    box(ax, 78, 30, 17, 9, "LOST", "mất dấu tạm thời,\nvẫn chờ khớp lại", C["lost"], fs=13)
    box(ax, 61, 9, 20, 8, "ENDED", "bị xoá khỏi danh sách\n(DeepSORT: deleted)", C["ended"], fs=13)

    ax.text(3, 43.5, "detection không khớp\ntrack nào → sinh track mới", ha="center", fontsize=9.5, color=INK)
    arrow(ax, (3, 40), (7.5, 34.2))
    arrow(ax, (22.7, 30), (35.3, 30), "khớp đủ n_init lần\n(vd: 2 frame liên tiếp)", tx=29, ty=34.2)
    arrow(ax, (52.7, 32), (69.3, 32), "không khớp detection", tx=61, ty=35.3)
    arrow(ax, (69.3, 28), (52.7, 28), "khớp lại (recover)", tx=61, ty=24.8, color="#1e8449")
    ax.add_patch(FancyArrowPatch((39, 35.1), (49, 35.1), arrowstyle="-|>", mutation_scale=16, lw=1.7,
                                 color=INK, connectionstyle="arc3,rad=-1.7"))
    ax.text(44, 46.2, "khớp detection →\ncập nhật hộp", ha="center", fontsize=9.5, color=INK)
    ax.add_patch(FancyArrowPatch((73, 35.1), (83, 35.1), arrowstyle="-|>", mutation_scale=16, lw=1.7,
                                 color=INK, connectionstyle="arc3,rad=-1.7"))
    ax.text(78, 46.2, "vẫn không khớp,\nmisses ≤ max_age", ha="center", fontsize=9.5, color=INK)
    arrow(ax, (80, 25.4), (70, 12.5), "misses > max_age\n(mất dấu quá lâu)", tx=79.5, ty=17.5, color="#922b21")
    arrow(ax, (11, 25.4), (51, 10.5), "", color="#922b21", rad=-0.04)
    ax.text(27, 17.2, "mất khi chưa xác nhận\n(coi là detection nhiễu)", ha="center", fontsize=9.5, color="#922b21",
            bbox=dict(fc="white", ec="none", alpha=0.92, pad=1.5))

    ax.text(50, 2.2, "Mỗi frame: dự đoán hộp của mọi track → ma trận IoU → Hungarian (bỏ cặp IoU < ngưỡng) "
                     "→ cập nhật trạng thái các track khớp / không khớp → sinh track mới từ detection còn lại",
            ha="center", fontsize=10, color=INK)
    fig.savefig(os.path.join(OUT, "track_lifecycle_baseline.png"), dpi=130, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, "track_lifecycle_baseline.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    methods_relationship()
    lifecycle_baseline()
    print("Đã lưu:", os.path.join(OUT, "methods_relationship.png"), "và", os.path.join(OUT, "track_lifecycle_baseline.png"))
