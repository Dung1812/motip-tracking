"""
mot_video_demo.py
=================
Giai đoạn 1 - Lương Thị Mai Phương
Đọc VIDEO -> ma trận ảnh (numpy) -> tìm hộp đối tượng -> tính IoU -> ghép ID -> nghiệm thu 7 điều kiện.

Cần file mot_baseline_demo.py nằm CÙNG THƯ MỤC (file này dùng lại IoUTracker, iou, ... của nó).

Chạy:
    python mot_video_demo.py                 # tạo 4 video mô phỏng -> đọc lại -> nghiệm thu 7/7 -> lưu video kết quả
    python mot_video_demo.py --video a.mp4   # theo dõi trong video thật của bạn (camera đứng yên), lưu out/a_tracked.mp4

Thư viện: numpy, opencv-python (cv2), scipy (không bắt buộc).   pip install numpy opencv-python scipy

Quy trình cho mỗi video
-----------------------
1. cv2.VideoCapture đọc từng frame  -> mỗi frame là MA TRẬN numpy (cao x rộng x 3), giá trị 0-255
2. Ước lượng ảnh nền = trung vị (median) theo thời gian của các frame (đúng khi camera đứng yên)
3. |frame - nền| > ngưỡng  -> mặt nạ nhị phân; cv2.findContours -> hộp bao (x1,y1,x2,y2) của từng vùng
4. Hộp này là "detection"; đưa vào IoUTracker như ở giai đoạn 1 (IoU -> Hungarian -> vòng đời track)
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import mot_baseline_demo as mb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
W, H, FPS = 640, 360, 10
COLORS = [(60, 76, 231), (219, 152, 52), (113, 204, 46), (15, 196, 241)]   # BGR


# =========================================================================== #
# 1. Tạo video mô phỏng từ kịch bản (có ground-truth) và ghi ra file .mp4
# =========================================================================== #
def render_video(path: str, objs: List[mb.Obj], n: int) -> List[List[Tuple[int, np.ndarray]]]:
    """Vẽ các hình chữ nhật chuyển động lên nền xám có nhiễu nhẹ, ghi video. Trả ground-truth theo frame."""
    rng = np.random.default_rng(0)
    bg = np.full((H, W, 3), 200, np.uint8)
    cv2.rectangle(bg, (0, 300), (W, H), (170, 170, 170), -1)          # vạch nền cho giống cảnh thật
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"Không ghi được video: {path}")
    gt = []
    for f in range(n):
        img = bg.copy()
        row = []
        for o in objs:
            b = o.box_at(f)
            if b is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[(o.gt_id - 1) % len(COLORS)], -1)
                row.append((o.gt_id, b))
        noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
        vw.write(np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8))
        gt.append(row)
    vw.release()
    return gt


SCENARIO_OBJS = {
    "simple": ([mb.Obj(i + 1, 60, 70 + 110 * i, 60 + 8 * 24, 70 + 110 * i, 0, 24) for i in range(3)], 25),
    "occlusion": ([mb.Obj(1, 60, 100, 60 + 8 * 24, 100, 0, 24, hidden=(8, 12)),
                   mb.Obj(2, 60, 250, 60 + 8 * 24, 250, 0, 24)], 25),
    "enter_exit": ([mb.Obj(1, 60, 80, 60 + 8 * 24, 80, 0, 24),
                    mb.Obj(2, 300, 230, 300 + 8 * 8, 230, 6, 14)], 25),
}


# =========================================================================== #
# 2. Đọc video -> ma trận ảnh
# =========================================================================== #
def read_video(path: str) -> np.ndarray:
    """Đọc toàn bộ video thành mảng numpy (số frame, cao, rộng, 3)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {path}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"Video không có frame nào: {path}")
    return np.stack(frames)


# =========================================================================== #
# 3. Ma trận ảnh -> hộp detection
# =========================================================================== #
def estimate_background(video: np.ndarray, max_samples: int = 60) -> np.ndarray:
    idx = np.linspace(0, len(video) - 1, min(len(video), max_samples)).astype(int)
    return np.median(video[idx], axis=0).astype(np.uint8)


def detect_boxes(frame: np.ndarray, background: np.ndarray, thr: int = 35, min_area: int = 300) -> np.ndarray:
    """Ảnh -> các hộp (x1,y1,x2,y2). Khác biệt so với nền lớn hơn `thr` ở bất kỳ kênh màu nào = điểm ảnh đối tượng."""
    diff = cv2.absdiff(frame, background).max(axis=2)
    mask = (diff > thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)       # bỏ nhiễu lẻ
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)      # lấp lỗ nhỏ
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cs:
        if cv2.contourArea(c) >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])             # cv2 cho hộp gồm cả điểm biên -> khớp mép hộp vẽ
    boxes.sort(key=lambda b: (b[0], b[1]))
    return np.array(boxes, float).reshape(-1, 4)


def detections_from_video(video: np.ndarray) -> List[np.ndarray]:
    bg = estimate_background(video)
    return [detect_boxes(f, bg) for f in video]


def label_with_gt(dets: List[np.ndarray], gt: List[List[Tuple[int, np.ndarray]]], min_iou: float = 0.5):
    """Gán mỗi detection (từ video) với đối tượng thật có IoU lớn nhất -> frames[f] = [(gt_id, hộp)].
    Dùng IoU để kiểm tra bộ phát hiện bằng ground-truth; chỉ dùng ở khâu nghiệm thu."""
    out, bad = [], 0
    for d, g in zip(dets, gt):
        row = []
        for b in d:
            best = max(((mb.iou(b, gb), gid) for gid, gb in g), default=(0.0, -1))
            if best[0] >= min_iou:
                row.append((best[1], b))
            else:
                bad += 1
        out.append(row)
    return out, bad


# =========================================================================== #
# 4. Nghiệm thu 7 điều kiện trên dữ liệu ĐỌC TỪ VIDEO
# =========================================================================== #
def acceptance_video() -> bool:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("NGHIỆM THU TRÊN VIDEO: tạo video -> đọc lại thành ma trận -> tìm hộp -> IoU -> ghép ID")
    print("=" * 78)
    frames_by_sc, det_ok = {}, True
    for name, (objs, n) in SCENARIO_OBJS.items():
        path = os.path.join(OUT_DIR, f"video_{name}.mp4")
        gt = render_video(path, objs, n)
        vid = read_video(path)
        dets = detections_from_video(vid)
        frames, bad = label_with_gt(dets, gt)
        n_det, n_gt = sum(len(d) for d in dets), sum(len(g) for g in gt)
        match_gt = sum(len(r) for r in frames)
        ok = (n_det == n_gt and match_gt == n_gt and bad == 0)
        det_ok &= ok
        print(f"  video_{name}.mp4: ma trận {vid.shape} (frame,cao,rộng,kênh) | hộp tìm được {n_det}/{n_gt} "
              f"| khớp ground-truth (IoU>=0.5) {match_gt}/{n_gt}  [{'ĐẠT' if ok else 'HỎNG'}]")
        frames_by_sc[name] = frames
    print()

    results = [det_ok]

    def check(label: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"  [{'ĐẠT' if ok else 'HỎNG'}] {label}" + (f"   {detail}" if detail else ""))

    def seeds(name, **kw):
        out = []
        for sd in mb.SEEDS:
            trk, pf = mb.run(frames_by_sc[name], seed=sd, **kw)
            out.append((trk, pf, mb.id_report(pf)))
        return out

    print("  [ĐẠT/HỎNG] Điều kiện 0: bộ phát hiện bằng video tìm đủ và đúng vị trí mọi đối tượng (xem trên)")
    R = seeds("simple")
    check("Ca đơn giản: 3 người song song giữ nguyên ID",
          all(r["id_switches"] == 0 and len(r["ids_per_gt"]) == 3 for _, _, r in R),
          f"ID switch tối đa = {max(r['id_switches'] for _, _, r in R)}")
    Rg = seeds("simple", matcher=mb.match_greedy)
    check("Ghép tham lam cũng đúng ở ca đơn giản", all(r["id_switches"] == 0 for _, _, r in Rg))
    Rv = seeds("occlusion", use_velocity=True)
    Rn = seeds("occlusion", use_velocity=False)
    check("Bị che 5 frame + ngoại suy vận tốc: GIỮ ID", all(len(r["ids_per_gt"][1]) == 1 for _, _, r in Rv),
          f"ID người bị che = {Rv[0][2]['ids_per_gt'][1]}")
    check("Bị che 5 frame, không dùng chuyển động: ĐỔI ID (giới hạn của IoU thuần)",
          all(len(r["ids_per_gt"][1]) > 1 for _, _, r in Rn), f"ID người bị che = {Rn[0][2]['ids_per_gt'][1]}")
    Re = seeds("enter_exit", max_age=5)
    check("Người vào muộn nhận ID mới, khác ID của người đang có",
          all(r["ids_per_gt"][2][0] not in r["ids_per_gt"][1] and r["id_switches"] == 0 for _, _, r in Re))
    ends = [[e for e in trk.events if e.kind == "end" and e.track_id == r["ids_per_gt"][2][0]] for trk, _, r in Re]
    check("Người rời đi: track kết thúc sau max_age frame mất dấu", all(len(e) == 1 for e in ends),
          f"{ends[0][0] if ends[0] else 'không có sự kiện end'}")
    dup = any(len(set(fr.values())) != len(fr) for grp in (R, Rg, Rv, Rn, Re) for _, pf, _ in grp for fr in pf)
    check("Không có hai đối tượng trùng ID trong cùng một frame", not dup)

    ok = all(results)
    print(f"\n  => {'TẤT CẢ ĐẠT' if ok else 'CÓ ĐIỀU KIỆN HỎNG'} ({sum(results)}/{len(results)})  "
          f"(điều kiện 0 là bước đọc video; 7 điều kiện còn lại giống file mot_baseline_demo.py)")

    # ảnh minh họa: frame thật từ video + hộp tìm được + ID
    try:
        vid = read_video(os.path.join(OUT_DIR, "video_occlusion.mp4"))
        trk = mb.IoUTracker(use_velocity=True)
        dets = detections_from_video(vid)
        shown = {}
        for f, d in enumerate(dets):
            outs = trk.step(f, d)
            if f in (0, 7, 10, 14):
                shown[f] = draw(vid[f].copy(), outs, trk, f, True, f"frame {f}")
        cv2.imwrite(os.path.join(OUT_DIR, "video_occlusion_frames.png"),
                    np.vstack([np.hstack([shown[0], shown[7]]), np.hstack([shown[10], shown[14]])]))
        print(f"  Đã lưu ảnh: {os.path.join(OUT_DIR, 'video_occlusion_frames.png')}")
    except Exception as e:                                   # ảnh chỉ là phụ
        print(f"  (bỏ qua ảnh minh họa: {e})")
    return ok


# =========================================================================== #
# 5. Vẽ kết quả & chạy trên video bất kỳ
# =========================================================================== #
def draw(img, outs, trk, f, vel, title):
    pal = [(60, 76, 231), (219, 152, 52), (113, 204, 46), (15, 196, 241), (182, 89, 155), (34, 126, 230)]
    for tid, b, _, state in outs:
        c = pal[(tid - 1) % len(pal)]
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(img, f"ID {tid}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    for t in trk.tracks:
        if t.state == mb.TrackState.LOST:
            p = t.predict(f, vel).astype(int)
            cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (90, 90, 90), 1)
            cv2.putText(img, f"ID {t.track_id} LOST", (p[0], p[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1)
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def track_video(path: str, use_velocity: bool = True) -> None:
    """Theo dõi trong video bất kỳ (camera đứng yên, đối tượng khác màu nền)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    vid = read_video(path)
    print(f"Đã đọc {path}: ma trận {vid.shape} = (số frame, cao, rộng, kênh màu)")
    dets = detections_from_video(vid)
    trk = mb.IoUTracker(use_velocity=use_velocity)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    out_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(path))[0] + "_tracked.mp4")
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid.shape[2], vid.shape[1]))
    for f, d in enumerate(dets):
        outs = trk.step(f, d)
        if f < 3 and len(d) > 1:
            M = mb.iou_matrix(np.array([t.box for t in trk.tracks]).reshape(-1, 4), d)
            print(f"frame {f}: ma trận IoU (track x detection)\n{np.round(M, 2)}")
        vw.write(draw(vid[f].copy(), outs, trk, f, use_velocity, f"frame {f}"))
    vw.release()
    print(f"Số track đã tạo: {trk._next_id - 1}. Đã lưu video kết quả: {out_path}")
    for e in trk.events[:30]:
        print("  ", e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="đường dẫn video thật cần theo dõi (không có thì chạy nghiệm thu)")
    args = ap.parse_args()
    if args.video:
        track_video(args.video)
        return 0
    return 0 if acceptance_video() else 1


if __name__ == "__main__":
    sys.exit(main())
