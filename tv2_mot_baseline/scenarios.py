"""
scenarios.py
============
Bộ mô phỏng video tổng hợp có ground-truth để kiểm thử tracking, và hàm chấm ID switch.

Mỗi đối tượng: quỹ đạo tuyến tính từng đoạn (waypoints), kích thước cố định, một vector
appearance (mô phỏng embedding của detector), khoảng frame xuất hiện, và các khoảng bị che
(không có detection). Detection = box + nhiễu, feature = appearance + nhiễu.
Dữ liệu tổng hợp nên chỉ kiểm chứng LOGIC quản lý ID, không phản ánh độ chính xác thực tế.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FEAT_DIM = 32


@dataclass
class SimObject:
    gt_id: int
    waypoints: List[Tuple[int, float, float]]       # (frame, cx, cy)
    start: int
    end: int                                        # frame cuối còn xuất hiện (bao gồm)
    size: Tuple[float, float] = (50.0, 90.0)        # (w, h)
    occluded: Sequence[Tuple[int, int]] = ()        # [a, b] bao gồm hai đầu
    score: float = 0.9

    def visible(self, f: int) -> bool:
        return self.start <= f <= self.end and not any(a <= f <= b for a, b in self.occluded)

    def center(self, f: int) -> Tuple[float, float]:
        fr = [w[0] for w in self.waypoints]
        return (float(np.interp(f, fr, [w[1] for w in self.waypoints])),
                float(np.interp(f, fr, [w[2] for w in self.waypoints])))


@dataclass
class SimFrame:
    boxes: np.ndarray       # (N,4) xyxy
    scores: np.ndarray      # (N,)
    feats: np.ndarray       # (N,C)
    gt_ids: np.ndarray      # (N,)  -1 = false positive


@dataclass
class Scenario:
    name: str
    num_frames: int
    frames: List[SimFrame]
    objects: List[SimObject]
    size: Tuple[int, int] = (640, 360)              # (W, H)


def render(objects: List[SimObject], num_frames: int, name: str, seed: int = 0,
           box_noise: float = 1.0, feat_noise: float = 0.4,
           false_pos: Optional[Dict[int, Tuple[float, float, float]]] = None,
           size: Tuple[int, int] = (640, 360)) -> Scenario:
    """false_pos: {frame: (cx, cy, score)} -- detection nhiễu 1 frame, không có đối tượng thật."""
    rng = np.random.default_rng(seed)
    app = {}
    for o in objects:
        v = rng.standard_normal(FEAT_DIM)
        app[o.gt_id] = v / np.linalg.norm(v)
    frames = []
    for f in range(num_frames):
        bx, sc, ft, gid = [], [], [], []
        for o in objects:
            if not o.visible(f):
                continue
            cx, cy = o.center(f)
            w, h = o.size
            n = rng.normal(0, box_noise, 4)
            bx.append([cx - w / 2 + n[0], cy - h / 2 + n[1], cx + w / 2 + n[2], cy + h / 2 + n[3]])
            sc.append(o.score)
            v = app[o.gt_id] + rng.standard_normal(FEAT_DIM) * feat_noise / np.sqrt(FEAT_DIM)
            ft.append(v / np.linalg.norm(v))
            gid.append(o.gt_id)
        if false_pos and f in false_pos:
            cx, cy, s = false_pos[f]
            bx.append([cx - 25, cy - 45, cx + 25, cy + 45]); sc.append(s)
            v = rng.standard_normal(FEAT_DIM); ft.append(v / np.linalg.norm(v)); gid.append(-1)
        frames.append(SimFrame(np.array(bx, float).reshape(-1, 4), np.array(sc, float),
                               np.array(ft, np.float32).reshape(-1, FEAT_DIM), np.array(gid, int)))
    return Scenario(name, num_frames, frames, objects, size)


def line(x0, y0, x1, y1, f0, f1):
    return [(f0, x0, y0), (f1, x1, y1)]


# --------------------------- các tình huống --------------------------- #
def parallel(seed=0) -> Scenario:
    """3 người đi song song, không tương tác. ID phải ổn định tuyệt đối."""
    objs = [SimObject(i + 1, line(80, 80 + 100 * i, 560, 80 + 100 * i, 0, 60), 0, 60) for i in range(3)]
    return render(objs, 61, "parallel", seed)


def crossing(seed=0) -> Scenario:
    """2 người đi ngang qua nhau (vận tốc không đổi)."""
    objs = [SimObject(1, line(100, 170, 540, 190, 0, 60), 0, 60),
            SimObject(2, line(540, 190, 100, 170, 0, 60), 0, 60)]
    return render(objs, 61, "crossing", seed)


def bounce(seed=0) -> Scenario:
    """2 người đi tới, chồng khít nhau tại frame 30 rồi quay đầu đi ngược lại.
    Kalman hằng vận tốc dự đoán 'tiếp tục đi thẳng' => IoU ghép chéo => dễ ID switch.
    Bộ ghép dựa trên appearance/ID thì không bị."""
    objs = [SimObject(1, [(0, 100, 180), (30, 320, 180), (60, 100, 180)], 0, 60),
            SimObject(2, [(0, 540, 182), (30, 320, 182), (60, 540, 182)], 0, 60)]
    return render(objs, 61, "bounce", seed)


def occlusion(gap: int = 8, seed=0) -> Scenario:
    """Một người bị che `gap` frame (frame 20 .. 19+gap) rồi xuất hiện lại tại đúng chỗ đi tiếp."""
    objs = [SimObject(1, line(80, 120, 560, 120, 0, 70), 0, 70, occluded=[(20, 19 + gap)]),
            SimObject(2, line(80, 260, 560, 260, 0, 70), 0, 70)]
    return render(objs, 71, f"occlusion_{gap}", seed)


def enter_exit(seed=0) -> Scenario:
    """A có mặt cả video, B vào lúc frame 15, C vào lúc 40, B rời đi sau frame 35."""
    objs = [SimObject(1, line(80, 90, 560, 90, 0, 60), 0, 60),
            SimObject(2, line(560, 180, 300, 180, 15, 35), 15, 35),
            SimObject(3, line(80, 270, 400, 270, 40, 60), 40, 60)]
    return render(objs, 61, "enter_exit", seed)


def sequential(n: int = 6, seed=0) -> Scenario:
    """n người lần lượt xuất hiện rồi rời đi, mỗi lúc chỉ 1-2 người -> buộc phải tái sử dụng nhãn
    khi K nhỏ."""
    objs = []
    for i in range(n):
        s = 20 * i
        objs.append(SimObject(i + 1, line(100, 180, 540, 180, s, s + 29), s, s + 29))
    return render(objs, 20 * n + 10, "sequential", seed)


def demo(seed=3) -> Scenario:
    """Kịch bản tổng hợp để quay demo: đi ngược chiều + quay đầu, bị che, vào, ra, tái sử dụng nhãn."""
    objs = [
        SimObject(1, [(0, 90, 140), (45, 320, 150), (90, 90, 140)], 0, 90),                  # A: tới rồi quay về
        SimObject(2, [(0, 560, 152), (45, 320, 150), (90, 560, 152)], 0, 90),                # B: bounce với A @45
        SimObject(3, line(120, 280, 520, 280, 15, 80), 15, 80, occluded=[(40, 49)]),       # C: bị che 10 frame
        SimObject(4, line(520, 60, 300, 60, 25, 55), 25, 55),                               # D: rời lúc 55
        SimObject(5, line(100, 60, 320, 60, 75, 110), 75, 110),                             # E: vào muộn
    ]
    return render(objs, 111, "demo", seed)


ALL = {"parallel": parallel, "crossing": crossing, "bounce": bounce, "occlusion": occlusion,
       "enter_exit": enter_exit, "sequential": sequential, "demo": demo}


# --------------------------- chấm điểm --------------------------- #
@dataclass
class IDMetrics:
    id_switches: int                      # tổng số lần ID của một gt_id đổi
    ids_per_gt: Dict[int, List[int]]      # gt_id -> danh sách ID đã dùng (theo thứ tự xuất hiện)
    fragmented_gts: int                   # số gt có > 1 ID
    shared_ids: List[int]                 # ID bị dùng cho >= 2 gt khác nhau

    @property
    def perfect(self) -> bool:
        return self.id_switches == 0 and not self.shared_ids


def evaluate_ids(assign: List[Dict[int, int]]) -> IDMetrics:
    """assign[f] = {gt_id: id_dự_đoán} cho các đối tượng được xuất ra ở frame f.
    ID switch: lần gt_id nhận ID khác với ID gần nhất nó từng có (bỏ qua frame vắng mặt).
    (Đây là cách đếm đơn giản kiểu MOTChallenge; không dùng matching IoU vì mô phỏng đã biết gt của từng detection.)"""
    last: Dict[int, int] = {}
    seen: Dict[int, List[int]] = {}
    sw = 0
    for frame in assign:
        for g, pid in frame.items():
            if g < 0:
                continue
            if g in last and last[g] != pid:
                sw += 1
            last[g] = pid
            if pid not in seen.setdefault(g, []):
                seen[g].append(pid)
    users: Dict[int, set] = {}
    for g, ids in seen.items():
        for i in ids:
            users.setdefault(i, set()).add(g)
    return IDMetrics(sw, seen, sum(len(v) > 1 for v in seen.values()),
                     sorted(i for i, u in users.items() if len(u) > 1))
