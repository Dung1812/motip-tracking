"""
mot_baseline_demo.py
====================
Giai đoạn 1 (Tuần 1-3) - Lương Thị Mai Phương
Ví dụ theo dõi đa đối tượng bằng phương pháp TRUYỀN THỐNG (tracking-by-detection):
    ghép IoU + thuật toán Hungarian + mô hình vòng đời track.

File này ĐỘC LẬP (không import file nào khác của dự án). Chạy trong Visual Studio / VS Code:
    python mot_baseline_demo.py              # chạy đủ: giải thích 1 frame, 4 kịch bản, kiểm thử, lưu ảnh
    python mot_baseline_demo.py --no-plot    # không vẽ ảnh (không cần matplotlib)
    python mot_baseline_demo.py --show       # mở cửa sổ ảnh thay vì chỉ lưu

Thư viện: numpy (bắt buộc); scipy (Hungarian, nếu thiếu sẽ tự chuyển sang ghép tham lam);
          matplotlib (chỉ để vẽ ảnh).

Nội dung chính
--------------
1. iou()/iou_matrix()      : độ đo IoU giữa hai hộp / giữa hai tập hộp
2. match_hungarian()/match_greedy() : ghép detection <-> track từ ma trận IoU
3. TrackState + Track      : vòng đời  NEWBORN -> ACTIVE <-> LOST -> ENDED
4. IoUTracker              : bộ theo dõi; cờ use_velocity bật/tắt dự đoán chuyển động đơn giản
5. 4 kịch bản mô phỏng + kiểm thử nghiệm thu "ghép ID đúng trong ca đơn giản"
"""
from __future__ import annotations

import argparse
import enum
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:                      # vẫn chạy được, chỉ là ghép tham lam
    linear_sum_assignment = None

_WARNED = False
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


# =========================================================================== #
# 1. IoU
# =========================================================================== #
def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU giữa hai hộp xyxy = diện tích giao / diện tích hợp, trong [0, 1]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(tracks_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    """Ma trận IoU (số track, số detection)."""
    M = np.zeros((len(tracks_boxes), len(det_boxes)))
    for i, tb in enumerate(tracks_boxes):
        for j, db in enumerate(det_boxes):
            M[i, j] = iou(tb, db)
    return M


# =========================================================================== #
# 2. Ghép từ ma trận IoU
# =========================================================================== #
Matches = Tuple[List[Tuple[int, int]], List[int], List[int]]   # (cặp khớp, track chưa khớp, det chưa khớp)


def _split(matches: List[Tuple[int, int]], n_t: int, n_d: int) -> Matches:
    mt, md = {i for i, _ in matches}, {j for _, j in matches}
    return matches, [i for i in range(n_t) if i not in mt], [j for j in range(n_d) if j not in md]


def match_hungarian(M: np.ndarray, thr: float) -> Matches:
    """Hungarian: gán một-một sao cho TỔNG IoU lớn nhất (tối ưu toàn cục), rồi bỏ cặp IoU < thr."""
    n_t, n_d = M.shape
    if M.size == 0:
        return _split([], n_t, n_d)
    if linear_sum_assignment is None:
        global _WARNED
        if not _WARNED:
            print("[cảnh báo] chưa cài scipy -> dùng ghép tham lam thay cho Hungarian", file=sys.stderr)
            _WARNED = True
        return match_greedy(M, thr)
    rows, cols = linear_sum_assignment(-M)               # cực đại IoU = cực tiểu (-IoU)
    return _split([(int(i), int(j)) for i, j in zip(rows, cols) if M[i, j] >= thr], n_t, n_d)


def match_greedy(M: np.ndarray, thr: float) -> Matches:
    """Tham lam: lấy cặp IoU lớn nhất, loại hàng/cột đó, lặp lại. Đơn giản nhưng không tối ưu toàn cục."""
    n_t, n_d = M.shape
    pairs, used_t, used_d = [], set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0] if M.size else []
    for i, j in order:
        if M[i, j] < thr:
            break
        if i in used_t or j in used_d:
            continue
        pairs.append((int(i), int(j))); used_t.add(i); used_d.add(j)
    return _split(pairs, n_t, n_d)


# =========================================================================== #
# 3. Vòng đời track
# =========================================================================== #
class TrackState(enum.Enum):
    NEWBORN = "newborn"   # vừa sinh từ detection chưa khớp; chưa tin cậy (gọi là 'tentative' trong SORT/DeepSORT)
    ACTIVE = "active"     # đã xác nhận, đang được quan sát (DeepSORT: 'confirmed')
    LOST = "lost"         # mất dấu tạm thời (bị che / detector bỏ sót), còn chờ khớp lại
    ENDED = "ended"       # kết thúc, bị xoá khỏi danh sách (DeepSORT: 'deleted')


@dataclass
class Track:
    track_id: int
    box: np.ndarray                       # hộp quan sát gần nhất
    birth_frame: int
    state: TrackState = TrackState.NEWBORN
    last_frame: int = 0                   # frame của lần quan sát gần nhất
    hits: int = 1                         # số lần được quan sát (tính cả lúc sinh)
    misses: int = 0                       # số frame liên tiếp không được khớp
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))   # độ dịch hộp mỗi frame
    history: List[Tuple[int, np.ndarray]] = field(default_factory=list)  # (frame, hộp) = lịch sử quỹ đạo

    def predict(self, frame: int, use_velocity: bool) -> np.ndarray:
        """Hộp dự đoán ở `frame`. Không dùng chuyển động: giữ nguyên hộp cuối. Có dùng: ngoại suy tuyến tính."""
        if not use_velocity:
            return self.box
        return self.box + self.velocity * (frame - self.last_frame)

    def update(self, frame: int, box: np.ndarray) -> None:
        dt = frame - self.last_frame
        if dt > 0:
            self.velocity = (box - self.box) / dt
        self.box, self.last_frame = box, frame
        self.hits += 1
        self.misses = 0
        self.history.append((frame, box))


@dataclass
class Event:
    frame: int
    kind: str          # birth | confirm | lost | recover | end
    track_id: int
    info: str = ""

    def __str__(self) -> str:
        return f"[frame {self.frame:02d}] {self.kind:<8} track {self.track_id}" + (f"  ({self.info})" if self.info else "")


# =========================================================================== #
# 4. Bộ theo dõi IoU
# =========================================================================== #
class IoUTracker:
    """Theo dõi bằng IoU + Hungarian, kèm vòng đời track.

    Tham số:
        iou_thr      cặp có IoU thấp hơn ngưỡng này không được ghép (SORT dùng 0.3)
        n_init       số lần khớp để NEWBORN thành ACTIVE
        max_age      số frame tối đa một track được chờ ở trạng thái LOST
        use_velocity True: dự đoán hộp bằng ngoại suy vận tốc (thay cho Kalman Filter, rất đơn giản)
        matcher      match_hungarian | match_greedy
    """

    def __init__(self, iou_thr: float = 0.3, n_init: int = 2, max_age: int = 5,
                 use_velocity: bool = False, matcher: Callable[[np.ndarray, float], Matches] = match_hungarian):
        self.iou_thr, self.n_init, self.max_age = iou_thr, n_init, max_age
        self.use_velocity, self.matcher = use_velocity, matcher
        self.tracks: List[Track] = []
        self.events: List[Event] = []
        self.ended: List[Track] = []
        self._next_id = 1
        self.last_debug: Dict = {}

    def step(self, frame: int, boxes: Sequence[Sequence[float]]) -> List[Tuple[int, np.ndarray, int, str]]:
        """Xử lý một frame. Trả về [(track_id, hộp, chỉ số detection, trạng thái)] của track được khớp/sinh ra."""
        boxes = np.asarray(boxes, float).reshape(-1, 4)

        # (1) dự đoán hộp của mọi track còn sống
        pred = np.array([t.predict(frame, self.use_velocity) for t in self.tracks]).reshape(-1, 4)
        # (2) ma trận IoU và ghép
        M = iou_matrix(pred, boxes)
        matches, un_t, un_d = self.matcher(M, self.iou_thr)
        self.last_debug = dict(track_ids=[t.track_id for t in self.tracks], iou=M, matches=matches,
                               unmatched_tracks=un_t, unmatched_dets=un_d)

        out = []
        # (3) track được khớp: cập nhật, có thể NEWBORN->ACTIVE hoặc LOST->ACTIVE
        for ti, dj in matches:
            t = self.tracks[ti]
            was_lost, lost_for = t.state == TrackState.LOST, t.misses
            t.update(frame, boxes[dj])
            if was_lost:
                t.state = TrackState.ACTIVE
                self.events.append(Event(frame, "recover", t.track_id, f"sau {lost_for} frame mất dấu"))
            elif t.state == TrackState.NEWBORN and t.hits >= self.n_init:
                t.state = TrackState.ACTIVE
                self.events.append(Event(frame, "confirm", t.track_id))
            out.append((t.track_id, boxes[dj].copy(), dj, t.state.value))
        # (4) track không được khớp: NEWBORN bị bỏ ngay; ACTIVE -> LOST; LOST quá max_age -> ENDED
        for ti in un_t:
            t = self.tracks[ti]
            if t.state == TrackState.NEWBORN:
                self._end(t, frame, "chưa xác nhận đã mất")
                continue
            t.misses += 1
            if t.state == TrackState.ACTIVE:
                t.state = TrackState.LOST
                self.events.append(Event(frame, "lost", t.track_id))
            if t.misses > self.max_age:
                self._end(t, frame, f"mất dấu quá {self.max_age} frame")
        # (5) detection chưa được khớp: sinh track mới
        for dj in un_d:
            t = Track(self._next_id, boxes[dj].copy(), frame, last_frame=frame, history=[(frame, boxes[dj].copy())])
            self._next_id += 1
            self.tracks.append(t)
            self.events.append(Event(frame, "birth", t.track_id))
            out.append((t.track_id, boxes[dj].copy(), dj, t.state.value))

        self.tracks = [t for t in self.tracks if t.state != TrackState.ENDED]
        return sorted(out, key=lambda o: o[2])

    def _end(self, t: Track, frame: int, reason: str) -> None:
        t.state = TrackState.ENDED
        self.ended.append(t)
        self.events.append(Event(frame, "end", t.track_id, reason))


# =========================================================================== #
# 5. Kịch bản mô phỏng (có ground-truth)
# =========================================================================== #
@dataclass
class Obj:
    gt_id: int
    x0: float; y0: float          # tâm tại frame `start`
    x1: float; y1: float          # tâm tại frame `end`
    start: int
    end: int
    w: float = 50.0
    h: float = 90.0
    hidden: Tuple[int, int] = (-1, -2)   # [a, b]: các frame bị che (không có detection)

    def box_at(self, f: int) -> Optional[np.ndarray]:
        if not (self.start <= f <= self.end) or self.hidden[0] <= f <= self.hidden[1]:
            return None
        k = (f - self.start) / max(1, self.end - self.start)
        cx, cy = self.x0 + k * (self.x1 - self.x0), self.y0 + k * (self.y1 - self.y0)
        return np.array([cx - self.w / 2, cy - self.h / 2, cx + self.w / 2, cy + self.h / 2])


def make_frames(objs: List[Obj], n: int) -> List[List[Tuple[int, np.ndarray]]]:
    """frames[f] = [(gt_id, hộp)] - các đối tượng nhìn thấy ở frame f."""
    return [[(o.gt_id, b) for o in objs if (b := o.box_at(f)) is not None] for f in range(n)]


def scenario_simple():          # 3 người đi song song, tốc độ 8 px/frame
    objs = [Obj(i + 1, 60, 70 + 110 * i, 60 + 8 * 24, 70 + 110 * i, 0, 24) for i in range(3)]
    return make_frames(objs, 25)


def scenario_occlusion():       # 1 người bị che 5 frame (8..12) rồi xuất hiện lại tại đúng chỗ đi tiếp
    objs = [Obj(1, 60, 100, 60 + 8 * 24, 100, 0, 24, hidden=(8, 12)),
            Obj(2, 60, 250, 60 + 8 * 24, 250, 0, 24)]
    return make_frames(objs, 25)


def scenario_enter_exit():      # B vào ở frame 6, rời ở frame 14
    objs = [Obj(1, 60, 80, 60 + 8 * 24, 80, 0, 24),
            Obj(2, 300, 230, 300 + 8 * 8, 230, 6, 14)]
    return make_frames(objs, 25)


def scenario_crossing():        # CA KHÓ: hai người đi ngược chiều, chồng khít nhau ở giữa
    objs = [Obj(1, 80, 170, 560, 190, 0, 30), Obj(2, 560, 190, 80, 170, 0, 30)]
    return make_frames(objs, 31)


SCENARIOS = {"simple": scenario_simple, "occlusion": scenario_occlusion,
             "enter_exit": scenario_enter_exit, "crossing": scenario_crossing}


def run(frames, seed: Optional[int] = None, **tracker_kw):
    """Chạy tracker, trả (tracker, ids_theo_frame[f] = {gt_id: track_id}).

    seed != None: xáo trộn thứ tự detection ở mỗi frame (detector thật không đảm bảo thứ tự nào).
    Việc này quan trọng: nếu luôn đưa detection theo cùng thứ tự thì khi hai hộp hoà IoU,
    thuật toán có thể 'ghép đúng' chỉ nhờ thứ tự chứ không phải nhờ tính đúng đắn.
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    trk = IoUTracker(**tracker_kw)
    per_frame = []
    for f, objs in enumerate(frames):
        objs = list(objs)
        if rng is not None:
            rng.shuffle(objs)
        outs = trk.step(f, [b for _, b in objs])
        per_frame.append({objs[d][0]: tid for tid, _, d, _ in outs})
    return trk, per_frame


def id_report(per_frame) -> Dict:
    """ID switch = số lần một đối tượng thật nhận ID khác ID gần nhất nó từng có."""
    last, seen, sw = {}, {}, 0
    for fr in per_frame:
        for g, tid in fr.items():
            if g in last and last[g] != tid:
                sw += 1
            last[g] = tid
            if tid not in seen.setdefault(g, []):
                seen[g].append(tid)
    return dict(id_switches=sw, ids_per_gt=seen)


# =========================================================================== #
# 6. In ra: giải thích một frame, bảng ID theo frame
# =========================================================================== #
def explain_one_frame() -> None:
    """Đi qua từng bước của MỘT lần ghép để thấy IoU/Hungarian làm gì."""
    print("=" * 78)
    print("A. GIẢI THÍCH MỘT FRAME GHÉP  (kịch bản 'simple', frame 1)")
    print("=" * 78)
    frames = scenario_simple()
    trk = IoUTracker()
    trk.step(0, [b for _, b in frames[0]])
    print("Frame 0: chưa có track nào -> cả 3 detection sinh track mới (NEWBORN), ID 1, 2, 3.")
    trk.step(1, [b for _, b in frames[1]])
    d = trk.last_debug
    print("\nFrame 1: ma trận IoU (hàng = track, cột = detection):")
    print("          " + "".join(f"det{j:<6}" for j in range(d['iou'].shape[1])))
    for tid, row in zip(d["track_ids"], d["iou"]):
        print(f"  track {tid}  " + "".join(f"{v:<9.2f}" for v in row))
    print(f"\nHungarian (ngưỡng IoU >= {trk.iou_thr}) ghép (track -> det): "
          + ", ".join(f"{d['track_ids'][i]}->{j}" for i, j in d["matches"]))
    print(f"Track chưa khớp: {d['unmatched_tracks']}   Detection chưa khớp: {d['unmatched_dets']}")
    print("Cả 3 track khớp lần 2 (n_init=2) -> NEWBORN thành ACTIVE:")
    for e in trk.events:
        print("  ", e)


def print_id_table(name: str, frames, per_frame, gt_ids: List[int], every: int = 1) -> None:
    print(f"\n  {'frame':>5} | " + " | ".join(f"GT{g}" for g in gt_ids))
    for f in range(0, len(per_frame), every):
        cells = [f"{per_frame[f][g]:>3}" if g in per_frame[f] else "  ." for g in gt_ids]
        print(f"  {f:>5} | " + " | ".join(f" {c}" for c in cells))


# =========================================================================== #
# 7. Kiểm thử nghiệm thu
# =========================================================================== #
SEEDS = range(10)     # mỗi kịch bản chạy với 10 thứ tự detection khác nhau


def run_seeds(sc_fn, **kw):
    """Chạy kịch bản với nhiều thứ tự detection; trả list (tracker, per_frame, báo cáo ID)."""
    out = []
    for sd in SEEDS:
        trk, pf = run(sc_fn(), seed=sd, **kw)
        out.append((trk, pf, id_report(pf)))
    return out


def acceptance() -> bool:
    print("\n" + "=" * 78)
    print(f"C. KIỂM THỬ NGHIỆM THU:  'ví dụ ghép ID đúng trong ca đơn giản'  (mỗi test chạy {len(SEEDS)} thứ tự detection)")
    print("=" * 78)
    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"  [{'ĐẠT' if ok else 'HỎNG'}] {name}" + (f"   {detail}" if detail else ""))

    # 1) ca đơn giản: 3 người song song -> 3 ID cố định
    R = run_seeds(scenario_simple)
    check("Ca đơn giản: 3 người song song giữ nguyên ID",
          all(r["id_switches"] == 0 and len(r["ids_per_gt"]) == 3 for _, _, r in R),
          f"ID switch tối đa = {max(r['id_switches'] for _, _, r in R)}")

    # 2) ghép tham lam cũng đúng ở ca đơn giản
    Rg = run_seeds(scenario_simple, matcher=match_greedy)
    check("Ghép tham lam cũng đúng ở ca đơn giản", all(r["id_switches"] == 0 for _, _, r in Rg))

    # 3) bị che 5 frame
    Rv = run_seeds(scenario_occlusion, use_velocity=True)
    Rn = run_seeds(scenario_occlusion, use_velocity=False)
    # (giá trị ID phụ thuộc thứ tự detection ở frame 0, nên chỉ đòi MỘT ID duy nhất, không đòi ID cụ thể)
    check("Bị che 5 frame + ngoại suy vận tốc: GIỮ ID", all(len(r["ids_per_gt"][1]) == 1 for _, _, r in Rv),
          f"ID người bị che = {Rv[0][2]['ids_per_gt'][1]}")
    check("Bị che 5 frame, không dùng chuyển động: ĐỔI ID (giới hạn của IoU thuần)",
          all(len(r["ids_per_gt"][1]) > 1 for _, _, r in Rn), f"ID người bị che = {Rn[0][2]['ids_per_gt'][1]}")

    # 4) vào / ra
    Re = run_seeds(scenario_enter_exit, max_age=5)
    check("Người vào muộn nhận ID mới, khác ID của người đang có",
          all(r["ids_per_gt"][2][0] not in r["ids_per_gt"][1] and r["id_switches"] == 0 for _, _, r in Re))
    ends = [[e for e in trk.events if e.kind == "end" and e.track_id == r["ids_per_gt"][2][0]] for trk, _, r in Re]
    check("Người rời đi: track kết thúc sau max_age frame mất dấu", all(len(e) == 1 for e in ends),
          f"{ends[0][0] if ends[0] else 'không có sự kiện end'}")

    # 5) không trùng ID trong một frame
    dup = any(len(set(fr.values())) != len(fr) for grp in (R, Rg, Rv, Rn, Re) for _, pf, _ in grp for fr in pf)
    check("Không có hai đối tượng trùng ID trong cùng một frame", not dup)

    ok = all(results)
    print(f"\n  => {'TẤT CẢ ĐẠT' if ok else 'CÓ TEST HỎNG'} ({sum(results)}/{len(results)})")
    return ok


def show_limit() -> None:
    print("\n" + "=" * 78)
    print("D. CA KHÓ (chỉ để quan sát, không nằm trong nghiệm thu): hai người chồng khít rồi đi tiếp")
    print("=" * 78)
    n = 30
    for vel in (False, True):
        sw = [id_report(run(scenario_crossing(), seed=sd, use_velocity=vel)[1])["id_switches"] for sd in range(n)]
        print(f"  use_velocity={vel!s:<5}  có ID switch ở {sum(x > 0 for x in sw)}/{n} lần chạy (thứ tự detection khác nhau)")
    print("  Khi hai hộp chồng khít, IoU của hai cách ghép gần như bằng nhau nên kết quả phụ thuộc may rủi (thứ tự detection).")
    print("  Đây là lý do DeepSORT thêm đặc trưng ngoại hình và MOTIP học cách dự đoán ID.")


# =========================================================================== #
# 8. Vẽ ảnh
# =========================================================================== #
def plot(show: bool) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    palette = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#e67e22", "#1abc9c"]
    frames = scenario_occlusion()
    shown = [0, 7, 10, 14]
    configs = [("IoU thuần (giữ nguyên hộp cuối)", False), ("IoU + ngoại suy vận tốc", True)]
    fig, axes = plt.subplots(2, len(shown), figsize=(15, 6.2))
    for r, (title, vel) in enumerate(configs):
        trk = IoUTracker(use_velocity=vel)
        for f, objs in enumerate(frames):
            outs = trk.step(f, [b for _, b in objs])
            if f in shown:
                ax = axes[r][shown.index(f)]
                ax.set_xlim(0, 300); ax.set_ylim(350, 0); ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"frame {f}" + (" (người 1 bị che)" if 8 <= f <= 12 else ""), fontsize=10)
                if shown.index(f) == 0:
                    ax.set_ylabel(title, fontsize=10)
                for tid, b, _, state in outs:
                    c = palette[(tid - 1) % len(palette)]
                    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=2.5))
                    ax.text(b[0], b[1] - 4, f"ID {tid}", color=c, fontsize=10, fontweight="bold")
                for t in trk.tracks:
                    if t.state == TrackState.LOST:
                        p = t.predict(f, vel)
                        ax.add_patch(Rectangle((p[0], p[1]), p[2] - p[0], p[3] - p[1], fill=False, ec="gray", lw=1.5, ls="--"))
                        ax.text(p[0], p[3] + 14, f"ID {t.track_id} LOST (dự đoán)", color="gray", fontsize=8)
    fig.suptitle("Che khuất 5 frame: IoU thuần mất ID, ngoại suy vận tốc giữ được ID", fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "demo_baseline_frames.png")
    fig.savefig(path, dpi=130)
    print(f"\nĐã lưu ảnh: {path}")
    if show:
        plt.show()


# =========================================================================== #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-plot", action="store_true", help="không vẽ ảnh")
    ap.add_argument("--show", action="store_true", help="mở cửa sổ ảnh")
    args = ap.parse_args()

    explain_one_frame()

    print("\n" + "=" * 78)
    print("B. BẢNG ID THEO FRAME  (cột = đối tượng thật GT; ô = ID track; '.' = không có detection)")
    print("=" * 78)
    for name, use_vel in (("simple", False), ("occlusion", True), ("enter_exit", False)):
        trk, pf = run(SCENARIOS[name](), use_velocity=use_vel)
        gts = sorted({g for fr in pf for g in fr})
        print(f"\n  Kịch bản '{name}' (use_velocity={use_vel})")
        print_id_table(name, None, pf, gts, every=2 if name == "simple" else 1)
        print("  Sự kiện vòng đời:")
        for e in trk.events:
            print("   ", e)

    ok = acceptance()
    show_limit()
    if not args.no_plot:
        try:
            plot(args.show)
        except ImportError:
            print("\n[bỏ qua vẽ ảnh] chưa cài matplotlib:  pip install matplotlib")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
