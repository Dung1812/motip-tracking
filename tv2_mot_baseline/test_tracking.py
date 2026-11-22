"""
test_tracking.py -- kiểm thử nghiệm thu: "ID ổn định trong các tình huống đơn giản;
quy tắc trùng ID và ID mới hoạt động đúng".   Chạy:  python -m pytest -q test_tracking.py
"""
import numpy as np
import pytest

import scenarios as S
from runner import run_motip, run_sort
from sort_baseline import SortTracker
from track_manager import LabelPool, TrackManager, TrackState
from tracking_inference import AppearanceMockDecoder, InferenceConfig, MOTIPTracker


def kinds(trk, kind):
    return [e for e in trk.events if e.kind == kind]


# ---------------------------------------------------------------- ID ổn định
@pytest.mark.parametrize("name", ["parallel", "crossing", "bounce", "enter_exit"])
@pytest.mark.parametrize("seed", range(10))
def test_ids_stable(name, seed):
    _, _, m = run_motip(S.ALL[name](seed=seed))
    assert m.id_switches == 0, m.ids_per_gt
    assert m.shared_ids == []
    assert m.fragmented_gts == 0


def test_short_occlusion_keeps_same_id():
    trk, _, m = run_motip(S.occlusion(gap=8))
    assert m.id_switches == 0 and m.fragmented_gts == 0
    assert len(kinds(trk, "lost")) == 1 and len(kinds(trk, "recover")) == 1
    assert trk.mgr.archive.lives[1].n_recoveries == 1


def test_long_occlusion_ends_old_track_and_makes_new_id():
    # gap 40 > max_lost 29: quỹ đạo cũ kết thúc, người xuất hiện lại là ID mới (hành vi mong đợi)
    trk, _, m = run_motip(S.occlusion(gap=40))
    assert m.ids_per_gt[1] == [1, 3]
    assert trk.mgr.archive.lives[1].end_reason == "lost_timeout"


# ---------------------------------------------------------------- ID mới
def test_newborn_gets_fresh_uid_and_unused_label():
    trk, per_frame, _ = run_motip(S.enter_exit())
    births = {e.uid: e for e in kinds(trk, "birth")}
    assert sorted(births) == [1, 2, 3]
    first = {uid: next(o for fr in per_frame for o in fr if o.uid == uid) for uid in births}
    assert all(o.is_new for o in first.values())
    assert first[2].label not in {first[1].label}            # B không đè nhãn của A
    assert first[3].label not in {first[1].label}


def test_low_score_detections_do_not_create_tracks():
    trk = MOTIPTracker(AppearanceMockDecoder(50), InferenceConfig())
    box = [[10, 10, 60, 100]]
    f = np.ones((1, 8), np.float32)
    assert trk.step(0, box, [0.2], f) == []        # < λ_det: bị lọc
    assert trk.step(1, box, [0.5], f) == []        # λ_det <= s < λ_new: không đủ để sinh newborn
    assert len(trk.step(2, box, [0.7], f)) == 1    # >= λ_new: sinh newborn
    assert len(kinds(trk, "birth")) == 1


# ---------------------------------------------------------------- trùng ID
class ForcedDecoder:
    """Decoder ép dự đoán: probs[i] = (nhãn, xác suất) cho detection i."""
    def __init__(self, num_labels, plan):
        self.num_labels, self.plan = num_labels, plan

    def __call__(self, batch, det_feats):
        p = np.zeros((len(det_feats), self.num_labels), np.float32)
        for i, (lab, pr) in enumerate(self.plan):
            p[i, lab] = pr
        return p


def _two_confirmed_tracks():
    sc = S.parallel()
    trk = MOTIPTracker(AppearanceMockDecoder(50), InferenceConfig())
    for f in range(3):
        trk.step(f, sc.frames[f].boxes[:2], sc.frames[f].scores[:2], sc.frames[f].feats[:2])
    assert {t.state for t in trk.mgr.live_tracks()} == {TrackState.ACTIVE}
    return sc, trk


def test_duplicate_label_higher_score_keeps_id_other_becomes_newborn():
    sc, trk = _two_confirmed_tracks()
    fr = sc.frames[3]
    scores = np.array([0.80, 0.95])                        # det1 điểm cao hơn
    trk.decoder = ForcedDecoder(50, [(0, 0.9), (0, 0.9)])  # cả hai cùng dự đoán nhãn 0
    outs = trk.step(3, fr.boxes[:2], scores, fr.feats[:2])
    by_det = {o.det_index: o for o in outs}
    assert by_det[1].uid == 1 and not by_det[1].is_new     # thắng -> giữ ID của quỹ đạo uid 1
    assert by_det[0].is_new and by_det[0].uid == 3         # thua -> newborn, uid MỚI (không phải 1, 2)
    assert by_det[0].label not in (0, 1)                   # nhãn mới, không trùng nhãn đang dùng
    assert len(kinds(trk, "duplicate")) == 1
    assert len({o.output_id for o in outs}) == len(outs)   # không có hai đối tượng cùng ID trong 1 frame


def test_duplicate_loser_below_lambda_new_is_dropped_not_born():
    sc, trk = _two_confirmed_tracks()
    fr = sc.frames[3]
    trk.decoder = ForcedDecoder(50, [(0, 0.9), (0, 0.9)])
    outs = trk.step(3, fr.boxes[:2], np.array([0.45, 0.95]), fr.feats[:2])   # thua nhưng 0.45 < λ_new
    assert [o.det_index for o in outs] == [1]
    assert len(kinds(trk, "birth")) == 2                   # chỉ 2 lần sinh ban đầu


def test_prediction_below_lambda_id_is_not_assigned():
    sc, trk = _two_confirmed_tracks()
    fr = sc.frames[3]
    trk.decoder = ForcedDecoder(50, [(0, 0.1), (1, 0.1)])   # p=0.1 < λ_id=0.2
    outs = trk.step(3, fr.boxes[:2], fr.scores[:2], fr.feats[:2])
    assert all(o.is_new for o in outs)                     # không ai được khớp -> cả hai newborn (score 0.9 >= 0.6)


def test_unallocated_label_is_never_predicted():
    sc, trk = _two_confirmed_tracks()
    fr = sc.frames[3]
    trk.decoder = ForcedDecoder(50, [(7, 0.99), (8, 0.99)])  # nhãn 7, 8 chưa có quỹ đạo
    outs = trk.step(3, fr.boxes[:2], fr.scores[:2], fr.feats[:2])
    assert all(o.is_new for o in outs)


# ---------------------------------------------------------------- kết thúc & tái sử dụng nhãn
def test_exit_ends_track_and_releases_label():
    trk, _, _ = run_motip(S.enter_exit(), InferenceConfig(max_lost=10, history_len=15))
    life = trk.mgr.archive.lives[2]
    assert life.end_reason == "lost_timeout"
    assert life.end_frame == 35 + 1 + 10                   # rời sau frame 35, quá max_lost=10
    # C sinh ở frame 40, lúc B (đã LOST từ frame 36) chưa kết thúc (frame 46) -> nhãn của B vẫn bị giữ
    assert trk.mgr.archive.lives[3].birth_frame == 40
    assert trk.mgr.archive.lives[3].label != life.label


def test_label_reuse_gives_new_uid_and_distinct_output_ids():
    cfg = InferenceConfig(num_labels=2, history_len=8, max_lost=5)
    trk, _, m = run_motip(S.sequential(n=6), cfg)
    lives = trk.mgr.archive.lives
    assert len(lives) == 6 and not kinds(trk, "no_label")
    labels = [lives[u].label for u in sorted(lives)]
    assert set(labels) <= {0, 1} and len(set(labels)) < 6   # nhãn ĐƯỢC tái sử dụng
    assert m.id_switches == 0 and m.shared_ids == [] and m.fragmented_gts == 0   # nhưng output ID không trùng
    assert len(m.ids_per_gt) == 6


def test_output_id_mode_label_shows_why_uid_is_default():
    cfg = InferenceConfig(num_labels=2, history_len=8, max_lost=5, output_id_mode="label")
    _, _, m = run_motip(S.sequential(n=6), cfg)
    assert m.shared_ids != []                               # người khác nhau dùng chung ID xuất ra


def test_ended_track_memory_is_cleared():
    cfg = InferenceConfig(num_labels=2, history_len=8, max_lost=5)
    trk, _, _ = run_motip(S.sequential(n=3), cfg)
    assert all(uid not in trk.mgr.memory for uid in trk.mgr.archive.lives)   # sau finish(): không còn bộ nhớ nào


def test_pool_exhausted_drops_detection_without_crash():
    cfg = InferenceConfig(num_labels=2)
    trk, per_frame, _ = run_motip(S.parallel(), cfg)        # 3 người, chỉ 2 nhãn
    assert len(trk.mgr.archive.lives) == 2
    assert kinds(trk, "no_label")
    assert all(len(fr) <= 2 for fr in per_frame)


def test_pool_exhausted_reclaims_label_of_longest_lost_track():
    # A bị che từ frame 10 mãi mãi; C xuất hiện ở frame 14 khi K=2 đã cạn -> thu hồi nhãn của A
    objs = [S.SimObject(1, S.line(80, 100, 300, 100, 0, 60), 0, 60, occluded=[(10, 60)]),
            S.SimObject(2, S.line(80, 250, 300, 250, 0, 60), 0, 60),
            S.SimObject(3, S.line(400, 180, 560, 180, 14, 60), 14, 60)]
    trk, _, _ = run_motip(S.render(objs, 61, "reclaim"), InferenceConfig(num_labels=2))
    lives = trk.mgr.archive.lives
    assert lives[1].end_reason == "label_reclaimed"
    assert lives[3].label == lives[1].label and lives[3].uid != lives[1].uid
    assert kinds(trk, "reclaim")


# ---------------------------------------------------------------- đơn vị
def test_label_pool_lowest_free_first_and_release():
    p = LabelPool(3)
    assert [p.acquire() for _ in range(3)] == [0, 1, 2] and p.acquire() is None
    p.release(1)
    assert p.acquire() == 1
    with pytest.raises(ValueError):
        p.release(5)


def test_max_lost_larger_than_history_is_rejected():
    with pytest.raises(ValueError):
        TrackManager(num_labels=5, history_len=10, max_lost=11)


def test_memory_window_is_limited_to_T_frames():
    cfg = InferenceConfig(history_len=5)
    sc = S.parallel()
    trk = MOTIPTracker(AppearanceMockDecoder(50), cfg)
    for f in range(20):
        trk.step(f, sc.frames[f].boxes, sc.frames[f].scores, sc.frames[f].feats)
    batch = trk.mgr.memory.build_batch(20, trk.mgr.live_tracks())
    assert batch.mask.sum(1).tolist() == [5, 5, 5]
    assert batch.dt.max() == 5 and batch.dt[batch.mask].min() == 1


# ---------------------------------------------------------------- baseline truyền thống
def test_sort_baseline_stable_on_simple_case():
    _, m = run_sort(S.parallel())
    assert m.id_switches == 0


def test_sort_baseline_fails_where_id_prediction_does_not():
    # Kalman hằng vận tốc + IoU bị đánh lừa khi hai đối tượng chồng khít rồi quay đầu
    for name in ("crossing", "bounce"):
        _, ms = run_sort(S.ALL[name]())
        _, mm = run_motip(S.ALL[name]())[1:]
        assert ms.id_switches >= 1 and mm.id_switches == 0
