"""
test_interface.py -- kiểm thử tv2_interface.py theo hợp đồng interface.md v0.1.
Chạy:  python -m pytest -q test_interface.py
"""
import numpy as np
import pytest

import scenarios as S
from tv1_bridge import MockDetector, adapt_tv1_output
from tv2_interface import (TrackerTV2, check_detection, only_drawn, parse_detections, to_numpy,
                           tracks_to_mot_lines, write_mot)


def det(box, score=0.9, emb=None, C=8):
    return {"bbox": list(box), "score": score,
            "embedding": emb if emb is not None else np.ones(C, np.float32)}


def run(name, **kw):
    sc = S.ALL[name]()
    d = MockDetector(sc)
    trk = TrackerTV2(AppearanceDec(), strict=True, **kw)
    res = {f: trk.update(d(None, f), f) for f in range(1, sc.num_frames + 1)}
    res[sc.num_frames + 1] = trk.finish(sc.num_frames + 1)
    return trk, res


def AppearanceDec():
    from tracking_inference import AppearanceMockDecoder
    return AppearanceMockDecoder(50)


# ------------------------------------------------------------ đúng hợp đồng đầu ra
def test_track_dict_has_exact_contract_keys_and_types():
    _, res = run("parallel")
    t = res[1][0]
    assert set(t) == {"track_id", "bbox", "score", "state"}
    assert isinstance(t["track_id"], int) and t["track_id"] >= 1
    assert len(t["bbox"]) == 4 and all(isinstance(v, float) for v in t["bbox"])
    assert isinstance(t["score"], float) and 0 <= t["score"] <= 1


def test_state_new_then_tracked():
    _, res = run("parallel")
    assert all(t["state"] == "new" for t in res[1])
    assert all(t["state"] == "tracked" for t in res[2])


def test_default_returns_all_four_states():
    """Trả lời câu hỏi mở mục 6: mặc định TV2 trả cả lost và terminated."""
    _, res = run("enter_exit")
    states = {t["state"] for fr in res.values() for t in fr}
    assert {"new", "tracked", "lost", "terminated"} <= states


def test_hide_lost_returns_only_new_and_tracked():
    _, res = run("occlusion", include_lost=False, include_terminated=False)
    states = {t["state"] for fr in res.values() for t in fr}
    assert states <= {"new", "tracked"}


def test_only_drawn_filters_lost_and_terminated():
    _, res = run("enter_exit")
    kept = [t for fr in res.values() for t in only_drawn(fr)]
    assert kept and {t["state"] for t in kept} <= {"new", "tracked"}


def test_lost_and_terminated_returned_when_requested():
    _, res = run("enter_exit", include_lost=True, include_terminated=True)
    states = {t["state"] for fr in res.values() for t in fr}
    assert {"new", "tracked", "lost", "terminated"} <= states


def test_lost_track_keeps_same_id_and_last_bbox():
    _, res = run("occlusion", include_lost=True)
    lost_frames = [f for f, fr in res.items() if any(t["state"] == "lost" for t in fr)]
    assert lost_frames
    f = lost_frames[0]
    lost = next(t for t in res[f] if t["state"] == "lost")
    prev = next(t for t in res[f - 1] if t["track_id"] == lost["track_id"])
    assert lost["bbox"] == prev["bbox"]                       # vị trí quan sát cuối
    back = [t for fr in res.values() for t in fr if t["track_id"] == lost["track_id"] and t["state"] == "tracked"]
    assert back and max(f2 for f2, fr in res.items() for t in fr
                        if t["track_id"] == lost["track_id"]) > f   # người bị che xuất hiện lại, cùng ID


def test_terminated_emitted_once_per_track():
    _, res = run("enter_exit", include_terminated=True)
    term = [t["track_id"] for fr in res.values() for t in fr if t["state"] == "terminated"]
    assert len(term) == len(set(term)) and term


# ------------------------------------------------------------ ID không tái sử dụng
def test_track_id_never_reused_after_termination():
    trk, res = run("sequential", include_terminated=True)
    ended_at = {}
    for f, fr in res.items():
        for t in fr:
            if t["state"] == "terminated":
                ended_at[t["track_id"]] = f
    assert ended_at
    for f, fr in res.items():
        for t in fr:
            if t["track_id"] in ended_at and t["state"] in ("new", "tracked"):
                assert f < ended_at[t["track_id"]], "ID đã kết thúc bị dùng lại"


def test_reset_restarts_ids_from_one():
    sc = S.ALL["parallel"]()
    d = MockDetector(sc)
    trk = TrackerTV2(AppearanceDec())
    first = trk.update(d(None, 1), 1)
    trk.reset()
    again = trk.update(d(None, 1), 1)             # frame_idx lại bắt đầu từ 1 sau reset
    assert sorted(t["track_id"] for t in first) == sorted(t["track_id"] for t in again) == [1, 2, 3]


# ------------------------------------------------------------ frame_idx
def test_frame_idx_must_start_at_one_and_increase():
    trk = TrackerTV2(AppearanceDec())
    with pytest.raises(ValueError):
        trk.update([], 0)
    trk.update([], 1)
    with pytest.raises(ValueError):
        trk.update([], 1)


def test_empty_frame_is_handled_and_makes_track_lost():
    trk = TrackerTV2(AppearanceDec(), include_lost=True)
    for f in (1, 2):
        trk.update([det([10, 10, 60, 100])], f)
    out = trk.update([], 3)
    assert [t["state"] for t in out] == ["lost"]


# ------------------------------------------------------------ kiểm tra detection (đầu ra TV1)
@pytest.mark.parametrize("bad", [
    {"bbox": [1, 2, 3], "score": .5, "embedding": np.ones(4)},                 # thiếu 1 số
    {"bbox": [10, 10, 5, 20], "score": .5, "embedding": np.ones(4)},           # x2 < x1
    {"bbox": [0, 0, 5, 5], "score": 1.5, "embedding": np.ones(4)},             # score ngoài [0,1]
    {"bbox": [0, 0, 5, 5], "score": .5, "embedding": np.ones((2, 2))},         # embedding sai shape
    {"bbox": [0, 0, 5, 5], "score": .5, "embedding": np.array([np.nan, 1.])},  # NaN
    {"bbox": [0, 0, 5, 5], "score": .5},                                       # thiếu embedding
])
def test_bad_detection_is_reported(bad):
    assert check_detection(bad) is not None
    with pytest.raises(ValueError):
        parse_detections([bad], strict=True)


def test_non_strict_drops_bad_detection_with_warning():
    trk = TrackerTV2(AppearanceDec(), strict=False)
    with pytest.warns(UserWarning):
        out = trk.update([det([10, 10, 5, 20]), det([10, 10, 60, 100], .9)], 1)
    assert len(out) == 1 and trk.n_dropped == 1


def test_embedding_dim_must_be_consistent():
    trk = TrackerTV2(AppearanceDec(), strict=True)
    trk.update([det([10, 10, 60, 100], C=8)], 1)
    with pytest.raises(ValueError, match="chiều"):
        trk.update([det([10, 10, 60, 100], C=16)], 2)


def test_torch_like_tensor_is_accepted():
    class FakeTensor:                                  # mô phỏng torch.Tensor (có grad, trên GPU)
        def __init__(self, a): self.a = a
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.a
    d = det([10, 10, 60, 100], emb=FakeTensor(np.arange(8, dtype=np.float32)))
    assert check_detection(d) is None
    assert to_numpy(d["embedding"]).shape == (8,)


def test_adapt_tv1_output_xywh_and_names():
    raw = [{"box": [10, 20, 30, 40], "conf": 0.8, "feat": np.ones(4)}]
    out = adapt_tv1_output(raw, bbox_key="box", score_key="conf", emb_key="feat", xywh=True)
    assert out[0]["bbox"] == [10.0, 20.0, 40.0, 60.0] and check_detection(out[0]) is None


# ------------------------------------------------------------ file MOT (mục 5)
def test_mot_line_matches_contract_example():
    tracks = [{"track_id": 3, "bbox": [433.0, 216.0, 513.0, 306.0], "score": 0.80, "state": "tracked"}]
    assert tracks_to_mot_lines(1, tracks) == ["1,3,433.00,216.00,80.00,90.00,0.80,-1,-1,-1"]


def test_mot_skips_lost_and_terminated(tmp_path):
    base = {"track_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9}
    res = {1: [dict(base, state="new")], 2: [dict(base, state="lost")], 3: [dict(base, state="terminated")]}
    p = tmp_path / "r.txt"
    assert write_mot(str(p), res) == 1
    assert p.read_text().splitlines() == ["1,1,0.00,0.00,10.00,10.00,0.90,-1,-1,-1"]


def test_mot_file_has_ten_columns_and_positive_frame():
    _, res = run("demo")
    lines = [l for f, fr in res.items() for l in tracks_to_mot_lines(f, fr)]
    assert lines
    for l in lines:
        c = l.split(",")
        assert len(c) == 10 and int(c[0]) >= 1 and int(c[1]) >= 1 and c[7:] == ["-1", "-1", "-1"]
        assert float(c[4]) > 0 and float(c[5]) > 0


# ------------------------------------------------------------ chất lượng giữ ID qua giao diện
@pytest.mark.parametrize("name", ["parallel", "crossing", "occlusion", "enter_exit"])
def test_ids_stable_through_interface(name):
    sc = S.ALL[name]()
    d = MockDetector(sc)
    trk = TrackerTV2(AppearanceDec())
    assign = []
    for f in range(1, sc.num_frames + 1):
        tracks = only_drawn(trk.update(d(None, f), f))
        gt = sc.frames[f - 1]
        row = {}
        for t in tracks:
            b = np.array(t["bbox"])
            ious = []
            for g in gt.boxes:
                ix = max(0, min(b[2], g[2]) - max(b[0], g[0])); iy = max(0, min(b[3], g[3]) - max(b[1], g[1]))
                inter = ix * iy
                ious.append(inter / ((b[2]-b[0])*(b[3]-b[1]) + (g[2]-g[0])*(g[3]-g[1]) - inter))
            if ious and max(ious) > 0.5:
                row[int(gt.gt_ids[int(np.argmax(ious))])] = t["track_id"]
        assign.append(row)
    assert S.evaluate_ids(assign).id_switches == 0
