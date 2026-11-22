"""
track_manager.py
================
Quản lý vòng đời quỹ đạo + ID cho tracking kiểu MOTIP.

Ba loại "ID" khác nhau -- đây là điểm dễ nhầm nhất:

  uid       ID NỘI BỘ. Tăng dần, KHÔNG BAO GIỜ tái sử dụng. Khoá chính của quỹ đạo.
  label     NHÃN trong ID dictionary của decoder, thuộc 0..K-1 (bài báo đánh số 1..K).
            Số lượng hữu hạn K nên PHẢI tái sử dụng khi quỹ đạo kết thúc (mục 3.4:
            "we recycle the ID labels of the concluded trajectories for reuse").
  output_id ID XUẤT RA VIDEO / file kết quả. Mặc định = uid, để hai người khác nhau
            không bao giờ mang cùng một ID trong cùng video (nếu dùng label thì
            sau khi tái sử dụng nhãn, metric như IDF1 sẽ coi là cùng một người).

Vòng đời (xem lifecycle_diagram.png):

    (detection mới, score >= λ_new)
              |
              v
          NEWBORN --match--> ACTIVE <--------match--------+
              |                 |                         |
          miss (chưa           miss                       |
          xác nhận)             v                         |
              |              LOST ----------match---------+
              |                 |
              |         lost > max_lost
              v                 v
            ENDED <-------------+      (nhãn được trả về pool, bộ nhớ bị xoá)

Module chỉ phụ thuộc numpy.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from track_memory import (ArchiveRecord, MemoryBank, Observation,
                          TrajectoryArchive)


class TrackState(enum.Enum):
    NEWBORN = "newborn"   # vừa sinh, chưa được xác nhận bằng lần khớp thứ hai
    ACTIVE = "active"     # đang được quan sát
    LOST = "lost"         # đang mất dấu (che khuất / miss detection), còn trong thời gian chờ
    ENDED = "ended"       # đã kết thúc, nhãn đã trả về pool


LIVE_STATES = (TrackState.NEWBORN, TrackState.ACTIVE, TrackState.LOST)


@dataclass
class Track:
    uid: int
    label: int
    birth_frame: int
    state: TrackState = TrackState.NEWBORN
    last_seen: int = -1
    hits: int = 1               # số lần quan sát (kể cả lúc sinh)
    lost_count: int = 0         # số frame liên tiếp đang mất dấu
    end_frame: Optional[int] = None
    end_reason: Optional[str] = None


@dataclass
class Event:
    """Nhật ký sự kiện (frame, loại, uid, nhãn, ghi chú) -- phục vụ demo & test."""
    frame: int
    kind: str      # birth | confirm | lost | recover | end | duplicate | no_label | reclaim
    uid: int
    label: int
    info: str = ""

    def __str__(self) -> str:
        extra = f" ({self.info})" if self.info else ""
        return f"[f{self.frame:03d}] {self.kind:<9} uid={self.uid:<3} label={self.label:<3}{extra}"


class LabelPool:
    """Pool nhãn của ID dictionary: cấp lowest-free-first, thu hồi khi quỹ đạo kết thúc.

    Cấp nhãn nhỏ nhất còn trống => khớp với mô tả "assign IDs from 1 to K"
    của bài báo, và cho kết quả xác định (deterministic).
    """

    def __init__(self, num_labels: int):
        if num_labels < 1:
            raise ValueError("num_labels phải >= 1")
        self.num_labels = num_labels
        self._in_use: Set[int] = set()

    def acquire(self) -> Optional[int]:
        for k in range(self.num_labels):
            if k not in self._in_use:
                self._in_use.add(k)
                return k
        return None   # hết nhãn

    def release(self, label: int) -> None:
        if label not in self._in_use:
            raise ValueError(f"nhãn {label} chưa được cấp, không thể trả")
        self._in_use.remove(label)

    @property
    def in_use(self) -> Set[int]:
        return set(self._in_use)

    @property
    def n_free(self) -> int:
        return self.num_labels - len(self._in_use)


class TrackManager:
    """Điều phối: sinh track, cập nhật, mất dấu, kết thúc, tái sử dụng nhãn, giải quyết trùng ID.

    Thứ tự gọi trong mỗi frame (do tracking_inference.py thực hiện):
        1. resolve_duplicate_labels(...)   -> ai được giữ nhãn dự đoán
        2. update_track(...)               -> track được khớp
        3. mark_missed(...)                -> track không được khớp (LOST / ENDED)
        4. create_track(...)               -> newborn
        5. record_outputs(...)             -> ghi lịch sử dài hạn
    """

    def __init__(self, num_labels: int = 50, history_len: int = 29,
                 max_lost: Optional[int] = None, confirm_hits: int = 2,
                 output_id_mode: str = "uid"):
        self.max_lost = history_len if max_lost is None else max_lost
        if self.max_lost > history_len:
            # sau T frame không quan sát, cửa sổ rỗng -> decoder không thể khớp lại;
            # giữ track "zombie" chỉ làm tốn nhãn.
            raise ValueError(f"max_lost ({self.max_lost}) không được lớn hơn history_len ({history_len})")
        if output_id_mode not in ("uid", "label"):
            raise ValueError("output_id_mode phải là 'uid' hoặc 'label'")
        self.history_len = history_len
        self.confirm_hits = confirm_hits
        self.output_id_mode = output_id_mode

        self.pool = LabelPool(num_labels)
        self.memory = MemoryBank(history_len)
        self.archive = TrajectoryArchive()
        self.tracks: Dict[int, Track] = {}      # chỉ track chưa ENDED
        self.events: List[Event] = []
        self._next_uid = 1

    # ------------------------------------------------------------------ #
    # truy vấn
    # ------------------------------------------------------------------ #
    def live_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.state in LIVE_STATES]

    def by_label(self) -> Dict[int, Track]:
        return {t.label: t for t in self.live_tracks()}

    def output_id(self, track: Track) -> int:
        return track.uid if self.output_id_mode == "uid" else track.label + 1

    def _log(self, frame: int, kind: str, t: Track, info: str = "") -> None:
        self.events.append(Event(frame, kind, t.uid, t.label, info))

    # ------------------------------------------------------------------ #
    # sinh track
    # ------------------------------------------------------------------ #
    def create_track(self, frame: int, box, feature, score: float) -> Optional[Track]:
        """Tạo quỹ đạo newborn với uid mới + nhãn lấy từ pool.

        Pool cạn -> thu hồi nhãn của track LOST đã mất dấu lâu nhất. Nếu không có
        track LOST nào thì bỏ detection này (trả None) thay vì cướp nhãn của
        một đối tượng đang được quan sát.
        """
        label = self.pool.acquire()
        if label is None:
            victim = self._longest_lost()
            if victim is None:
                self.events.append(Event(frame, "no_label", -1, -1,
                                         "pool cạn, không có track LOST để thu hồi -> bỏ detection"))
                return None
            self._log(frame, "reclaim", victim, "bị thu hồi nhãn để cấp cho newborn")
            self._end(victim, frame, "label_reclaimed")
            label = self.pool.acquire()
            assert label is not None

        t = Track(uid=self._next_uid, label=label, birth_frame=frame, last_seen=frame)
        self._next_uid += 1
        self.tracks[t.uid] = t
        mem = self.memory.create(t.uid)
        mem.append(Observation(frame, np.asarray(box, float), np.asarray(feature, np.float32), float(score)))
        self.archive.open_life(t.uid, t.label, frame)
        self._log(frame, "birth", t, f"score={score:.2f}")
        return t

    def _longest_lost(self) -> Optional[Track]:
        lost = [t for t in self.tracks.values() if t.state == TrackState.LOST]
        return max(lost, key=lambda t: t.lost_count) if lost else None

    # ------------------------------------------------------------------ #
    # cập nhật
    # ------------------------------------------------------------------ #
    def update_track(self, track: Track, frame: int, box, feature, score: float) -> None:
        """Track được khớp với một detection ở frame này."""
        self.memory.get(track.uid).append(
            Observation(frame, np.asarray(box, float), np.asarray(feature, np.float32), float(score)))
        track.hits += 1
        track.last_seen = frame
        if track.state == TrackState.LOST:
            track.state = TrackState.ACTIVE
            self.archive.lives[track.uid].n_recoveries += 1
            self._log(frame, "recover", track, f"sau {track.lost_count} frame mất dấu")
        elif track.state == TrackState.NEWBORN and track.hits >= self.confirm_hits:
            track.state = TrackState.ACTIVE
            self._log(frame, "confirm", track)
        track.lost_count = 0

    def mark_missed(self, track: Track, frame: int) -> None:
        """Track không được khớp ở frame này."""
        if track.state == TrackState.NEWBORN:
            # chưa xác nhận mà đã mất -> nhiều khả năng là detection nhiễu
            self._end(track, frame, "unconfirmed")
            return
        track.lost_count += 1
        self.archive.lives[track.uid].n_lost_frames += 1
        if track.state == TrackState.ACTIVE:
            track.state = TrackState.LOST
            self._log(frame, "lost", track)
        if track.lost_count > self.max_lost:
            self._end(track, frame, "lost_timeout")

    def _end(self, track: Track, frame: int, reason: str) -> None:
        track.state = TrackState.ENDED
        track.end_frame, track.end_reason = frame, reason
        self.pool.release(track.label)       # trả nhãn -> có thể tái sử dụng
        self.memory.drop(track.uid)          # xoá token -> không rò danh tính cũ
        self.archive.close_life(track.uid, frame, reason)
        self._log(frame, "end", track, reason)
        del self.tracks[track.uid]

    def end_all(self, frame: int, reason: str = "video_end") -> None:
        for t in list(self.tracks.values()):
            self._end(t, frame, reason)

    # ------------------------------------------------------------------ #
    # trùng ID (mục 3.4)
    # ------------------------------------------------------------------ #
    def resolve_duplicate_labels(self, frame: int, pred_labels: Sequence[int],
                                 det_scores: Sequence[float]) -> Tuple[Dict[int, int], List[int]]:
        """Nhiều detection cùng dự đoán một nhãn -> chỉ detection có confidence cao nhất giữ nhãn.

        Các detection còn lại KHÔNG bị bỏ: chúng được coi là chưa khớp và (nếu đủ
        điểm λ_new) sẽ thành newborn. Đó là luật của bài báo: "the object with the
        lower confidence is treated as a newborn".

        pred_labels[i] = nhãn dự đoán của detection i, hoặc -1 nếu không dự đoán
        (xác suất < λ_id). Trả về (label -> det_index thắng, danh sách det_index thua).
        Hoà điểm: lấy detection có chỉ số nhỏ hơn (xác định).
        """
        winners: Dict[int, int] = {}
        losers: List[int] = []
        for i, lab in enumerate(pred_labels):
            if lab < 0:
                continue
            if lab not in winners:
                winners[lab] = i
            else:
                j = winners[lab]
                if det_scores[i] > det_scores[j]:
                    winners[lab], loser = i, j
                else:
                    loser = i
                losers.append(loser)
                tr = self.by_label().get(lab)
                self.events.append(Event(frame, "duplicate", tr.uid if tr else -1, lab,
                                         f"det {loser} thua det {winners[lab]} (trùng nhãn {lab})"))
        return winners, sorted(losers)

    # ------------------------------------------------------------------ #
    # lịch sử dài hạn
    # ------------------------------------------------------------------ #
    def record_output(self, frame: int, track: Track, box, score: float) -> None:
        self.archive.record(ArchiveRecord(frame, track.uid, track.label, self.output_id(track),
                                          np.asarray(box, float), float(score), track.state.value))
