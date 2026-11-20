"""runner.py -- chạy tracker (MOTIP-style hoặc SORT) trên Scenario và trả về ID theo frame."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from scenarios import Scenario, evaluate_ids, IDMetrics
from sort_baseline import SortTracker
from tracking_inference import (AppearanceMockDecoder, InferenceConfig, MOTIPTracker,
                                TrackOutput)


def run_motip(sc: Scenario, cfg: Optional[InferenceConfig] = None, decoder=None
              ) -> Tuple[MOTIPTracker, List[List[TrackOutput]], IDMetrics]:
    cfg = cfg or InferenceConfig()
    decoder = decoder or AppearanceMockDecoder(cfg.num_labels)
    trk = MOTIPTracker(decoder, cfg)
    per_frame, assign = [], []
    for f, fr in enumerate(sc.frames):
        outs = trk.step(f, fr.boxes, fr.scores, fr.feats)
        per_frame.append(outs)
        assign.append({int(fr.gt_ids[o.det_index]): o.output_id for o in outs})
    trk.finish(sc.num_frames)
    return trk, per_frame, evaluate_ids(assign)


def run_sort(sc: Scenario, **kw) -> Tuple[List[List[Tuple[int, object, int]]], IDMetrics]:
    trk = SortTracker(**kw)
    per_frame, assign = [], []
    for fr in sc.frames:
        outs = trk.step(fr.boxes, fr.scores)
        per_frame.append(outs)
        assign.append({int(fr.gt_ids[d]): tid for tid, _, d in outs})
    return per_frame, evaluate_ids(assign)
