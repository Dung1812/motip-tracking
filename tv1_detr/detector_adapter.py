"""MOTIP detector adapter: bbox/score/object embedding aligned by query index.

Modes:
  --mode self-test       Offline synthetic tensor checks (NO pretrained model).
  --mode motip           Load official MCG-NJU/MOTIP repository + .pth checkpoint.

This file is standalone and does not modify the user's existing stage-1 source.
Official MOTIP detector raw outputs: pred_logits, pred_boxes, outputs.
The `outputs` key contains per-query decoder object features, NOT ID embeddings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


@dataclass
class DetectionBatch:
    boxes_xyxy: torch.Tensor       # [N,4] absolute pixel coordinates
    scores: torch.Tensor           # [N]
    labels: torch.Tensor           # [N] class indices (MOTIP DanceTrack: 0 = person)
    embeddings: torch.Tensor       # [N,D] decoder object features
    query_indices: torch.Tensor    # [N] indices into all DETR queries
    image_size_hw: tuple[int, int]
    raw_shapes: dict[str, list[int]]

    def validate(self) -> None:
        n = int(self.scores.numel())
        assert self.boxes_xyxy.shape == (n, 4), self.boxes_xyxy.shape
        assert self.labels.shape == (n,), self.labels.shape
        assert self.embeddings.ndim == 2 and self.embeddings.shape[0] == n
        assert self.query_indices.shape == (n,), self.query_indices.shape
        assert bool(torch.isfinite(self.boxes_xyxy).all())
        assert bool(torch.isfinite(self.scores).all())
        assert bool(torch.isfinite(self.embeddings).all())
        assert bool(((self.scores >= 0) & (self.scores <= 1)).all())
        if n:
            assert bool((self.boxes_xyxy[:, 2] > self.boxes_xyxy[:, 0]).all())
            assert bool((self.boxes_xyxy[:, 3] > self.boxes_xyxy[:, 1]).all())

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "image_size_hw": list(self.image_size_hw),
            "raw_shapes": self.raw_shapes,
            "num_detections": len(self.scores),
            "embedding_dim": int(self.embeddings.shape[-1]),
            "box_format": "xyxy_pixel",
            "embedding_kind": "DETR_decoder_object_feature_NOT_ID_embedding",
            "detections": [
                {
                    "box_xyxy": [round(float(v), 3) for v in self.boxes_xyxy[i].tolist()],
                    "score": round(float(self.scores[i].item()), 6),
                    "label": int(self.labels[i].item()),
                    "query_index": int(self.query_indices[i].item()),
                }
                for i in range(len(self.scores))
            ],
        }


class DetectorAdapter:
    """Convert the official DETR-format output for ONE frame to aligned detections.

    Raw tensor shapes: pred_logits [1,Q,C], pred_boxes [1,Q,4], outputs [1,Q,D].
    MOTIP Deformable DETR uses sigmoid per class, not vanilla DETR softmax.
    bbox is normalized [cx,cy,w,h] and is converted to pixel [x1,y1,x2,y2].
    """

    def __init__(self, threshold: float = 0.30, max_detections: int = 100,
                 class_index: int | None = 0, score_type: str = "sigmoid") -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0,1]")
        if max_detections < 1:
            raise ValueError("max_detections must be >= 1")
        if score_type not in {"sigmoid", "softmax_background_last"}:
            raise ValueError("Unsupported score_type")
        self.threshold = threshold
        self.max_detections = max_detections
        self.class_index = class_index
        self.score_type = score_type

    @torch.no_grad()
    def convert(self, raw: Mapping[str, torch.Tensor], image_size_hw: tuple[int, int]) -> DetectionBatch:
        for key in ("pred_logits", "pred_boxes", "outputs"):
            if key not in raw:
                raise KeyError(f"Missing '{key}': this is required for bbox + embedding alignment")

        logits = torch.as_tensor(raw["pred_logits"]).detach().cpu().float()
        boxes = torch.as_tensor(raw["pred_boxes"]).detach().cpu().float()
        features = torch.as_tensor(raw["outputs"]).detach().cpu().float()
        if logits.ndim != 3 or boxes.ndim != 3 or features.ndim != 3:
            raise ValueError("Expected all three raw tensors to have shape [B,Q,*]")
        if logits.shape[0] != 1 or boxes.shape[0] != 1 or features.shape[0] != 1:
            raise ValueError("Adapter handles one frame [B=1]; call once per frame")
        q = logits.shape[1]
        if boxes.shape != (1, q, 4) or features.shape[1] != q:
            raise ValueError("Query dimension Q must match across logits, boxes and outputs")
        if features.shape[-1] < 1:
            raise ValueError("Missing decoder feature dimension")
        if not all(bool(torch.isfinite(t).all()) for t in (logits, boxes, features)):
            raise ValueError("Raw outputs contain NaN or infinity")

        h, w = map(int, image_size_hw)
        if w < 1 or h < 1:
            raise ValueError("Invalid image size")

        if self.score_type == "sigmoid":
            prob = logits[0].sigmoid()  # official MOTIP / Deformable DETR
        else:
            if logits.shape[-1] < 2:
                raise ValueError("softmax background requires last no-object class")
            prob = logits[0].softmax(dim=-1)[:, :-1]

        if self.class_index is not None:
            if not 0 <= self.class_index < prob.shape[-1]:
                raise ValueError(f"class_index={self.class_index} outside [0,{prob.shape[-1]-1}]")
            score = prob[:, self.class_index]
            label = torch.full((q,), self.class_index, dtype=torch.long)
        else:
            score, label = prob.max(dim=-1)

        # Select from ONE shared set of query indices: bounding boxes, scores and features stay aligned.
        indices = torch.nonzero(score >= self.threshold, as_tuple=True)[0]
        indices = indices[torch.argsort(score[indices], descending=True)[:self.max_detections]]

        b = boxes[0, indices]  # normalized cx,cy,w,h
        cx, cy, bw, bh = b.unbind(-1)
        px = torch.stack(((cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h), dim=-1)
        px[:, [0, 2]] = px[:, [0, 2]].clamp(0, w)
        px[:, [1, 3]] = px[:, [1, 3]].clamp(0, h)
        valid = (px[:, 2] > px[:, 0]) & (px[:, 3] > px[:, 1])
        indices = indices[valid]
        px = px[valid]

        batch = DetectionBatch(
            boxes_xyxy=px.contiguous(), scores=score[indices].contiguous(),
            labels=label[indices].contiguous(),
            embeddings=features[0, indices].contiguous(),
            query_indices=indices.contiguous(), image_size_hw=(h, w),
            raw_shapes={k: list(torch.as_tensor(raw[k]).shape)
                        for k in ("pred_logits", "pred_boxes", "outputs")},
        )
        batch.validate()
        return batch


def save_results(batch: DetectionBatch, image_bgr: np.ndarray | None,
                 output_prefix: Path, make_image: bool = True) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    (output_prefix.with_suffix(".json")).write_text(
        json.dumps(batch.to_jsonable(), ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        boxes_xyxy=batch.boxes_xyxy.numpy(), scores=batch.scores.numpy(),
        labels=batch.labels.numpy(), embeddings=batch.embeddings.numpy(),
        query_indices=batch.query_indices.numpy(),
    )
    if make_image and image_bgr is not None:
        import cv2
        canvas = image_bgr.copy()
        for i, (box, score, label) in enumerate(zip(batch.boxes_xyxy, batch.scores, batch.labels)):
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, f"cls{int(label)} {float(score):.2f} q{int(batch.query_indices[i])}",
                        (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(output_prefix.with_suffix(".jpg")), canvas)


def make_nested_frame(frame_bgr: np.ndarray, device: torch.device, max_long_side: int = 1440):
    """ImageNet normalization, preserving aspect ratio; no padding (B=1).

    This is a standalone MOTIP-compatible preprocessing demonstration. Compare
    with the data transforms in your exact MOTIP commit before benchmarking.
    """
    import cv2
    from utils.nested_tensor import NestedTensor  # official MOTIP, available after repo path setup

    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_long_side / max(h, w))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)))
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = (x - mean) / std
    image = x.unsqueeze(0).to(device)
    mask = torch.zeros((1, image.shape[-2], image.shape[-1]), dtype=torch.bool, device=device)
    return NestedTensor(image, mask)


def load_official_motip(repo: Path, config_path: Path, checkpoint_path: Path, device: torch.device):
    """Build exact official architecture from official config, load weights strictly.

    Works only after installing/compiling the external MOTIP repository dependencies.
    We never silently ignore missing/unexpected checkpoint keys.
    """
    repo = repo.resolve()
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    if not (repo / "models" / "motip" / "__init__.py").is_file():
        raise FileNotFoundError(f"Not a checkout of MCG-NJU/MOTIP: {repo}")
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Missing config or checkpoint; download a MATCHING pair from the MOTIP model zoo")

    sys.path.insert(0, str(repo))
    previous_cwd = Path.cwd()
    try:
        os.chdir(repo)  # official SUPER_CONFIG_PATH typically starts at ./configs/...
        from utils.misc import yaml_to_dict
        from configs.util import load_super_config
        from models.motip import build as build_motip
        from models.misc import load_detr_pretrain

        cfg = yaml_to_dict(str(config_path))
        cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
        cfg["DEVICE"] = str(device)
        model, _ = build_motip(cfg)
        # Trusted checkpoints only. PyTorch >=2.4 supports weights_only=True.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise ValueError("Expected a MOTIP checkpoint dictionary with key 'model'")
        state = ckpt["model"]
        if not isinstance(state, dict):
            raise ValueError("checkpoint['model'] must be a state_dict")
        if any(k.startswith("detr.") for k in state):
            model.load_state_dict(state, strict=True)
            checkpoint_kind = "full_MOTIP"
        elif any(k.startswith("bbox_embed.") for k in state):
            # Official COCO detector pretrain may have 91 classes, while DanceTrack has 1.
            load_detr_pretrain(model=model, pretrain_path=str(checkpoint_path),
                               num_classes=int(cfg["NUM_CLASSES"]))
            checkpoint_kind = "DETR_pretrain"
        else:
            raise RuntimeError("Unknown state_dict keys: check model type/version and matching YAML")
        model.to(device).eval()
        return model, cfg, checkpoint_kind
    finally:
        os.chdir(previous_cwd)


def run_motip(args: argparse.Namespace) -> None:
    import cv2
    device = torch.device(args.device)
    model, cfg, ckpt_kind = load_official_motip(
        Path(args.repo), Path(args.config), Path(args.checkpoint), device)
    print(f"Checkpoint loaded strictly: {ckpt_kind}")
    print(f"Model config: queries={cfg['DETR_NUM_QUERIES']} classes={cfg['NUM_CLASSES']} "
          f"hidden={cfg['DETR_HIDDEN_DIM']}")
    adapter = DetectorAdapter(threshold=args.threshold,
                              max_detections=args.max_detections,
                              class_index=args.class_index, score_type="sigmoid")

    src = int(args.source) if args.source.isdecimal() else args.source
    still = isinstance(src, str) and Path(src).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    cap = None
    if still:
        frame = cv2.imread(str(src))
        if frame is None:
            raise RuntimeError(f"Cannot open image: {src}")
        frames = [(0, frame)]
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video/webcam: {src}")
        def iterate():
            i = 0
            while i < args.limit:
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, frame
                i += 1
        frames = iterate()

    try:
        for i, frame in frames:
            nested = make_nested_frame(frame, device, args.max_long_side)
            with torch.inference_mode():
                raw = model(frames=nested, part="detr")
            batch = adapter.convert(raw, image_size_hw=frame.shape[:2])
            prefix = Path(args.out_dir) / f"frame_{i:05d}"
            save_results(batch, frame, prefix)
            print(f"frame={i}: raw={batch.raw_shapes}, kept={len(batch.scores)}, "
                  f"bbox={list(batch.boxes_xyxy.shape)}, score={list(batch.scores.shape)}, "
                  f"embedding={list(batch.embeddings.shape)}")
    finally:
        if cap is not None:
            cap.release()


def self_test(out_dir: Path) -> None:
    """Fully offline test; deliberately fabricated data, never call this real inference."""
    logits = torch.full((1, 300, 1), -10.0)
    logits[0, 2, 0] = 3.0
    logits[0, 7, 0] = 1.5
    boxes = torch.full((1, 300, 4), 0.5)
    boxes[0, 2] = torch.tensor([0.5, 0.5, 0.2, 0.5])
    boxes[0, 7] = torch.tensor([0.2, 0.2, 0.1, 0.2])
    features = torch.zeros(1, 300, 256)
    features[0, 2, 0] = 2.0
    features[0, 7, 0] = 7.0
    raw = {"pred_logits": logits, "pred_boxes": boxes, "outputs": features}
    batch = DetectorAdapter(threshold=0.7, max_detections=10).convert(raw, (480, 640))
    assert batch.query_indices.tolist() == [2, 7]
    assert batch.embeddings[:, 0].tolist() == [2.0, 7.0]
    assert batch.boxes_xyxy[0].tolist() == [256, 120, 384, 360]
    assert batch.boxes_xyxy.shape == (2, 4)
    assert batch.embeddings.shape == (2, 256)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results(batch, None, out_dir / "synthetic_only", make_image=False)
    print("PASS: bbox, score, feature and query_index remain aligned")
    print("PASS: raw [1,300,1], [1,300,4], [1,300,256] -> N=2, D=256")
    print(f"Synthetic test files: {out_dir.resolve()} (NOT a trained detector result)")


def main() -> None:
    ap = argparse.ArgumentParser(description="MOTIP detector adapter and shape inspector")
    ap.add_argument("--mode", choices=["self-test", "motip"], default="self-test")
    ap.add_argument("--repo", default="external/MOTIP", help="Path to official cloned MOTIP repo")
    ap.add_argument("--config", help="Matching official MOTIP YAML config")
    ap.add_argument("--checkpoint", help="Matching official .pth checkpoint")
    ap.add_argument("--source", default="0", help="Image, video file, or webcam index 0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--class-index", type=int, default=0, help="MOTIP DanceTrack class 0=person")
    ap.add_argument("--max-detections", type=int, default=100)
    ap.add_argument("--limit", type=int, default=3, help="Maximum video/webcam frames")
    ap.add_argument("--max-long-side", type=int, default=1440)
    ap.add_argument("--out-dir", default="outputs/stage2_detector")
    args = ap.parse_args()
    if args.mode == "self-test":
        self_test(Path(args.out_dir))
    else:
        if not args.config or not args.checkpoint:
            ap.error("--mode motip requires --repo, --config, --checkpoint")
        run_motip(args)


if __name__ == "__main__":
    main()
