"""
synthetic_data.py  -  Dữ liệu tổng hợp để huấn luyện/kiểm thử ID Decoder (chỉ cần numpy).

Mô phỏng đúng bài toán "in-context ID prediction" (MOTIP, Sec 3.1):
  - M quỹ đạo, mỗi quỹ đạo được gán NGẪU NHIÊN một nhãn trong 0..K-1.
  - Mỗi quỹ đạo có T token lịch sử (feature + nhãn của quỹ đạo).
  - Frame hiện tại có N detection: mỗi detection thuộc một quỹ đạo (target = nhãn của quỹ đạo đó)
    hoặc là newborn (target = K, chỉ số của token đặc biệt).
Trajectory augmentation (Sec 3.3): random occlusion (bỏ token) và random switch (đổi ID token
của hai quỹ đạo trong cùng một frame lịch sử). Cách diễn giải cụ thể ở hai hàm tương ứng là LỰA CHỌN
CÀI ĐẶT của nhóm, bài báo không nêu chi tiết.

Chỉ kiểm chứng LOGIC học ID, không phản ánh độ chính xác trên dữ liệu thật (DanceTrack...).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

IGNORE = -100


@dataclass
class SynthConfig:
    num_labels: int = 50          # K
    feat_dim: int = 256           # C  (phải bằng C của detector TV1)
    history_len: int = 29         # T
    min_tracks: int = 2
    max_tracks: int = 12
    feat_noise: float = 0.5       # độ lớn nhiễu feature (chuẩn L2 của vector nhiễu)
    similarity: float = 0.5       # 0 = các đối tượng khác hẳn nhau, ->1 = rất giống nhau (khó hơn)
    p_present: float = 0.9        # xác suất một quỹ đạo có detection ở frame hiện tại
    max_newborn: int = 2
    lambda_occ: float = 0.5
    lambda_sw: float = 0.5


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _appearance(rng, n: int, cfg: SynthConfig, common: np.ndarray) -> np.ndarray:
    own = _unit(rng.standard_normal((n, cfg.feat_dim)))
    return _unit(cfg.similarity * common[None] + (1.0 - cfg.similarity) * own)


def _noise(rng, shape, cfg: SynthConfig) -> np.ndarray:
    return rng.standard_normal(shape) * (cfg.feat_noise / np.sqrt(cfg.feat_dim))


def random_occlusion(rng, mask: np.ndarray, lam: float) -> None:
    """Mỗi quỹ đạo, với xác suất lam, bị bỏ token với tỉ lệ ngẫu nhiên r~U(0,1). Luôn giữ lại >=1 token."""
    for m in range(mask.shape[0]):
        if rng.random() < lam:
            drop = rng.random(mask.shape[1]) < rng.random()
            if drop.all():
                drop[-1] = False
            mask[m] = ~drop


def random_switch(rng, hist_labels: np.ndarray, lam: float) -> None:
    """Với xác suất lam: chọn 1-2 frame lịch sử, đổi nhãn của hai quỹ đạo ngẫu nhiên tại frame đó
    (mô phỏng lúc suy luận gán nhầm ID). Nhãn mục tiêu của frame hiện tại KHÔNG đổi."""
    M, T = hist_labels.shape
    if M < 2 or rng.random() >= lam:
        return
    for _ in range(int(rng.integers(1, 3))):
        t = int(rng.integers(0, T))
        a, b = rng.choice(M, size=2, replace=False)
        hist_labels[a, t], hist_labels[b, t] = hist_labels[b, t], hist_labels[a, t]


def make_sample(rng: np.random.Generator, cfg: SynthConfig, augment: bool = True) -> Dict[str, np.ndarray]:
    K, T, C = cfg.num_labels, cfg.history_len, cfg.feat_dim
    M = int(rng.integers(cfg.min_tracks, min(cfg.max_tracks, K) + 1))
    labels = rng.permutation(K)[:M].astype(np.int64)            # nhãn ngẫu nhiên: ý tưởng cốt lõi của MOTIP
    common = _unit(rng.standard_normal(C))
    app = _appearance(rng, M, cfg, common)

    hist_feats = (app[:, None, :] + _noise(rng, (M, T, C), cfg)).astype(np.float32)
    hist_labels = np.repeat(labels[:, None], T, axis=1)
    hist_dt = np.tile(np.arange(T, 0, -1, dtype=np.int64), (M, 1))   # token cuối có dt=1, đầu có dt=T
    hist_mask = np.ones((M, T), bool)
    if augment:
        random_occlusion(rng, hist_mask, cfg.lambda_occ)
        random_switch(rng, hist_labels, cfg.lambda_sw)

    present = rng.random(M) < cfg.p_present
    det_feats = [app[present] + _noise(rng, (int(present.sum()), C), cfg)]
    targets = [labels[present]]
    n_new = int(rng.integers(0, cfg.max_newborn + 1))
    if present.sum() + n_new == 0:
        n_new = 1
    if n_new:
        det_feats.append(_appearance(rng, n_new, cfg, common) + _noise(rng, (n_new, C), cfg))
        targets.append(np.full(n_new, K, dtype=np.int64))        # K = token đặc biệt "newborn"
    det_feats = np.concatenate(det_feats).astype(np.float32)
    targets = np.concatenate(targets)
    perm = rng.permutation(len(targets))                          # thứ tự detection không mang thông tin
    return dict(hist_feats=hist_feats, hist_labels=hist_labels, hist_dt=hist_dt, hist_mask=hist_mask,
                det_feats=det_feats[perm], targets=targets[perm])


def collate(samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Pad về (B, Mmax, T, C) và (B, Nmax, C). Chỗ pad: hist_mask=False, det_mask=False, target=IGNORE."""
    B = len(samples)
    Mx = max(s["hist_feats"].shape[0] for s in samples)
    Nx = max(s["det_feats"].shape[0] for s in samples)
    T, C = samples[0]["hist_feats"].shape[1:]
    out = dict(
        hist_feats=np.zeros((B, Mx, T, C), np.float32), hist_labels=np.zeros((B, Mx, T), np.int64),
        hist_dt=np.ones((B, Mx, T), np.int64), hist_mask=np.zeros((B, Mx, T), bool),
        det_feats=np.zeros((B, Nx, C), np.float32), det_mask=np.zeros((B, Nx), bool),
        targets=np.full((B, Nx), IGNORE, np.int64))
    for i, s in enumerate(samples):
        m, n = s["hist_feats"].shape[0], s["det_feats"].shape[0]
        out["hist_feats"][i, :m] = s["hist_feats"]
        out["hist_labels"][i, :m] = s["hist_labels"]
        out["hist_dt"][i, :m] = s["hist_dt"]
        out["hist_mask"][i, :m] = s["hist_mask"]
        out["det_feats"][i, :n] = s["det_feats"]
        out["det_mask"][i, :n] = True
        out["targets"][i, :n] = s["targets"]
    return out
