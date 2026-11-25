"""
id_decoder.py  -  ID Dictionary + Tracklet + ID Decoder của MOTIP (Gao et al., CVPR 2025, Sec 3.2)
Người phụ trách: G4-TV3 Đào Trung Dũng.

Luồng dữ liệu (một frame):
    DecoderBatch của TV2 (M quỹ đạo x T token)  ->  tracklet token  tau = concat(f, i_k)   (2C chiều)
    detection hiện tại  (N, C)                  ->  tau = concat(f, i_spec)                (2C chiều)
    ID Decoder (self-attn giữa detection + cross-attn tới lịch sử)  ->  head K+1  ->  softmax
    bỏ cột newborn  ->  (N, K) xác suất  ->  TV2 (tracking_inference.py)

Hai lớp cần phân biệt:
    MOTIPIDDecoder      nn.Module, dùng khi HUẤN LUYỆN (forward / forward_batched, ra logit K+1).
    IDDecoderForTracker bọc ngoài, khớp hợp đồng IDDecoder của TV2: .num_labels và
                        __call__(batch, det_feats) -> np.ndarray (N, K) đã softmax.

Quy ước nhãn: 0..K-1 là nhãn thường (TV2 dùng 0..K-1, bài báo đánh 1..K), chỉ số K là token newborn.

LƯU Ý TRUNG THỰC: đây là bản cài đặt theo MÔ TẢ trong bài báo, KHÔNG phải mã chính thức. Số layer/head,
cách mã hoá vị trí tương đối (ở đây: embedding học được theo dt, cộng vào token lịch sử) là lựa chọn của
nhóm; trọng số của checkpoint chính thức MCG-NJU/MOTIP sẽ KHÔNG nạp thẳng vào lớp này.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEWBORN_LOGIT = 20.0   # logit cố định cho newborn khi không có lịch sử nào để đọc


@dataclass
class IDDecoderConfig:
    num_labels: int = 50      # K  (bài báo: 50)
    feat_dim: int = 256       # C  = số chiều object embedding của DETR (TV1: D=256)
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.0
    max_dt: int = 128         # dt lớn hơn bị cắt về giá trị này (T lớn nhất của bài báo là 59)


# --------------------------------------------------------------------------- #
class IDDictionary(nn.Module):
    """K+1 embedding học được: K nhãn thường + 1 token đặc biệt (newborn) ở chỉ số K."""

    def __init__(self, num_ids: int = 50, embedding_dim: int = 256):
        super().__init__()
        self.num_ids, self.embedding_dim = num_ids, embedding_dim
        self.newborn_id = num_ids
        self.id_embedding = nn.Embedding(num_ids + 1, embedding_dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """labels: tensor số nguyên bất kỳ shape (...) trong [0, K]  ->  (..., C)."""
        return self.id_embedding(labels)

    def newborn(self, batch: int, n: int) -> torch.Tensor:
        """Token newborn lặp lại thành (batch, n, C)."""
        return self.id_embedding.weight[self.newborn_id].expand(batch, n, -1)

    # tương thích với id_embedding_demo.py cũ
    def get_embedding(self, object_id: int) -> torch.Tensor:
        if object_id < 0 or object_id >= self.num_ids:
            raise ValueError("ID khong hop le!")
        return self.id_embedding(torch.tensor(object_id, dtype=torch.long))

    def get_newborn_embedding(self) -> torch.Tensor:
        return self.id_embedding(torch.tensor(self.newborn_id, dtype=torch.long))

    def get_all_embeddings(self) -> torch.Tensor:
        return self.id_embedding.weight


# --------------------------------------------------------------------------- #
class MOTIPIDDecoder(nn.Module):
    def __init__(self, cfg: Optional[IDDecoderConfig] = None, **kw):
        super().__init__()
        self.cfg = cfg or IDDecoderConfig(**kw)
        c = self.cfg
        self.dictionary = IDDictionary(c.num_labels, c.feat_dim)
        self.token_proj = nn.Linear(2 * c.feat_dim, c.d_model)      # 2C -> d_model, dùng chung cho lịch sử và truy vấn
        self.dt_embed = nn.Embedding(c.max_dt + 1, c.d_model)        # mã hoá khoảng cách thời gian tương đối
        layer = nn.TransformerDecoderLayer(c.d_model, c.num_heads, c.ffn_dim, c.dropout,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, c.num_layers, norm=nn.LayerNorm(c.d_model))
        self.head = nn.Linear(c.d_model, c.num_labels + 1)           # K nhãn + newborn

    # ------------------------------------------------------------------ #
    def forward_batched(self, hist_feats, hist_labels, hist_dt, hist_mask, det_feats, det_mask=None):
        """
        hist_feats (B,M,T,C) float | hist_labels (B,M,T) long in [0,K) | hist_dt (B,M,T) long
        hist_mask  (B,M,T) bool, True = token thật (cùng quy ước DecoderBatch.mask của TV2)
        det_feats  (B,N,C) float   | det_mask (B,N) bool, True = detection thật (None = tất cả thật)
        Trả về logit (B,N,K+1).
        """
        c = self.cfg
        B, M, T, C = hist_feats.shape
        N = det_feats.shape[1]
        if C != c.feat_dim or det_feats.shape[-1] != c.feat_dim:
            raise ValueError(f"feature phải có {c.feat_dim} chiều (C của detector), nhận {C}/{det_feats.shape[-1]}")

        tok = self.token_proj(torch.cat([hist_feats, self.dictionary(hist_labels)], dim=-1))
        tok = tok + self.dt_embed(hist_dt.clamp(0, c.max_dt))
        memory = tok.reshape(B, M * T, -1)
        mem_pad = ~hist_mask.bool().reshape(B, M * T)                # PyTorch: True = BỎ QUA (ngược với mask của TV2)

        empty = mem_pad.all(dim=1)                                   # mẫu không có token lịch sử nào
        if bool(empty.any()):
            mem_pad = mem_pad.clone()
            mem_pad[empty, 0] = False                                # tránh softmax toàn -inf -> NaN; kết quả bị ghi đè bên dưới

        query = self.token_proj(torch.cat([det_feats, self.dictionary.newborn(B, N)], dim=-1))
        tgt_pad = None if det_mask is None else ~det_mask.bool()
        out = self.decoder(query, memory, tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=mem_pad)
        logits = self.head(out)

        if bool(empty.any()):                                        # không có lịch sử -> chỉ có thể là newborn
            fixed = torch.zeros(c.num_labels + 1, device=logits.device, dtype=logits.dtype)
            fixed[c.num_labels] = NEWBORN_LOGIT
            logits = torch.where(empty[:, None, None], fixed, logits)
        return logits

    def forward(self, hist_feats, hist_labels, hist_dt, hist_mask, det_feats):
        """Một frame, không có chiều batch: hist_* (M,T,...), det_feats (N,C) -> logit (N,K+1)."""
        return self.forward_batched(hist_feats[None], hist_labels[None], hist_dt[None],
                                    hist_mask[None], det_feats[None])[0]


def id_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """L_id: cross-entropy K+1 lớp (Sec 3.3). logits (B,N,K+1), targets (B,N) trong [0,K], pad = ignore_index."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=ignore_index)


# --------------------------------------------------------------------------- #
class IDDecoderForTracker:
    """Khớp hợp đồng IDDecoder của TV2 (tracking_inference.py): num_labels + __call__(batch, det_feats)."""

    def __init__(self, model: MOTIPIDDecoder, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.num_labels = model.cfg.num_labels
        self.device = device

    def __call__(self, batch, det_feats: np.ndarray) -> np.ndarray:
        K, C = self.num_labels, self.model.cfg.feat_dim
        det_feats = np.asarray(det_feats, dtype=np.float32)
        if det_feats.ndim != 2 or det_feats.shape[1] != C:
            raise ValueError(f"det_feats phải có shape (N,{C}), nhận {det_feats.shape}")
        if len(det_feats) == 0:
            return np.zeros((0, K), np.float32)
        if batch.labels.size and (batch.labels.min() < 0 or batch.labels.max() >= K):
            raise ValueError("nhãn lịch sử nằm ngoài [0,K)")
        t = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)
        with torch.inference_mode():
            logits = self.model(t(batch.feats, torch.float32), t(batch.labels, torch.long),
                                t(batch.dt, torch.long), t(batch.mask, torch.bool), t(det_feats, torch.float32))
            probs = torch.softmax(logits.float(), dim=-1)[:, :K]    # bỏ cột newborn: phần còn lại = xác suất newborn
        return probs.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #
def save_id_decoder(model: MOTIPIDDecoder, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"kind": "tv3_id_decoder", "config": asdict(model.cfg), "model": model.state_dict()}, path)


def load_id_decoder(path: str, device: str = "cpu") -> IDDecoderForTracker:
    """Nạp checkpoint do save_id_decoder tạo ra; strict=True, không che lỗi thiếu/thừa khoá."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)      # chỉ nạp file tin cậy
    if not isinstance(ckpt, dict) or ckpt.get("kind") != "tv3_id_decoder":
        raise ValueError("Không phải checkpoint của TV3 (thiếu kind='tv3_id_decoder'). "
                         "Checkpoint MOTIP chính thức: dùng inspect_checkpoint() để xem khoá trước.")
    model = MOTIPIDDecoder(IDDecoderConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"], strict=True)
    return IDDecoderForTracker(model, device)


def inspect_checkpoint(path: str, skip_prefix: str = "detr.") -> None:
    """In khoá + shape của checkpoint (vd. MOTIP chính thức), bỏ qua phần detector. Dùng để đối chiếu kiến trúc."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    for k, v in state.items():
        if not k.startswith(skip_prefix):
            print(f"{k:80s} {tuple(v.shape)}")
