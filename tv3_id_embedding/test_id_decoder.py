"""Chạy:  python -m pytest -q test_id_decoder.py      (cần torch, numpy, pytest)
Test tích hợp với TV2 tự bỏ qua nếu không tìm thấy thư mục ../tv2_mot_baseline."""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from id_decoder import (IDDecoderConfig, IDDecoderForTracker, IDDictionary, MOTIPIDDecoder, id_loss,
                        load_id_decoder, save_id_decoder)
from synthetic_data import IGNORE, SynthConfig, collate, make_sample

K, C, T = 10, 16, 6
SMALL = dict(num_labels=K, feat_dim=C, d_model=32, num_heads=4, num_layers=2, ffn_dim=64)


@dataclass
class FakeBatch:   # cùng trường với DecoderBatch của TV2 (track_memory.py)
    feats: np.ndarray
    labels: np.ndarray
    dt: np.ndarray
    mask: np.ndarray


def fake_batch(M=3, rng=None):
    rng = rng or np.random.default_rng(0)
    mask = np.ones((M, T), bool)
    mask[0, :3] = False                                   # quỹ đạo 0 chỉ có 3 token cuối (TV2 pad về cuối)
    mask = mask[:, ::-1].copy()                           # token thật ở đầu, pad ở cuối, như build_batch
    return FakeBatch(rng.standard_normal((M, T, C)).astype(np.float32),
                     np.tile(np.arange(M)[:, None], (1, T)).astype(np.int64),
                     np.tile(np.arange(T, 0, -1), (M, 1)).astype(np.int64), mask)


def make_model():
    torch.manual_seed(0)
    return MOTIPIDDecoder(IDDecoderConfig(**SMALL)).eval()


def test_output_shape_and_probabilities():
    dec = IDDecoderForTracker(make_model())
    p = dec(fake_batch(), np.random.default_rng(1).standard_normal((4, C)).astype(np.float32))
    assert dec.num_labels == K and p.shape == (4, K) and p.dtype == np.float32
    assert (p >= 0).all() and (p.sum(1) <= 1 + 1e-5).all()          # phần còn lại là xác suất newborn


def test_zero_detections_returns_empty():
    p = IDDecoderForTracker(make_model())(fake_batch(), np.zeros((0, C), np.float32))
    assert p.shape == (0, K)


def test_padded_tokens_do_not_change_output():
    """Mask của TV2 (True = thật) phải được đổi đúng sang key_padding_mask (True = bỏ qua)."""
    model, b = make_model(), fake_batch()
    det = torch.randn(3, C)
    args = lambda feats: (torch.as_tensor(feats), torch.as_tensor(b.labels), torch.as_tensor(b.dt),
                          torch.as_tensor(b.mask), det)
    with torch.no_grad():
        a = model(*args(b.feats))
        f2 = b.feats.copy()
        f2[~b.mask] = 1e3 * np.random.default_rng(5).standard_normal(f2[~b.mask].shape)   # phá nội dung ở chỗ pad
        c = model(*args(f2))
        f3 = b.feats.copy()
        f3[b.mask] += 1.0                                                                  # đổi token thật -> phải khác
        d = model(*args(f3))
    assert torch.allclose(a, c, atol=1e-5)
    assert not torch.allclose(a, d, atol=1e-3)


def test_empty_history_predicts_newborn():
    model = make_model()
    b = fake_batch()
    with torch.no_grad():
        logits = model(torch.as_tensor(b.feats), torch.as_tensor(b.labels), torch.as_tensor(b.dt),
                       torch.zeros(b.mask.shape, dtype=torch.bool), torch.randn(2, C))
    assert torch.isfinite(logits).all() and (logits.argmax(-1) == K).all()


def test_wrong_feature_dim_and_labels_are_rejected():
    dec = IDDecoderForTracker(make_model())
    with pytest.raises(ValueError):
        dec(fake_batch(), np.zeros((2, C + 1), np.float32))
    bad = fake_batch()
    bad.labels[0, 0] = K
    with pytest.raises(ValueError):
        dec(bad, np.zeros((2, C), np.float32))


def test_dictionary_has_k_plus_1_words_and_gradient():
    d = IDDictionary(5, 8)
    assert d.get_all_embeddings().shape == (6, 8) and d.newborn_id == 5
    d(torch.tensor([1, 2, 5])).sum().backward()
    g = d.id_embedding.weight.grad
    assert g[1].abs().sum() > 0 and g[5].abs().sum() > 0 and g[3].abs().sum() == 0


def test_gradients_reach_all_parts():
    model = MOTIPIDDecoder(IDDecoderConfig(**SMALL)).train()
    sc = SynthConfig(num_labels=K, feat_dim=C, history_len=T, max_tracks=4)
    rng = np.random.default_rng(0)
    b = collate([make_sample(rng, sc) for _ in range(4)])
    t = lambda k, dt: torch.as_tensor(b[k], dtype=dt)
    loss = id_loss(model.forward_batched(t("hist_feats", torch.float32), t("hist_labels", torch.long),
                                         t("hist_dt", torch.long), t("hist_mask", torch.bool),
                                         t("det_feats", torch.float32), t("det_mask", torch.bool)),
                   t("targets", torch.long), IGNORE)
    loss.backward()
    for name in ("dictionary.id_embedding.weight", "dt_embed.weight", "token_proj.weight", "head.weight"):
        assert dict(model.named_parameters())[name].grad.abs().sum() > 0, name


def test_model_learns_in_context_id_prediction():
    """Quan trọng nhất: sau huấn luyện ngắn trên dữ liệu tổng hợp, độ chính xác phải vượt xa ngẫu nhiên."""
    torch.manual_seed(0)
    sc = SynthConfig(num_labels=K, feat_dim=C, history_len=T, min_tracks=2, max_tracks=5, similarity=0.0,
                     feat_noise=0.3)
    model = MOTIPIDDecoder(IDDecoderConfig(**SMALL)).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)

    def run(batch, train):
        t = lambda k, dt: torch.as_tensor(batch[k], dtype=dt)
        logits = model.forward_batched(t("hist_feats", torch.float32), t("hist_labels", torch.long),
                                       t("hist_dt", torch.long), t("hist_mask", torch.bool),
                                       t("det_feats", torch.float32), t("det_mask", torch.bool))
        return logits, t("targets", torch.long)

    for _ in range(300):
        logits, y = run(collate([make_sample(rng, sc, augment=False) for _ in range(16)]), True)
        loss = id_loss(logits, y, IGNORE)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits, y = run(collate([make_sample(np.random.default_rng(99), sc, augment=False) for _ in range(64)]), False)
    valid = y != IGNORE
    acc = ((logits.argmax(-1) == y) & valid).sum().item() / valid.sum().item()
    assert acc > 0.8, f"acc={acc:.3f}"          # ngẫu nhiên ~ 1/(K+1) ≈ 0.09


def test_checkpoint_roundtrip(tmp_path):
    model = make_model()
    p = str(tmp_path / "m.pth")
    save_id_decoder(model, p)
    dec = load_id_decoder(p)
    b, det = fake_batch(), np.random.default_rng(2).standard_normal((3, C)).astype(np.float32)
    assert np.allclose(IDDecoderForTracker(model)(b, det), dec(b, det), atol=1e-6)
    torch.save({"model": model.state_dict()}, p)
    with pytest.raises(ValueError):
        load_id_decoder(p)


# ----------------------------------------------------------------- tích hợp với TV2 (nếu có trong repo)
TV2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tv2_mot_baseline")


@pytest.mark.skipif(not os.path.isdir(TV2), reason="không thấy tv2_mot_baseline")
def test_plugs_into_tv2_tracker_without_shape_errors():
    sys.path.insert(0, TV2)
    import scenarios as S
    from tracking_inference import InferenceConfig, MOTIPTracker
    torch.manual_seed(0)
    cfg = InferenceConfig(num_labels=K)
    model = MOTIPIDDecoder(IDDecoderConfig(num_labels=K, feat_dim=S.FEAT_DIM, d_model=32, num_heads=4,
                                           num_layers=2, ffn_dim=64))
    trk = MOTIPTracker(IDDecoderForTracker(model), cfg)       # decoder chưa huấn luyện: chỉ kiểm tra hợp đồng
    sc = S.parallel()
    for f, fr in enumerate(sc.frames):
        trk.step(f, fr.boxes, fr.scores, fr.feats)            # sai shape/nhãn sẽ ném ValueError ở đây
    assert len(trk.mgr.archive.lives) >= 1
