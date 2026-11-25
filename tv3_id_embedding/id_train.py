"""
id_train.py  -  Huấn luyện thu nhỏ ID Decoder trên dữ liệu tổng hợp (synthetic_data.py) để chứng minh
decoder HỌC được in-context ID prediction. KHÔNG phải huấn luyện MOTIP đầy đủ (cần DanceTrack + GPU).

    python id_train.py --steps 1500 --C 32 --T 8 --layers 2 --d-model 128 --save checkpoints/id_decoder_synth.pth
"""
import argparse

import numpy as np
import torch

from id_decoder import IDDecoderConfig, MOTIPIDDecoder, id_loss, save_id_decoder
from synthetic_data import IGNORE, SynthConfig, collate, make_sample


def to_t(b, device):
    f = lambda k, dt: torch.as_tensor(b[k], dtype=dt, device=device)
    return dict(hist_feats=f("hist_feats", torch.float32), hist_labels=f("hist_labels", torch.long),
                hist_dt=f("hist_dt", torch.long), hist_mask=f("hist_mask", torch.bool),
                det_feats=f("det_feats", torch.float32), det_mask=f("det_mask", torch.bool)), f("targets", torch.long)


@torch.no_grad()
def evaluate(model, cfg: SynthConfig, device, n_batches=20, batch_size=16, seed=123):
    model.eval()
    rng = np.random.default_rng(seed)
    ok = tot = ok_new = tot_new = 0
    for _ in range(n_batches):
        x, y = to_t(collate([make_sample(rng, cfg, augment=False) for _ in range(batch_size)]), device)
        pred = model.forward_batched(**x).argmax(-1)
        valid = y != IGNORE
        ok += int(((pred == y) & valid).sum()); tot += int(valid.sum())
        new = y == cfg.num_labels
        ok_new += int(((pred == y) & new).sum()); tot_new += int(new.sum())
    model.train()
    return ok / max(tot, 1), ok_new / max(tot_new, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--C", type=int, default=256, help="số chiều feature; = C của detector")
    ap.add_argument("--T", type=int, default=29, help="độ dài lịch sử")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="checkpoints/id_decoder_synth.pth")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    scfg = SynthConfig(num_labels=a.K, feat_dim=a.C, history_len=a.T)
    model = MOTIPIDDecoder(IDDecoderConfig(num_labels=a.K, feat_dim=a.C, d_model=a.d_model,
                                           num_heads=a.heads, num_layers=a.layers, ffn_dim=4 * a.d_model)).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    print(f"tham số: {sum(p.numel() for p in model.parameters()):,} | thiết bị: {a.device}")

    acc, acc_new = evaluate(model, scfg, a.device)
    print(f"[trước huấn luyện] acc={acc:.3f}  newborn_recall={acc_new:.3f}")
    for step in range(1, a.steps + 1):
        x, y = to_t(collate([make_sample(rng, scfg, augment=not a.no_aug) for _ in range(a.batch)]), a.device)
        loss = id_loss(model.forward_batched(**x), y, IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == a.steps:
            acc, acc_new = evaluate(model, scfg, a.device)
            print(f"step {step:5d}  loss={loss.item():.4f}  acc={acc:.3f}  newborn_recall={acc_new:.3f}")
    save_id_decoder(model, a.save)
    print("Đã lưu:", a.save)


if __name__ == "__main__":
    main()
