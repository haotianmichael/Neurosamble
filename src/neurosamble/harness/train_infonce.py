"""
Single-GPU InfoNCE training for the signal-domain SignalEncoder (transformer body).

Clean baseline replacing the evaluate/ DDP+FAISS stack:
  real ecoli R9 reads (blow5) + truth.paf coords
    -> SignalPairDataset (query real signal vs pore-model expected ref signal
                          + H near-coordinate HARD NEGATIVES)
    -> shared SignalEncoder(encoder_type='transformer')
    -> AveragePooler -> L2-normalize
    -> logits = [ y1 @ y2.T | <y1, yneg> ] / temp ; CrossEntropy(., arange(B))

Hard negatives are REQUIRED here: with in-batch (random, far-away) negatives only,
the query/reference domain gap lets the encoder collapse every embedding to one
point (cos~0.99, loss stuck at ln(B)). Near-coordinate negatives force fine
positional discrimination and break the collapse — this is the single difference
from ESA's working train_encoder_hardneg.

Run:
    cd /home/nfs/mahaotian/ESA/Neurosamble
    python -m neurosamble.harness.train_infonce
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[2]          # .../Neurosamble/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neurosamble.paths import P
from neurosamble.model.pore_model import PoreModel
from neurosamble.model.signal_dataset import SignalPairDataset, signal_collate
from neurosamble.model.signal_encoder import SignalEncoder
from neurosamble.model.pooling import AveragePooler


class Cfg:
    # data
    pairs_pkl        = str(P.pairs_pkl)
    unit_length      = 300
    input_signal_len = 2000
    downsample_factor = 5
    # model
    encoder_type     = "transformer"
    conv_channels_1  = 64
    conv_channels_2  = 512
    conv_kernel_1    = 11
    n_blocks         = 6
    num_heads        = 6
    embedding_dim    = 384
    dropout          = 0.1
    # pore model
    kmer_len         = 6
    samples_per_kmer = 9
    # contrastive
    hard_negatives   = 8        # <<< the fix: near-coordinate hard negs
    hard_neg_min_bp  = 30
    hard_neg_max_bp  = 300
    temperature      = 0.07
    # optim
    batch_size       = 32       # 48 -> 16 ; combined = 16+16+16*8 = 160 样本
    lr               = 3e-4     # max_lr for OneCycle
    max_steps        = 30000
    warmup_frac      = 0.1
    log_interval     = 20
    device           = "cuda:0"
    seed             = 42


def build_encoder(cfg):
    return SignalEncoder(
        encoder_type=cfg.encoder_type,
        conv_channels_1=cfg.conv_channels_1,
        conv_channels_2=cfg.conv_channels_2,
        conv_kernel_1=cfg.conv_kernel_1,
        num_heads=cfg.num_heads,
        embedding_dim=cfg.embedding_dim,
        dropout=cfg.dropout,
        input_signal_len=cfg.input_signal_len,
        downsample_factor=cfg.downsample_factor,
        n_blocks=cfg.n_blocks,
    )


def encode_flat(encoder, pooler, signal, mask):
    """signal [N, L], mask [N, T] -> [N, D] L2-normalized."""
    h = encoder(signal, attention_mask=mask)
    h = h["last_hidden_state"] if isinstance(h, dict) else h
    y = pooler(h, mask)
    return F.normalize(y, dim=-1)


def forward_embeddings(encoder, pooler, x1, x2, xneg, device):
    """One combined forward over [anchors | positives | (B*H hard negs)].
    Returns y1 [B,D], y2 [B,D], yneg [B,H,D] | None."""
    b = x1["signal"].shape[0]
    sigs  = [x1["signal"], x2["signal"]]
    masks = [x1["attention_mask"], x2["attention_mask"]]
    H = 0
    if xneg is not None:
        H = xneg["signal"].shape[1]
        L = xneg["signal"].shape[2]
        T = xneg["attention_mask"].shape[2]
        sigs.append(xneg["signal"].reshape(b * H, L))
        masks.append(xneg["attention_mask"].reshape(b * H, T))
    signal = torch.cat(sigs, 0).to(device)
    mask   = torch.cat(masks, 0).to(device)
    y = encode_flat(encoder, pooler, signal, mask)
    y1 = y[:b]
    y2 = y[b:2 * b]
    yneg = y[2 * b:].reshape(b, H, -1) if H > 0 else None
    return y1, y2, yneg


def infonce_loss(y1, y2, yneg, temp):
    """logits = [y1@y2.T | <y1,yneg_h>] / temp ; label = diagonal."""
    B = y1.shape[0]
    logits = (y1 @ y2.t()) / temp                       # [B, B]
    if yneg is not None:
        neg = torch.einsum("bd,bhd->bh", y1, yneg) / temp   # [B, H]
        logits = torch.cat([logits, neg], dim=1)            # [B, B+H]
    labels = torch.arange(B, device=logits.device)
    return F.cross_entropy(logits, labels), logits, labels


def main():
    cfg = Cfg()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    d = pickle.load(open(cfg.pairs_pkl, "rb"))
    pore = PoreModel(
        kmer_table_path=os.environ.get("PORE_MODEL_PATH", str(P.pore_model)),
        kmer_len=cfg.kmer_len, samples_per_kmer=cfg.samples_per_kmer,
    )
    ds = SignalPairDataset(
        query_signals=d["query_signals"], query_coords=d["query_coords"],
        query_ends=d.get("query_ends"), query_strands=d.get("query_strands"),
        reference_seq=d["reference_seq"], pore_model=pore,
        unit_length=cfg.unit_length, input_signal_len=cfg.input_signal_len,
        downsample_factor=cfg.downsample_factor,
        hard_negatives=cfg.hard_negatives,
        hard_neg_min_bp=cfg.hard_neg_min_bp, hard_neg_max_bp=cfg.hard_neg_max_bp,
    )
    # num_workers=0: IterableDataset + multi-worker duplicates the RNG stream.
    loader = DataLoader(ds, batch_size=cfg.batch_size, collate_fn=signal_collate,
                        num_workers=0, drop_last=True)

    encoder = build_encoder(cfg).to(device)
    pooler  = AveragePooler().to(device)
    opt     = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr)
    sched   = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.max_steps + 1,
        pct_start=cfg.warmup_frac,
    )

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[model] SignalEncoder({cfg.encoder_type}) params={n_params/1e6:.2f}M "
          f"| reads={len(d['query_signals'])} | H={cfg.hard_negatives} | device={device}")

    # ---- collapse probe (random init): y2 std ~0 = collapse; gap>0 = learnable ----
    with torch.no_grad():
        encoder.eval()
        batch = next(iter(loader))
        x1, x2 = batch[0], batch[1]
        xneg = batch[2] if len(batch) == 3 else None
        y1, y2, _ = forward_embeddings(encoder, pooler, x1, x2, xneg, device)
        pos = (y1 * y2).sum(-1).mean().item()
        negm = (y1 @ y2.t()).fill_diagonal_(0).sum().item() / (y1.size(0) * (y1.size(0) - 1))
        print(f"[probe] y1_std={y1.std(0).mean():.4f} y2_std={y2.std(0).mean():.4f} "
              f"cos(pos)={pos:.4f} cos(neg)={negm:.4f} gap={pos-negm:.4f}")
        encoder.train()

    step, running, run_acc = 0, 0.0, 0.0
    for batch in loader:
        x1, x2 = batch[0], batch[1]
        xneg = batch[2] if len(batch) == 3 else None

        y1, y2, yneg = forward_embeddings(encoder, pooler, x1, x2, xneg, device)
        loss, logits, labels = infonce_loss(y1, y2, yneg, cfg.temperature)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        opt.step()
        sched.step()

        running += loss.item()
        run_acc += (logits.argmax(1) == labels).float().mean().item()
        step += 1
        if step % cfg.log_interval == 0:
            lr = opt.param_groups[0]["lr"]
            print(f"[step {step:5d}] loss={running/cfg.log_interval:.4f} "
                  f"acc={run_acc/cfg.log_interval:.3f} lr={lr:.2e}", flush=True)
            running, run_acc = 0.0, 0.0
        if step >= cfg.max_steps:
            break

    P.ensure_out()
    ckpt = P.baseline_encoder
    torch.save({"model": encoder.state_dict(), "cfg": vars(cfg)}, ckpt)
    print(f"[done] saved {ckpt}")


if __name__ == "__main__":
    main()