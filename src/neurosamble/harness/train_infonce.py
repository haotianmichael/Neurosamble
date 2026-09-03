"""
Minimal single-GPU InfoNCE training loop for the signal-domain SignalEncoder.

This is the clean baseline that REPLACES the evaluate/ DDP+FAISS stack:
  real ecoli R9 reads (blow5) + truth.paf coords
    -> SignalPairDataset (query real signal  vs  pore-model expected ref signal)
    -> shared SignalEncoder (encoder_type='transformer')
    -> AveragePooler -> cosine similarity (B x B) -> InfoNCE (CrossEntropy vs arange)

Purpose: get a correct, converging PyTorch baseline. Later, swap individual
operators (attention / FFN / similarity) for hand-written CuTe/CUTLASS kernels
and RE-check each against this reference (see validate.py).

Run:
    cd /home/nfs/mahaotian/ESA/Neurosamble
    export PORE_MODEL_PATH=/home/nfs/mahaotian/ESA/Rawhash2/extern/kmer_models/legacy/legacy_r9.4_180mv_450bps_6mer/template_median68pA.model
    python src/neurosamble/harness/train_infonce.py
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from neurosamble.paths import P

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- make `neurosamble` importable regardless of CWD ---
_SRC = Path(__file__).resolve().parents[2]          # .../Neurosamble/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neurosamble.model.pore_model import PoreModel
from neurosamble.model.signal_dataset import SignalPairDataset, signal_collate
from neurosamble.model.signal_encoder import SignalEncoder            # adjust if your factory differs
from neurosamble.model.similarity import SimilarityWithTemperature
from neurosamble.model.pooling import AveragePooler


# --------------------------------------------------------------------------- #
# Config (hardcoded from SignalModelConfigSchema defaults; edit here, no pydantic)
# --------------------------------------------------------------------------- #
class Cfg:
    # data
    pairs_pkl = str(P.pairs_pkl)
    unit_length      = 300      # ref window in bp  (== index window span)
    input_signal_len = 2000     # samples per window (divisible by downsample_factor)
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
    # train
    batch_size       = 48
    lr               = 1e-4
    temperature      = 0.05
    max_steps        = 2000
    log_interval     = 20
    device           = "cuda:0"
    seed             = 42


def build_encoder(cfg: Cfg) -> nn.Module:
    """Construct the SignalEncoder. If your signal_encoder.py exposes a
    `signal_encoder_from_config`, prefer that and pass an equivalent config obj.
    The kwargs below mirror SignalModelConfigSchema; rename to match your ctor."""
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
        n_mamba_blocks=cfg.n_blocks
    )


def encode(encoder, pooler, x, device):
    """x: {'signal': [B, L], 'attention_mask': [B, T]} -> [B, D] normalized."""
    sig  = x["signal"].to(device)             # [B, L]
    mask = x["attention_mask"].to(device)     # [B, T]
    out  = encoder(sig, attention_mask=mask)  # -> [B, T, D]  (adjust if your API differs)
    hidden = out["last_hidden_state"] if isinstance(out, dict) else out
    emb = pooler(hidden, mask)                 # [B, D] masked mean
    return F.normalize(emb, dim=-1)


def main():
    cfg = Cfg()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # --- data ---
    d = pickle.load(open(cfg.pairs_pkl, "rb"))
    pore = PoreModel(
        kmer_table_path=os.environ.get("PORE_MODEL_PATH"),
        kmer_len=cfg.kmer_len,
        samples_per_kmer=cfg.samples_per_kmer,
    )
    ds = SignalPairDataset(
        query_signals=d["query_signals"],
        query_coords=d["query_coords"],
        query_ends=d.get("query_ends"),
        query_strands=d.get("query_strands"),
        reference_seq=d["reference_seq"],
        pore_model=pore,
        unit_length=cfg.unit_length,
        input_signal_len=cfg.input_signal_len,
        downsample_factor=cfg.downsample_factor,
    )
    # IterableDataset: no shuffle; it yields random pairs forever.
    loader = DataLoader(ds, batch_size=cfg.batch_size, collate_fn=signal_collate,
                        num_workers=2, drop_last=True)

    # --- model ---
    encoder = build_encoder(cfg).to(device)
    pooler  = AveragePooler().to(device)
    sim_fn  = SimilarityWithTemperature(temperature=cfg.temperature)
    ce      = nn.CrossEntropyLoss()
    opt     = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr)

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[model] SignalEncoder({cfg.encoder_type}) params={n_params/1e6:.2f}M "
          f"| reads={len(d['query_signals'])} | device={device}")

    # --- train loop (the 15-line forward+loss, unrolled) ---
    encoder.train()
    step = 0
    running = 0.0
    for x_1, x_2 in loader:                     # H=0 path -> (x_1, x_2)
        y1 = encode(encoder, pooler, x_1, device)   # [B, D]
        y2 = encode(encoder, pooler, x_2, device)   # [B, D]

        # (B, B) similarity; diagonal are the positive pairs
        sim = sim_fn(y1.unsqueeze(1), y2.unsqueeze(0))   # [B, B]
        target = torch.arange(sim.size(0), device=device)
        loss = ce(sim, target)                            # InfoNCE

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        opt.step()

        running += loss.item()
        step += 1
        if step % cfg.log_interval == 0:
            avg = running / cfg.log_interval
            with torch.no_grad():                 # quick in-batch acc: is diagonal argmax?
                acc = (sim.argmax(1) == target).float().mean().item()
            print(f"[step {step:5d}] loss={avg:.4f}  inbatch_acc={acc:.3f}", flush=True)
            running = 0.0
            # collapse-to-0 is a leakage signal (same warning as stage4b):
            if avg < 1e-3:
                print("[warn] loss ~0 — check for pair leakage / trivial positives.")
        if step >= cfg.max_steps:
            break

    ckpt = _SRC.parent / "data" / "baseline_encoder.pt"
    torch.save({"model": encoder.state_dict(), "cfg": vars(cfg)}, ckpt)
    print(f"[done] saved {ckpt}")


if __name__ == "__main__":
    main()