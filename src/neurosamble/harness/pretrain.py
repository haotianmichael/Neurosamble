"""
Stage 1 — canonical PRE-TRAINING, 2-GPU DDP, STRICT replica of ESA's
pilot_recall.train_encoder_hardneg config:
  optimizer=Adam, scheduler=OneCycleLR(max_lr=lr, total_steps=steps+1) [default
  pct_start=0.3], temperature=0.05, lr=1e-4, batch_size=48 (effective, held
  constant via all-gather regardless of GPU count), hard_negatives=8 (30..300bp),
  grad clip 1.0, (loss*world_size).backward().

Positive pairs are signal-to-signal at the same locus (pore-model expected
signal, tiny base jitter on the query side). Reference genome + pore model
generate the canonical signals; no real reads here.

Output: data/canonical_encoder.pt

Launch (2 GPU):
    cd /home/nfs/mahaotian/ESA/Neurosamble
    PYTHONPATH=$PWD/src torchrun --nproc_per_node=2 --master-port <PORT> \
        -m neurosamble.harness.pretrain
"""
from __future__ import annotations
import os, sys, random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import IterableDataset, DataLoader

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neurosamble.utils.paths import P
from neurosamble.model.pore_model import PoreModel
from neurosamble.model.signal_dataset import preprocess_window, signal_collate
from neurosamble.model.signal_encoder import SignalEncoder
from neurosamble.model.pooling import AveragePooler
from neurosamble.harness._ddp_common import (
    setup_distributed, dist_info, forward_local_embeddings, gathered_infonce_loss,
)


class Cfg:
    # --- model / data (LOCK identical to finetune.py for warm-start) ---
    unit_length       = 300
    input_signal_len  = 2000
    downsample_factor = 5
    encoder_type      = "transformer"
    conv_channels_1   = 64
    conv_channels_2   = 512
    conv_kernel_1     = 11
    n_blocks          = 6
    num_heads         = 6
    embedding_dim     = 384
    dropout           = 0.1
    kmer_len          = 6
    samples_per_kmer  = 9
    # --- contrastive (ESA defaults) ---
    hard_negatives    = 8
    hard_neg_min_bp   = 30
    hard_neg_max_bp   = 300
    temperature       = 0.05         # ESA
    # --- optim (ESA: Adam + OneCycle default pct_start; batch 48; lr 1e-4) ---
    batch_size        = 48           # EFFECTIVE contrastive batch (all-gathered)
    lr                = 1e-4
    train_steps       = 20000
    log_interval      = 100
    seed              = 42
    jitter_bp         = 10


def _revcomp(s):
    return s.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


class CanonicalPairDataset(IterableDataset):
    def __init__(self, reference_seq, pore_model, cfg):
        self.ref = reference_seq; self.pm = pore_model; self.cfg = cfg
        self.win_bp = cfg.unit_length
        self.max_start = len(reference_seq) - cfg.unit_length - cfg.hard_neg_max_bp - 1

    def _win(self, coord, strand):
        coord = int(min(max(coord, 0), self.max_start))
        bases = self.ref[coord:coord + self.win_bp]
        if strand == "-":
            bases = _revcomp(bases)
        sig = self.pm.sequence_to_signal(bases)
        return preprocess_window(sig, self.cfg.input_signal_len, self.cfg.downsample_factor)

    def _pair(self):
        anchor = random.randint(0, self.max_start)
        strand = "+" if random.random() < 0.5 else "-"
        jit = random.randint(-self.cfg.jitter_bp, self.cfg.jitter_bp)
        q_sig, q_mask = self._win(anchor + jit, strand)
        r_sig, r_mask = self._win(anchor, strand)
        x1 = {"signal": q_sig, "attention_mask": q_mask}
        x2 = {"signal": r_sig, "attention_mask": r_mask}
        if self.cfg.hard_negatives <= 0:
            return x1, x2
        ns, nm = [], []
        for _ in range(self.cfg.hard_negatives):
            sign = 1 if random.random() < 0.5 else -1
            off = random.randint(self.cfg.hard_neg_min_bp, self.cfg.hard_neg_max_bp)
            s, m = self._win(anchor + sign * off, strand); ns.append(s); nm.append(m)
        xneg = {"signal": np.stack(ns), "attention_mask": np.stack(nm)}
        return x1, x2, xneg

    def __iter__(self):
        while True:
            yield self._pair()


def build_encoder(cfg):
    return SignalEncoder(
        encoder_type=cfg.encoder_type, conv_channels_1=cfg.conv_channels_1,
        conv_channels_2=cfg.conv_channels_2, conv_kernel_1=cfg.conv_kernel_1,
        num_heads=cfg.num_heads, embedding_dim=cfg.embedding_dim, dropout=cfg.dropout,
        input_signal_len=cfg.input_signal_len, downsample_factor=cfg.downsample_factor,
        n_blocks=cfg.n_blocks,
    )


def read_ref(path):
    name, buf = None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name: break
                name = line[1:].split()[0]
            elif name:
                buf.append(line.strip())
    return "".join(buf)


def main():
    cfg = Cfg()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed + rank); random.seed(cfg.seed + rank); np.random.seed(cfg.seed + rank)

    if cfg.batch_size % world_size != 0:
        raise ValueError(f"batch_size {cfg.batch_size} must be divisible by world_size {world_size}")
    local_bs = cfg.batch_size // world_size

    if rank == 0:
        P.ensure_out()
    ref_seq = read_ref(str(P.ref))
    pore = PoreModel(kmer_table_path=os.environ.get("PORE_MODEL_PATH", str(P.pore_model)),
                     kmer_len=cfg.kmer_len, samples_per_kmer=cfg.samples_per_kmer)
    ds = CanonicalPairDataset(ref_seq, pore, cfg)
    loader = DataLoader(ds, batch_size=local_bs, collate_fn=signal_collate)

    encoder = build_encoder(cfg).to(device)
    pooling = AveragePooler().to(device)
    model = DDP(encoder, device_ids=[local_rank] if torch.cuda.is_available() else None) \
            if world_size > 1 else encoder

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)          # ESA: Adam
    scheduler = OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=cfg.train_steps + 1)  # ESA: default pct_start

    if rank == 0:
        n = sum(p.numel() for p in encoder.parameters())
        print(f"[pretrain] {cfg.encoder_type} params={n/1e6:.2f}M | eff_batch={cfg.batch_size} "
              f"local_bs={local_bs} world={world_size} H={cfg.hard_negatives} "
              f"lr={cfg.lr} temp={cfg.temperature} steps={cfg.train_steps}", flush=True)

    model.train()
    it = iter(loader)
    for step in range(cfg.train_steps):
        batch = next(it)
        x1, x2 = batch[0], batch[1]; xneg = batch[2] if len(batch) == 3 else None
        y1l, y2l, ynegl = forward_local_embeddings(model, pooling, x1, x2, xneg, device)
        loss, logits, labels = gathered_infonce_loss(y1l, y2l, ynegl, cfg.temperature, world_size, rank)
        (loss * world_size).backward()                                    # ESA grad compensation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(); scheduler.step()
        if rank == 0 and step % cfg.log_interval == 0:
            acc = (logits.argmax(1) == labels).float().mean().item()
            print(f"[step {step:6d}] loss {loss.item():.4f} acc {acc:.3f} "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} (H={cfg.hard_negatives}, "
                  f"world={world_size}, local_bs={local_bs})", flush=True)

    if rank == 0:
        ckpt = Path(P.OUT_DIR) / "canonical_encoder.pt"
        payload = {f: getattr(cfg, f) for f in
                   ("encoder_type", "conv_channels_1", "conv_channels_2", "conv_kernel_1",
                    "n_blocks", "num_heads", "embedding_dim", "dropout",
                    "input_signal_len", "downsample_factor")}
        torch.save({"model": encoder.state_dict(), "signal_config": payload}, ckpt)
        print(f"[done] saved {ckpt}", flush=True)
    if world_size > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()