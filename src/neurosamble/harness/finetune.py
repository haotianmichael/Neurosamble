"""
Stage 2 — real-read FINE-TUNING, 2-GPU DDP, warm-started from
data/canonical_encoder.pt. STRICT replica of ESA stage4b_finetune config:
  optimizer=Adam, OneCycleLR(default pct_start), temperature=0.05, lr=1e-4,
  batch_size=48 (effective, all-gathered), hard_negatives=8 (30..300bp),
  grad clip 1.0, (loss*world_size).backward(), train_steps=20000 (~3 epochs).

Query side = real ecoli reads (data/ecoli_pairs.pkl from prep_data.py),
database side = pore-model expected signals + near-coord hard negatives.

Prereq: prep_data.py -> data/ecoli_pairs.pkl ; pretrain.py -> data/canonical_encoder.pt

Output: data/real_encoder_v1.pt

Launch (2 GPU):
    cd /home/nfs/mahaotian/ESA/Neurosamble
    PYTHONPATH=$PWD/src torchrun --nproc_per_node=2 --master-port <PORT> \
        -m neurosamble.harness.finetune
"""
from __future__ import annotations
import os, sys, pickle
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neurosamble.utils.paths import P
from neurosamble.model.pore_model import PoreModel
from neurosamble.model.signal_dataset import SignalPairDataset, signal_collate
from neurosamble.model.signal_encoder import SignalEncoder
from neurosamble.model.pooling import AveragePooler
from neurosamble.harness._ddp_common import (
    setup_distributed, dist_info, forward_local_embeddings, gathered_infonce_loss,
)


class Cfg:
    pairs_pkl         = str(P.pairs_pkl)
    canonical         = str(Path(P.OUT_DIR) / "canonical_encoder.pt")
    # --- model / data (LOCK identical to pretrain.py) ---
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
    # --- contrastive (ESA) ---
    hard_negatives    = 8
    hard_neg_min_bp   = 30
    hard_neg_max_bp   = 300
    temperature       = 0.05
    # --- optim (ESA stage4b) ---
    batch_size        = 48
    lr                = 1e-4
    train_steps       = 20000
    log_interval      = 100
    seed              = 42


def build_encoder(cfg):
    return SignalEncoder(
        encoder_type=cfg.encoder_type, conv_channels_1=cfg.conv_channels_1,
        conv_channels_2=cfg.conv_channels_2, conv_kernel_1=cfg.conv_kernel_1,
        num_heads=cfg.num_heads, embedding_dim=cfg.embedding_dim, dropout=cfg.dropout,
        input_signal_len=cfg.input_signal_len, downsample_factor=cfg.downsample_factor,
        n_blocks=cfg.n_blocks,
    )


def main():
    cfg = Cfg()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed + rank)

    if cfg.batch_size % world_size != 0:
        raise ValueError(f"batch_size {cfg.batch_size} must be divisible by world_size {world_size}")
    local_bs = cfg.batch_size // world_size

    if rank == 0:
        P.ensure_out()
    d = pickle.load(open(cfg.pairs_pkl, "rb"))
    pore = PoreModel(kmer_table_path=os.environ.get("PORE_MODEL_PATH", str(P.pore_model)),
                     kmer_len=cfg.kmer_len, samples_per_kmer=cfg.samples_per_kmer)
    ds = SignalPairDataset(
        query_signals=d["query_signals"], query_coords=d["query_coords"],
        query_ends=d.get("query_ends"), query_strands=d.get("query_strands"),
        reference_seq=d["reference_seq"], pore_model=pore,
        unit_length=cfg.unit_length, input_signal_len=cfg.input_signal_len,
        downsample_factor=cfg.downsample_factor, hard_negatives=cfg.hard_negatives,
        hard_neg_min_bp=cfg.hard_neg_min_bp, hard_neg_max_bp=cfg.hard_neg_max_bp,
    )
    loader = DataLoader(ds, batch_size=local_bs, collate_fn=signal_collate)

    encoder = build_encoder(cfg).to(device)
    # --- WARM START from canonical ---
    ck = torch.load(cfg.canonical, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if rank == 0:
        print(f"[warmstart] {cfg.canonical} | missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if len(missing) + len(unexpected) > 0:
            print(f"  [!] missing={missing}\n  [!] unexpected={unexpected}", flush=True)

    pooling = AveragePooler().to(device)
    model = DDP(encoder, device_ids=[local_rank] if torch.cuda.is_available() else None) \
            if world_size > 1 else encoder

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=cfg.train_steps + 1)

    if rank == 0:
        n = sum(p.numel() for p in encoder.parameters())
        print(f"[finetune] {cfg.encoder_type} params={n/1e6:.2f}M reads={len(d['query_signals'])} "
              f"eff_batch={cfg.batch_size} local_bs={local_bs} world={world_size} "
              f"H={cfg.hard_negatives} lr={cfg.lr} temp={cfg.temperature} steps={cfg.train_steps}", flush=True)

    model.train()
    it = iter(loader)
    for step in range(cfg.train_steps):
        batch = next(it)
        x1, x2 = batch[0], batch[1]; xneg = batch[2] if len(batch) == 3 else None
        y1l, y2l, ynegl = forward_local_embeddings(model, pooling, x1, x2, xneg, device)
        loss, logits, labels = gathered_infonce_loss(y1l, y2l, ynegl, cfg.temperature, world_size, rank)
        (loss * world_size).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(); scheduler.step()
        if rank == 0 and step % cfg.log_interval == 0:
            acc = (logits.argmax(1) == labels).float().mean().item()
            print(f"[step {step:6d}] loss {loss.item():.4f} acc {acc:.3f} "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} (H={cfg.hard_negatives}, "
                  f"world={world_size}, local_bs={local_bs})", flush=True)

    if rank == 0:
        ckpt = Path(P.OUT_DIR) / "real_encoder_v1.pt"
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