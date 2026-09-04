"""
Shared DDP + InfoNCE machinery for pretrain.py / finetune.py.

This is a faithful port of ESA's evaluate/pilot_recall.py DDP scheme:
  - autograd-aware all-gather (_gather_cat, MoCo-v3/SimCLR splice pattern)
  - one COMBINED forward over [anchors | positives | H hard negs]
  - gathered InfoNCE loss over the GLOBAL batch: logits=[B,B | B,H], label=diag
  - (loss * world_size).backward() so the DDP-averaged grad == single-GPU
    batch-`batch_size` grad exactly.

Nothing here touches the core model modules (SignalEncoder / AveragePooler);
DDP is a wrapper around them.
"""
from __future__ import annotations
import os, datetime
import torch
import torch.distributed as dist
import torch.nn.functional as F


def dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, 0


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=6))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def gather_cat(t, world_size, rank):
    """Autograd-aware all-gather along dim 0, rank order (MoCo-v3/SimCLR splice)."""
    if world_size == 1:
        return t
    gathered = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t.contiguous())
    gathered[rank] = t  # keep local slice grad-connected
    return torch.cat(gathered, dim=0)


def forward_local_embeddings(model, pooling, x1, x2, xneg, device):
    """One combined forward -> L2-normalized local y1[b,D], y2[b,D], yneg[b,H,D]|None."""
    b = x1["signal"].shape[0]
    sigs = [x1["signal"], x2["signal"]]
    masks = [x1["attention_mask"], x2["attention_mask"]]
    H = 0
    if xneg is not None:
        H = xneg["signal"].shape[1]
        L = xneg["signal"].shape[2]
        T = xneg["attention_mask"].shape[2]
        sigs.append(xneg["signal"].reshape(b * H, L))
        masks.append(xneg["attention_mask"].reshape(b * H, T))
    signal = torch.cat(sigs, 0).to(device)
    attention_mask = torch.cat(masks, 0).to(device)
    h = model(signal=signal, attention_mask=attention_mask)   # DDP-wrapped encoder
    h = h["last_hidden_state"] if isinstance(h, dict) else h
    y = pooling(h, attention_mask=attention_mask)
    y = F.normalize(y, dim=-1)
    y1 = y[:b]; y2 = y[b:2 * b]
    yneg = y[2 * b:].reshape(b, H, -1) if H > 0 else None
    return y1, y2, yneg


def gathered_infonce_loss(y1_local, y2_local, yneg_local, temp, world_size, rank):
    y1 = gather_cat(y1_local, world_size, rank)
    y2 = gather_cat(y2_local, world_size, rank)
    B = y1.shape[0]
    logits_pos = (y1 @ y2.t()) / temp
    if yneg_local is not None:
        yneg = gather_cat(yneg_local, world_size, rank)
        logits_neg = torch.einsum("bd,bhd->bh", y1, yneg) / temp
        logits = torch.cat([logits_pos, logits_neg], dim=1)
    else:
        logits = logits_pos
    labels = torch.arange(B, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss, logits, labels