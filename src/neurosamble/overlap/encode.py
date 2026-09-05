"""
Neurosamble Phase 4 -- 2-GPU data-parallel read-window encode -> disk (once).

Full-scale all-vs-all needs every read-window embedding materialized once. The
Phase-2 path recomputed embeddings during query and held everything in RAM; at
~353k reads (~27M windows, ~42 GB of float32) that is intractable. This module
encodes ALL windows in parallel across both GPUs and persists them to per-rank
shard files, so the index/query steps never touch the encoder again.

Launch (world_size = NUM_GPUS):
    torchrun --nproc_per_node=2 overlap_encode_mp.py --real_reads ... --out_dir ...

Pure sharded inference: NO DDP wrapper, NO all-gather, NO gradients. Each rank
binds cuda:{LOCAL_RANK}, streams the blow5 (pyslow5, pA=True -- identical to
read_blow5, so embeddings match the subset path), keeps reads with
``global_read_index % world_size == rank``, tiles (tile_read, win/stride as
Phase 2), encodes (SignalEvalModel.encode), and appends to its OWN shard files
(disjoint -- no shared-memmap contention):

  embeddings_shard{r}.f32  raw float32 [n_win_r, D], L2-normalized (encode does it)
  windows_shard{r}.npy     int64 [n_win_r, 2] rows (global_read_id, offset_samples)
  read_ids_shard{r}.txt     lines: "global_read_id<TAB>read_name<TAB>n_samples"
  shard{r}.done             completion sentinel with counts (resume checkpoint)

Rank 0 waits for all sentinels and writes encode_manifest.json.
Streaming keeps memory bounded to one batch; nothing holds all reads at once.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from neurosamble.overlap.tile import tile_read


def _dist_env():
    """(rank, world_size, local_rank) from torchrun env; single-process fallback."""
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 2-GPU sharded read-window encode")
    p.add_argument("--real_reads", required=True, help="full blow5/slow5 (ALL reads)")
    p.add_argument("--load_encoder", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--win", type=int, default=2000)
    p.add_argument("--stride", type=int, default=1000)
    p.add_argument("--encode_batch", type=int, default=4096,
                   help="windows flushed to the encoder at once (encode sub-batches internally)")
    return p.parse_args()


def _shard_paths(out_dir, rank):
    return (
        os.path.join(out_dir, f"embeddings_shard{rank}.f32"),
        os.path.join(out_dir, f"windows_shard{rank}.npy"),
        os.path.join(out_dir, f"read_ids_shard{rank}.txt"),
        os.path.join(out_dir, f"shard{rank}.done"),
    )


def main():
    args = parse_args()
    rank, world_size, local_rank = _dist_env()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from neurosamble.overlap.eval_model import load_encoder

    emb_path, win_path, rid_path, done_path = _shard_paths(args.out_dir, rank)

    # Resume checkpoint: this rank already finished.
    if os.path.exists(done_path) and os.path.exists(emb_path) and os.path.exists(win_path):
        print(f"[encode][rank{rank}] shard complete; skipping (checkpoint: {done_path})",
              flush=True)
    else:
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            torch.cuda.reset_peak_memory_stats(local_rank)
        model, cfg = load_encoder(args.load_encoder, device)
        D = int(model.get_sentence_embedding_dimension())
        print(f"[encode][rank{rank}] world_size={world_size} device={device} D={D} "
              f"win={args.win} stride={args.stride} (reads where gid%{world_size}=={rank})",
              flush=True)

        import pyslow5
        s = pyslow5.Open(args.real_reads, "r")

        n_reads_rank = 0
        n_win_rank = 0
        win_chunks = []
        buf_sigs, buf_meta = [], []
        t0 = time.time()

        emb_f = open(emb_path, "wb")
        rid_f = open(rid_path, "w")

        def flush():
            nonlocal buf_sigs, buf_meta
            if not buf_sigs:
                return
            vecs = model.encode(buf_sigs)          # [b, D] float32, L2-normalized
            vecs.astype(np.float32).tofile(emb_f)
            win_chunks.append(np.asarray(buf_meta, dtype=np.int64))
            buf_sigs, buf_meta = [], []

        gid = -1
        for rec in s.seq_reads(pA=True):           # pA=True == read_blow5's signal
            gid += 1
            if (gid % world_size) != rank:
                continue
            sig = np.asarray(rec["signal"], dtype=np.float32)
            tiles = tile_read(sig, win=args.win, stride=args.stride)
            if not tiles:
                continue
            rid_f.write(f"{gid}\t{rec['read_id']}\t{int(len(sig))}\n")
            n_reads_rank += 1
            for off, w in tiles:
                buf_sigs.append(w)
                buf_meta.append((gid, int(off)))
                n_win_rank += 1
                if len(buf_sigs) >= args.encode_batch:
                    flush()
            if (n_reads_rank % 20000) == 0:
                dt = time.time() - t0
                print(f"[encode][rank{rank}] reads={n_reads_rank} windows={n_win_rank} "
                      f"({n_reads_rank/max(dt,1e-9):.0f} reads/s)", flush=True)
        flush()
        s.close()
        emb_f.close()
        rid_f.close()

        windows = (np.concatenate(win_chunks, axis=0) if win_chunks
                   else np.zeros((0, 2), dtype=np.int64))
        np.save(win_path, windows)

        wall = time.time() - t0
        peak_gpu_gb = (torch.cuda.max_memory_allocated(local_rank) / 1e9
                       if torch.cuda.is_available() else 0.0)
        with open(done_path, "w") as f:
            json.dump({
                "rank": rank, "world_size": world_size,
                "n_reads": n_reads_rank, "n_windows": int(n_win_rank),
                "D": D, "win": args.win, "stride": args.stride,
                "wall_sec": round(wall, 1), "peak_gpu_mem_gb": round(peak_gpu_gb, 2),
                "emb_file": os.path.basename(emb_path),
                "win_file": os.path.basename(win_path),
                "rid_file": os.path.basename(rid_path),
            }, f)
        print(f"[encode][rank{rank}] DONE reads={n_reads_rank} windows={n_win_rank} "
              f"wall={wall:.1f}s peak_gpu={peak_gpu_gb:.2f}GB", flush=True)

    # ---- rank 0 aggregates the manifest once every shard is done ------------- #
    if rank == 0:
        manifest_path = os.path.join(args.out_dir, "encode_manifest.json")
        shards = []
        for r in range(world_size):
            dp = _shard_paths(args.out_dir, r)[3]
            waited = 0
            while not os.path.exists(dp):
                time.sleep(5)
                waited += 5
                if waited % 300 == 0:
                    print(f"[encode][rank0] waiting for shard{r}.done ({waited}s)...", flush=True)
            with open(dp) as f:
                shards.append(json.load(f))
        total_windows = sum(sh["n_windows"] for sh in shards)
        total_reads = sum(sh["n_reads"] for sh in shards)
        manifest = {
            "n_shards": world_size,
            "total_n_windows": int(total_windows),
            "total_n_reads": int(total_reads),
            "D": shards[0]["D"], "win": shards[0]["win"], "stride": shards[0]["stride"],
            "peak_gpu_mem_gb": round(max(sh["peak_gpu_mem_gb"] for sh in shards), 2),
            "shards": [
                {"rank": sh["rank"], "n_rows": sh["n_windows"],
                 "emb_file": sh["emb_file"], "win_file": sh["win_file"],
                 "rid_file": sh["rid_file"]}
                for sh in sorted(shards, key=lambda x: x["rank"])
            ],
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[encode][rank0] wrote {manifest_path}: total_reads={total_reads} "
              f"total_windows={total_windows} D={manifest['D']}", flush=True)


if __name__ == "__main__":
    main()
