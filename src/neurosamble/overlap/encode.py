"""
Neurosamble Phase 4 -- 2-GPU data-parallel read-window encode -> disk (once).

Encodes ALL windows across both GPUs and persists them to per-rank shard files,
so the index/query steps never touch the encoder again.

Launch (world_size = NUM_GPUS):
    torchrun --nproc_per_node=2 -m neurosamble.overlap.encode --real_reads ... --out_dir ...

Per rank: bind cuda:{LOCAL_RANK}, stream the blow5 (pyslow5, pA=True), keep reads
with ``gid % world_size == rank`` (gid = file position, stable & identical across
ranks), tile, encode, append to OWN shard files.

PERF NOTES (encode acceleration):
  * A background READER THREAD does the blow5 decode + tiling so the heavy VBZ
    decompression overlaps the GPU forward (the old path was fully serial and left
    the GPU >95% idle). The reader stays single-threaded and sequential, so gid
    assignment and output row order are byte-for-byte identical to the serial path
    (windows.npy rows still align with the emb shard and with the query grouping).
  * ``--encode_batch`` is now the GPU forward batch (default 2048, was effectively
    256). GPU peak was only 2.77 GB, so there is ~10x headroom on a 32 GB card.
  * ``--fp16_out`` stores embeddings as float16 (half the shard files, half the
    query-side memmap RAM and -- with an fp16 index -- half the index RAM). The
    stored dtype is recorded in the manifest; downstream readers honor it.

Shard files (per rank r):
  embeddings_shard{r}.f32  raw embeddings [n_win_r, D], L2-normalized (dtype per manifest)
  windows_shard{r}.npy     int64 [n_win_r, 2] rows (global_read_id, offset_samples)
  read_ids_shard{r}.txt    lines: "global_read_id<TAB>read_name<TAB>n_samples"
  shard{r}.done            completion sentinel with counts (resume checkpoint)
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time

import numpy as np

from neurosamble.overlap.tile import tile_read


def _dist_env():
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
    p.add_argument("--encode_batch", type=int, default=2048,
                   help="GPU forward batch (windows per encoder call)")
    p.add_argument("--fp16_out", action="store_true",
                   help="store embeddings as float16 (half disk + half query RAM)")
    p.add_argument("--no_amp", action="store_true",
                   help="disable bf16 autocast in the encoder forward")
    p.add_argument("--prefetch", type=int, default=8,
                   help="reads buffered by the decode thread (0 = serial, no thread)")
    return p.parse_args()


def _shard_paths(out_dir, rank):
    return (
        os.path.join(out_dir, f"embeddings_shard{rank}.f32"),
        os.path.join(out_dir, f"windows_shard{rank}.npy"),
        os.path.join(out_dir, f"read_ids_shard{rank}.txt"),
        os.path.join(out_dir, f"shard{rank}.done"),
    )


def _reader_worker(reads_path, world_size, rank, win, stride, q):
    """Sequential blow5 decode + tiling in a background thread (order-preserving)."""
    import pyslow5
    s = pyslow5.Open(reads_path, "r")
    try:
        gid = -1
        for rec in s.seq_reads(pA=True):          # pA=True == read_blow5's signal
            gid += 1
            if (gid % world_size) != rank:
                continue
            sig = np.asarray(rec["signal"], dtype=np.float32)
            tiles = tile_read(sig, win=win, stride=stride)
            if not tiles:
                continue
            q.put((gid, rec["read_id"], int(len(sig)), tiles))
    finally:
        q.put(None)
        s.close()


def main():
    args = parse_args()
    rank, world_size, local_rank = _dist_env()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from neurosamble.overlap.eval_model import load_encoder

    out_dtype = np.float16 if args.fp16_out else np.float32
    dtype_str = "float16" if args.fp16_out else "float32"

    emb_path, win_path, rid_path, done_path = _shard_paths(args.out_dir, rank)

    if os.path.exists(done_path) and os.path.exists(emb_path) and os.path.exists(win_path):
        print(f"[encode][rank{rank}] shard complete; skipping (checkpoint: {done_path})",
              flush=True)
    else:
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            torch.cuda.reset_peak_memory_stats(local_rank)
        model, cfg = load_encoder(
            args.load_encoder, device,
            batch_size=args.encode_batch, use_amp=(not args.no_amp))
        D = int(model.get_sentence_embedding_dimension())
        print(f"[encode][rank{rank}] world_size={world_size} device={device} D={D} "
              f"win={args.win} stride={args.stride} encode_batch={args.encode_batch} "
              f"out_dtype={dtype_str} (reads where gid%{world_size}=={rank})", flush=True)

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
            vecs = model.encode(buf_sigs, out_dtype=out_dtype)   # [b, D], L2-normalized
            vecs.tofile(emb_f)
            win_chunks.append(np.asarray(buf_meta, dtype=np.int64))
            buf_sigs, buf_meta = [], []

        # ---- decode in a background thread; encode on the main thread -------- #
        if args.prefetch > 0:
            q: "queue.Queue" = queue.Queue(maxsize=args.prefetch)
            reader = threading.Thread(
                target=_reader_worker,
                args=(args.real_reads, world_size, rank, args.win, args.stride, q),
                daemon=True)
            reader.start()

            def read_stream():
                while True:
                    item = q.get()
                    if item is None:
                        break
                    yield item
            stream = read_stream()
        else:
            import pyslow5
            s = pyslow5.Open(args.real_reads, "r")

            def serial_stream():
                gid = -1
                for rec in s.seq_reads(pA=True):
                    gid += 1
                    if (gid % world_size) != rank:
                        continue
                    sig = np.asarray(rec["signal"], dtype=np.float32)
                    tiles = tile_read(sig, win=args.win, stride=args.stride)
                    if not tiles:
                        continue
                    yield (gid, rec["read_id"], int(len(sig)), tiles)
                s.close()
            stream = serial_stream()

        for gid, read_id, nsamp, tiles in stream:
            rid_f.write(f"{gid}\t{read_id}\t{nsamp}\n")
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
                "D": D, "win": args.win, "stride": args.stride, "dtype": dtype_str,
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
            "dtype": shards[0].get("dtype", "float32"),
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
              f"total_windows={total_windows} D={manifest['D']} dtype={manifest['dtype']}",
              flush=True)


if __name__ == "__main__":
    main()