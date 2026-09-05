"""
Neurosamble Phase 4 -- scale-safe FAISS IVF index over ALL encode shards (CPU).

The Phase-2 store used an in-memory ``IndexFlatIP`` (quadratic search, ~42 GB at
full scale). Here we build a FAISS ``IndexIVFFlat`` (inner-product on the already
L2-normalized vectors == cosine) directly, NOT through the frozen
``SignalFaissStore``. Row-id -> (read_id, offset) is a plain concatenation of the
shard ``windows_shard*.npy`` in rank order, so a global ``windows.npy`` +
``read_ids.txt`` is the row->metadata table.

Checkpointed: if ``ivf.index`` + ``windows.npy`` + ``read_ids.txt`` exist, skip.

Index kept on CPU RAM. ``--index_type ivfflat`` (default, ~42 GB) or ``ivfpq``
(m=48 -> ~1.3 GB, recall cost). If IVFFlat OOMs on CPU RAM we STOP and report --
we never silently switch to PQ or downsample.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 IVF index build (CPU, checkpointed)")
    p.add_argument("--encode_dir", required=True, help="dir with encode_manifest.json + shards")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--index_type", choices=["ivfflat", "ivfpq"], default="ivfflat")
    p.add_argument("--pq_m", type=int, default=48, help="ivfpq subquantizers (D must be divisible)")
    p.add_argument("--train_sample", type=int, default=2_000_000, help="max vectors to train nlist")
    p.add_argument("--add_chunk", type=int, default=1_000_000, help="rows per add() call")
    p.add_argument("--threads", type=int, default=0, help="faiss omp threads (0 = leave default)")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def _shard_emb_memmap(encode_dir, sh, D):
    path = os.path.join(encode_dir, sh["emb_file"])
    n = int(sh["n_rows"])
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n, D))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    index_path = os.path.join(args.out_dir, "ivf.index")
    windows_path = os.path.join(args.out_dir, "windows.npy")
    read_ids_path = os.path.join(args.out_dir, "read_ids.txt")
    stats_path = os.path.join(args.out_dir, "index_stats.json")

    if all(os.path.exists(p) for p in (index_path, windows_path, read_ids_path)):
        print(f"[ivf] checkpoint present; skipping build ({index_path})", flush=True)
        return

    import faiss

    if args.threads > 0:
        faiss.omp_set_num_threads(args.threads)

    with open(os.path.join(args.encode_dir, "encode_manifest.json")) as f:
        manifest = json.load(f)
    D = int(manifest["D"])
    shards = manifest["shards"]
    total_nw = int(manifest["total_n_windows"])
    print(f"[ivf] D={D} total_windows={total_nw} shards={len(shards)} "
          f"index_type={args.index_type}", flush=True)

    # 1) Global row->metadata table: concat windows_shard*.npy + read_ids in rank order.
    win_parts = [np.load(os.path.join(args.encode_dir, sh["win_file"])) for sh in shards]
    windows = np.concatenate(win_parts, axis=0) if win_parts else np.zeros((0, 2), np.int64)
    np.save(windows_path, windows)
    with open(read_ids_path, "w") as out:
        for sh in shards:
            with open(os.path.join(args.encode_dir, sh["rid_file"])) as fin:
                for line in fin:
                    out.write(line)
    assert windows.shape[0] == total_nw, (windows.shape[0], total_nw)
    print(f"[ivf] wrote windows.npy ({windows.shape[0]} rows) + read_ids.txt", flush=True)

    # 2) nlist and quantizer.
    nlist = int(min(4 * math.sqrt(max(total_nw, 1)), 65536))
    nlist = max(1, nlist)
    quantizer = faiss.IndexFlatIP(D)
    if args.index_type == "ivfpq":
        if D % args.pq_m != 0:
            raise SystemExit(f"[ivf] D={D} not divisible by pq_m={args.pq_m}")
        index = faiss.IndexIVFPQ(quantizer, D, nlist, args.pq_m, 8, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    print(f"[ivf] nlist={nlist} metric=INNER_PRODUCT (cosine on normalized vectors)", flush=True)

    t0 = time.time()
    try:
        # 3) Train on a random sample (<= train_sample) drawn across shards.
        rng = np.random.default_rng(args.seed)
        per_shard = max(1, args.train_sample // max(1, len(shards)))
        train_parts = []
        for sh in shards:
            mm = _shard_emb_memmap(args.encode_dir, sh, D)
            n = mm.shape[0]
            take = min(per_shard, n)
            idx = np.sort(rng.choice(n, size=take, replace=False)) if take < n else np.arange(n)
            train_parts.append(np.ascontiguousarray(mm[idx]))
        train = np.concatenate(train_parts, axis=0).astype(np.float32)
        print(f"[ivf] training on {train.shape[0]} vectors...", flush=True)
        index.train(train)
        del train, train_parts

        # 4) Add every shard's vectors in shard order (chunked to bound memory).
        added = 0
        for sh in shards:
            mm = _shard_emb_memmap(args.encode_dir, sh, D)
            n = mm.shape[0]
            for start in range(0, n, args.add_chunk):
                chunk = np.ascontiguousarray(mm[start:start + args.add_chunk])
                index.add(chunk)
                added += chunk.shape[0]
            print(f"[ivf] added shard rank{sh['rank']} ({n} rows); total added={added}",
                  flush=True)
        assert added == total_nw == index.ntotal, (added, total_nw, index.ntotal)
    except (MemoryError, RuntimeError) as e:
        print(f"[ivf][STOP] index build failed (likely CPU-RAM OOM): {e}", flush=True)
        print("[ivf][STOP] NOT switching to PQ or downsampling -- rerun with "
              "--index_type ivfpq, or provide more RAM (D handles hardware).", flush=True)
        sys.exit(3)

    faiss.write_index(index, index_path)
    build_sec = time.time() - t0
    with open(stats_path, "w") as f:
        json.dump({
            "index_type": args.index_type, "nlist": nlist, "D": D,
            "total_n_windows": total_nw, "ntotal": int(index.ntotal),
            "build_sec": round(build_sec, 1),
        }, f, indent=2)
    print(f"[ivf] DONE ntotal={index.ntotal} nlist={nlist} build={build_sec:.1f}s "
          f"-> {index_path}", flush=True)


if __name__ == "__main__":
    main()