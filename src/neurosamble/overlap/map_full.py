"""
Neurosamble Phase 4 -- streaming IVF query + chaining -> neurosamble.paf.

Replaces the Phase-2 in-memory pair-anchor dict (O(N^2) RAM) with a per-query-read
streaming pass: memory is bounded by ONE read's anchors. The encoder is NOT loaded
here -- query vectors come straight from the encode memmap shards.

For each query read R (windows are contiguous per read in the global row order):
  * fetch R's window vectors from its shard memmap, ``index.search(nprobe, topk)``,
  * keep neighbors whose target read T satisfies ``T != R`` AND
    ``name(R) < name(T)`` -- self-excludes and dedups (A,B)/(B,A) in one condition,
    halving work,  (implemented with a precomputed lexicographic rank per read),
  * group anchors by T -> ``chain_anchors`` per (R,T) -> append surviving chains,
  * free R's anchor structures before the next read.

The neighbor filtering is vectorized with numpy (critical: at high topk the
per-neighbor Python loop is 28M*topk iterations and would dominate). Only the
per-pair chaining stays in Python.

topk MUST scale with coverage: each window has ~coverage true neighbors, so a
fixed small topk caps recall at ~topk/coverage.

FAISS on CPU by default; ``--faiss_gpu 1`` shards the index across all GPUs (exact
IVFFlat search, fp16 storage) -- required for full-scale high-topk throughput.

PAF format matches Phase 2: 12 std cols + ``mt:f:0.0`` tag on every line; qname/tname
are real read-ids; coords in bp = offset_samples // samples_per_kmer.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from neurosamble.overlap.chain import chain_anchors


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 streaming IVF query + chaining")
    p.add_argument("--index_dir", required=True, help="dir with ivf.index, windows.npy, read_ids.txt")
    p.add_argument("--encode_dir", required=True, help="dir with encode_manifest.json + emb shards")
    p.add_argument("--out_paf", required=True)
    p.add_argument("--nprobe", type=int, default=64)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--threads", type=int, default=0, help="faiss omp threads (0 = default)")
    p.add_argument("--samples_per_kmer", type=int, default=9)
    p.add_argument("--min_chaining_score", type=float, default=40.0)
    p.add_argument("--min_num_anchors", type=int, default=5)
    p.add_argument("--max_gap_bp", type=int, default=2500)
    p.add_argument("--bw_bp", type=int, default=5000)
    p.add_argument("--faiss_gpu", type=int, default=0,
                   help="1 = shard index across all GPUs (exact IVFFlat, fp16)")
    p.add_argument("--query_batch", type=int, default=65536,
                   help="max query rows per index.search call (chunks GPU temp mem)")
    p.add_argument("--gpu_temp_mb", type=int, default=8192,
                   help="FAISS GPU temp-memory pool per device (MB); must exceed a "
                        "single search's temp alloc or search raises TemporaryMemoryOverflow")
    return p.parse_args()


def load_read_table(path):
    """read_ids.txt lines 'gid<TAB>name<TAB>nsamp' -> arrays indexed by gid.

    Returns (id2name list, name_rank int64[max_gid+1], nsamp int64[max_gid+1]).
    name_rank is the lexicographic rank of each read's name (names are unique, so
    ranks are unique) -- the canonical R<T test becomes rank[R] < rank[T].
    """
    gids, names, nsamps = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gids.append(int(parts[0]))
            names.append(parts[1])
            nsamps.append(int(parts[2]))
    max_gid = max(gids)
    id2name = [None] * (max_gid + 1)
    nsamp = np.zeros(max_gid + 1, dtype=np.int64)
    for g, nm, ns in zip(gids, names, nsamps):
        id2name[g] = nm
        nsamp[g] = ns
    rank_of = {nm: i for i, nm in enumerate(sorted(names))}
    name_rank = np.full(max_gid + 1, -1, dtype=np.int64)
    for g, nm in zip(gids, names):
        name_rank[g] = rank_of[nm]
    return id2name, name_rank, nsamp


def filter_neighbors(r_gid, r_rank, nid, sc, qoff, win_gid, win_off, name_rank):
    """Vectorized keep of canonical cross-read neighbors for one query read.

    nid/sc/qoff are flattened [nq*topk] neighbor row-ids / scores / query offsets.
    Keeps rows where the neighbor is a real hit, a DIFFERENT read, and canonical
    (name(R) < name(T)  <=>  rank[T] > r_rank). Returns (t_gid, t_off, qoff, sc).
    """
    valid = nid >= 0
    nid = nid[valid]; sc = sc[valid]; qoff = qoff[valid]
    if nid.size == 0:
        z = np.empty(0, dtype=np.int64)
        return z, z, z, np.empty(0, dtype=np.float32)
    t_gid = win_gid[nid]
    keep = (t_gid != r_gid) & (name_rank[t_gid] > r_rank)
    return t_gid[keep], win_off[nid][keep], qoff[keep], sc[keep]


def batched_search(index, qvecs, topk, bs):
    """index.search in sub-batches of <= bs rows, concatenated in order.

    Bounds the FAISS GPU temporary allocation per call. Results are identical to a
    single un-chunked search (row order preserved by concatenation).
    """
    if bs <= 0 or qvecs.shape[0] <= bs:
        return index.search(qvecs, topk)
    all_scores, all_ids = [], []
    for i in range(0, qvecs.shape[0], bs):
        s, d = index.search(qvecs[i:i + bs], topk)
        all_scores.append(s)
        all_ids.append(d)
    return np.concatenate(all_scores, 0), np.concatenate(all_ids, 0)


def main():
    args = parse_args()
    spk = max(1, int(args.samples_per_kmer))

    import faiss

    if args.threads > 0:
        faiss.omp_set_num_threads(args.threads)

    with open(os.path.join(args.encode_dir, "encode_manifest.json")) as f:
        manifest = json.load(f)
    D = int(manifest["D"])
    win_bp = max(1, int(manifest["win"]) // spk)
    shards = manifest["shards"]

    # Global-row -> shard memmap mapping (row order == shard concatenation order).
    cum = [0]
    mmaps = []
    for sh in shards:
        n = int(sh["n_rows"])
        mm = np.memmap(os.path.join(args.encode_dir, sh["emb_file"]),
                       dtype=np.float32, mode="r", shape=(n, D))
        mmaps.append(mm)
        cum.append(cum[-1] + n)
    cum = np.asarray(cum)

    windows = np.load(os.path.join(args.index_dir, "windows.npy"))   # [N,2] (gid, off)
    win_gid = np.ascontiguousarray(windows[:, 0])
    win_off = np.ascontiguousarray(windows[:, 1])
    id2name, name_rank, nsamp = load_read_table(os.path.join(args.index_dir, "read_ids.txt"))
    N = windows.shape[0]

    index = faiss.read_index(os.path.join(args.index_dir, "ivf.index"))
    # Set nprobe on the CPU IVF index FIRST -- it propagates into the GPU clone.
    # (A sharded GPU IndexShards has no settable .nprobe / ParameterSpace hook.)
    try:
        index.nprobe = args.nprobe
    except Exception:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)
    gpu_res = []  # kept alive for the index's lifetime (must outlive all searches)
    if args.faiss_gpu:
        print("[map] sharding index across all GPUs (exact IVFFlat, fp16)", flush=True)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        temp_bytes = int(args.gpu_temp_mb) * 1024 * 1024
        try:
            # Explicit per-GPU resources so we can raise the temp-memory pool above
            # the (fixed, default ~1.5GB) ceiling that overflows at large nprobe/topk.
            ngpu = faiss.get_num_gpus()
            gpu_res = [faiss.StandardGpuResources() for _ in range(ngpu)]
            for r in gpu_res:
                r.setTempMemory(temp_bytes)
            index = faiss.index_cpu_to_gpu_multiple_py(gpu_res, index, co)
            print(f"[map] GPU temp mem = {args.gpu_temp_mb} MB x {ngpu} GPU(s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[map][warn] explicit GPU resources failed ({e}); "
                  f"falling back to index_cpu_to_all_gpus (default temp mem)", flush=True)
            index = faiss.index_cpu_to_all_gpus(index, co=co)
        # Best-effort: also set nprobe on the GPU shards (GPU has its own API).
        try:
            faiss.GpuParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)
        except Exception:
            pass
    print(f"[map] N_windows={N} nprobe={args.nprobe} topk={args.topk} spk={spk} win_bp={win_bp} "
          f"gpu={args.faiss_gpu} query_batch={args.query_batch} gpu_temp_mb={args.gpu_temp_mb} "
          f"thresholds(mcs={args.min_chaining_score},"
          f"mna={args.min_num_anchors},gap={args.max_gap_bp},bw={args.bw_bp})", flush=True)

    def shard_rows(i, j):
        """Vectors for global rows [i, j) -- a single read is within one shard."""
        s = int(np.searchsorted(cum, i, side="right") - 1)
        return np.ascontiguousarray(mmaps[s][i - cum[s]: j - cum[s]])

    # Per-read groups = maximal runs of equal gid in row order.
    boundaries = np.flatnonzero(np.diff(win_gid)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [N]])

    out = open(args.out_paf, "w")
    n_reads = n_pairs = n_chains = 0
    t0 = time.time()

    for i, j in zip(starts.tolist(), ends.tolist()):
        r_gid = int(win_gid[i])
        r_name = id2name[r_gid]
        if r_name is None:
            continue
        n_reads += 1
        r_rank = int(name_rank[r_gid])
        qvecs = shard_rows(i, j)
        q_offs = win_off[i:j]
        scores, ids = batched_search(index, qvecs, args.topk, args.query_batch)

        nid = ids.reshape(-1)
        sc = scores.reshape(-1)
        qoff = np.repeat(q_offs, args.topk)
        t_gid, t_off, q_kept, s_kept = filter_neighbors(
            r_gid, r_rank, nid, sc, qoff, win_gid, win_off, name_rank)

        if t_gid.size:
            # Group by target read, then chain each (R,T).
            order = np.argsort(t_gid, kind="stable")
            tg = t_gid[order]; to = t_off[order]; qo = q_kept[order]; ss = s_kept[order]
            seg = np.flatnonzero(np.diff(tg)) + 1
            seg_starts = np.concatenate([[0], seg])
            seg_ends = np.concatenate([seg, [tg.size]])
            qlen = max(1, int(nsamp[r_gid]) // spk)

            for a, b in zip(seg_starts.tolist(), seg_ends.tolist()):
                t = int(tg[a])
                anchors = {}
                qo_a = qo[a:b]; to_a = to[a:b]; ss_a = ss[a:b]
                for k in range(b - a):
                    key = (int(qo_a[k]), int(to_a[k]))
                    v = float(ss_a[k])
                    cur = anchors.get(key)
                    if cur is None or v > cur:
                        anchors[key] = v
                chains = chain_anchors(
                    anchors, spk, args.min_num_anchors, args.min_chaining_score,
                    max_gap_bp=args.max_gap_bp, bw_bp=args.bw_bp)
                if not chains:
                    continue
                t_name = id2name[t]
                tlen = max(1, int(nsamp[t]) // spk)
                wrote = False
                for ch in chains:
                    qs = max(0, min(ch.q_start, qlen)); qe = max(0, min(ch.q_end, qlen))
                    ts = max(0, min(ch.t_start, tlen)); te = max(0, min(ch.t_end, tlen))
                    if qe <= qs or te <= ts:
                        continue
                    span = qe - qs
                    mapq = min(60, max(1, int(round(ch.score / win_bp))))
                    cols = [r_name, qlen, qs, qe, "+", t_name, tlen, ts, te, span, span, mapq]
                    out.write("\t".join(str(c) for c in cols) + "\tmt:f:0.0\n")
                    n_chains += 1
                    wrote = True
                if wrote:
                    n_pairs += 1

        if (n_reads % 20000) == 0:
            dt = time.time() - t0
            print(f"[map] reads={n_reads} pairs={n_pairs} chains={n_chains} "
                  f"({n_reads/max(dt,1e-9):.0f} reads/s)", flush=True)

    out.close()
    query_sec = time.time() - t0

    import resource
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB->GB on Linux

    stats = {
        "n_reads": n_reads, "n_windows": int(N),
        "n_pairs_reported": n_pairs, "n_chains": n_chains,
        "query_sec": round(query_sec, 1), "peak_rss_gb": round(peak_rss_gb, 2),
        "nprobe": args.nprobe, "topk": args.topk, "faiss_gpu": int(args.faiss_gpu),
    }
    with open(os.path.join(args.index_dir, "query_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[map] DONE reads={n_reads} windows={N} pairs={n_pairs} chains={n_chains} "
          f"query={query_sec:.1f}s peak_rss={peak_rss_gb:.2f}GB -> {args.out_paf}", flush=True)


if __name__ == "__main__":
    main()
