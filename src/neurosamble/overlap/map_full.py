"""
Neurosamble Phase 4 -- streaming IVF query + chaining -> neurosamble.paf.

Replaces the Phase-2 in-memory pair-anchor dict (O(N^2) RAM) with a per-query-read
streaming pass: memory is bounded by ONE read's anchors. The encoder is NOT loaded
here -- query vectors come straight from the encode memmap shards.

For each query read R (windows are contiguous per read in the global row order):
  * fetch R's window vectors from its shard memmap, ``index.search(nprobe, topk)``,
  * keep neighbors whose target read T satisfies ``T != R`` AND
    ``name(R) < name(T)`` -- self-excludes and dedups (A,B)/(B,A) in one condition,
  * group anchors by T -> chain each (R,T) -> append surviving chains.

GPU / single-vs-multi GPU
-------------------------
FAISS on CPU by default. ``--faiss_gpu 1`` puts the index on GPU:
  * ``--gpu_id -1`` (default): shard across ALL visible GPUs (multi-GPU).
  * ``--gpu_id N``          : put the whole index on ONE GPU N (shard=False).
Either way, ``CUDA_VISIBLE_DEVICES`` still restricts which physical GPUs are seen,
so ``CUDA_VISIBLE_DEVICES=0`` alone already forces single-GPU behaviour.

Chaining parallelism
--------------------
The per-read chaining (anchor-dict build + colinear DP + PAF formatting) is the
CPU bottleneck. ``--chain_workers`` distributes it across processes, ONE read per
unit of work, mirroring minimap2/rawhash2's read-level threading. The GPU search
stays on the main process. Results are byte-identical to the serial path: W=1 and
W>1 run the SAME chaining code; the pool's outputs are reassembled in read order.
The worker pool is forked BEFORE any CUDA init (so children are CUDA-clean and do
not inherit the large GPU index), and the big read-only tables are shared via fork
copy-on-write rather than pickled.

topk MUST scale with coverage: each window has ~coverage true neighbors, so a
fixed small topk caps recall at ~topk/coverage.

PAF format: 12 std cols + ``mt:f:0.0`` tag on every line; qname/tname are real
read-ids; coords in bp = offset_samples // samples_per_kmer.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from neurosamble.overlap.chain import chain_anchors


# ---------------------------------------------------------------------------
# Worker side (module-level so it is fork-inheritable / picklable). The large
# read-only tables live in _CTX, populated in main() BEFORE the pool is forked;
# forked children inherit _CTX via copy-on-write and never touch CUDA.
# ---------------------------------------------------------------------------
_CTX: dict = {}


def _chain_one(job):
    """Chain ONE read's grouped anchors -> (paf_text, n_pairs, n_chains).

    job = (r_name, qlen, tg, to, qo, ss) where tg/to/qo/ss are the read's
    neighbor arrays already filtered to canonical cross-read hits and sorted by
    target gid (tg ascending). Identical logic to the original serial loop.
    """
    r_name, qlen, tg, to, qo, ss = job
    id2name = _CTX["id2name"]; nsamp = _CTX["nsamp"]; spk = _CTX["spk"]
    win_bp = _CTX["win_bp"]; mna = _CTX["mna"]; mcs = _CTX["mcs"]
    gap = _CTX["gap"]; bw = _CTX["bw"]

    lines = []
    n_pairs = 0
    n_chains = 0
    if tg.size:
        seg = np.flatnonzero(np.diff(tg)) + 1
        seg_starts = np.concatenate([[0], seg])
        seg_ends = np.concatenate([seg, [tg.size]])
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
                anchors, spk, mna, mcs, max_gap_bp=gap, bw_bp=bw)
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
                lines.append("\t".join(str(c) for c in cols) + "\tmt:f:0.0\n")
                n_chains += 1
                wrote = True
            if wrote:
                n_pairs += 1
    return "".join(lines), n_pairs, n_chains


def _chain_batch(batch):
    """Chain a list of read-jobs -> (paf_text, n_pairs, n_chains)."""
    parts = []
    np_ = 0
    nc_ = 0
    for job in batch:
        s, p, c = _chain_one(job)
        parts.append(s)
        np_ += p
        nc_ += c
    return "".join(parts), np_, nc_


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
                   help="1 = put index on GPU (see --gpu_id for single vs sharded)")
    p.add_argument("--gpu_id", type=int, default=-1,
                   help="-1 = shard across all visible GPUs; N = single GPU N (shard=False)")
    p.add_argument("--chain_workers", type=int, default=0,
                   help="processes for read-level chaining (0 = auto=cpu-2; 1 = serial)")
    p.add_argument("--chain_batch", type=int, default=64,
                   help="reads per chaining task (amortizes IPC)")
    p.add_argument("--query_batch", type=int, default=65536,
                   help="max query rows per index.search call (chunks GPU temp mem)")
    p.add_argument("--gpu_temp_mb", type=int, default=8192,
                   help="FAISS GPU temp-memory pool per device (MB)")
    return p.parse_args()


def load_read_table(path):
    """read_ids.txt lines 'gid<TAB>name<TAB>nsamp' -> arrays indexed by gid."""
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
    """Vectorized keep of canonical cross-read neighbors for one query read."""
    valid = nid >= 0
    nid = nid[valid]; sc = sc[valid]; qoff = qoff[valid]
    if nid.size == 0:
        z = np.empty(0, dtype=np.int64)
        return z, z, z, np.empty(0, dtype=np.float32)
    t_gid = win_gid[nid]
    keep = (t_gid != r_gid) & (name_rank[t_gid] > r_rank)
    return t_gid[keep], win_off[nid][keep], qoff[keep], sc[keep]


def batched_search(index, qvecs, topk, bs):
    """index.search in sub-batches of <= bs rows, concatenated in order."""
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
    emb_dtype = np.float16 if manifest.get("dtype", "float32") == "float16" else np.float32

    # Global-row -> shard memmap mapping (row order == shard concatenation order).
    cum = [0]
    mmaps = []
    for sh in shards:
        n = int(sh["n_rows"])
        mm = np.memmap(os.path.join(args.encode_dir, sh["emb_file"]),
                       dtype=emb_dtype, mode="r", shape=(n, D))
        mmaps.append(mm)
        cum.append(cum[-1] + n)
    cum = np.asarray(cum)

    windows = np.load(os.path.join(args.index_dir, "windows.npy"))   # [N,2] (gid, off)
    win_gid = np.ascontiguousarray(windows[:, 0])
    win_off = np.ascontiguousarray(windows[:, 1])
    id2name, name_rank, nsamp = load_read_table(os.path.join(args.index_dir, "read_ids.txt"))
    N = windows.shape[0]

    # Per-read groups = maximal runs of equal gid in row order.
    boundaries = np.flatnonzero(np.diff(win_gid)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [N]])

    # Publish read-only chaining context BEFORE forking workers (fork inherits it).
    global _CTX
    _CTX = {
        "id2name": id2name, "nsamp": nsamp, "spk": spk, "win_bp": win_bp,
        "mna": args.min_num_anchors, "mcs": args.min_chaining_score,
        "gap": args.max_gap_bp, "bw": args.bw_bp,
    }

    # ---- chaining process pool: fork NOW, before any CUDA init and before the
    # large CPU index is read, so children are CUDA-clean and don't inherit it.
    W = args.chain_workers
    if W == 0:
        W = max(1, (os.cpu_count() or 2) - 2)
    pool = None
    if W > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
            pool = ctx.Pool(processes=W)   # forks all W workers immediately
            print(f"[map] chaining pool: {W} forked workers (read-level parallel)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[map][warn] could not fork chaining pool ({e}); serial chaining", flush=True)
            pool = None
            W = 1

    index = faiss.read_index(os.path.join(args.index_dir, "ivf.index"))
    try:
        index.nprobe = args.nprobe
    except Exception:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)

    gpu_res = []  # keep GPU resources alive for the index's lifetime
    if args.faiss_gpu:
        temp_bytes = int(args.gpu_temp_mb) * 1024 * 1024
        if args.gpu_id >= 0:
            print(f"[map] single GPU {args.gpu_id} (fp16), temp={args.gpu_temp_mb}MB", flush=True)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            res = faiss.StandardGpuResources()
            res.setTempMemory(temp_bytes)
            gpu_res = [res]
            index = faiss.index_cpu_to_gpu(res, int(args.gpu_id), index, co)
            try:
                faiss.GpuParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)
            except Exception:
                pass
        else:
            print("[map] sharding index across all visible GPUs (fp16)", flush=True)
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            try:
                ngpu = faiss.get_num_gpus()
                gpu_res = [faiss.StandardGpuResources() for _ in range(ngpu)]
                for r in gpu_res:
                    r.setTempMemory(temp_bytes)
                index = faiss.index_cpu_to_gpu_multiple_py(gpu_res, index, co)
                print(f"[map] GPU temp mem = {args.gpu_temp_mb} MB x {ngpu} GPU(s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[map][warn] explicit GPU resources failed ({e}); "
                      f"falling back to index_cpu_to_all_gpus", flush=True)
                index = faiss.index_cpu_to_all_gpus(index, co=co)
            try:
                faiss.GpuParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)
            except Exception:
                pass
        # Free the fp32/CPU index now (its ~tens of GB) before the query loop
        # faults the embedding memmap into page cache.
        import gc
        gc.collect()

    print(f"[map] N_windows={N} nprobe={args.nprobe} topk={args.topk} spk={spk} win_bp={win_bp} "
          f"gpu={args.faiss_gpu} gpu_id={args.gpu_id} chain_workers={W} "
          f"query_batch={args.query_batch} gpu_temp_mb={args.gpu_temp_mb} "
          f"thresholds(mcs={args.min_chaining_score},"
          f"mna={args.min_num_anchors},gap={args.max_gap_bp},bw={args.bw_bp})", flush=True)

    def shard_rows(i, j):
        """Vectors for global rows [i, j) -- a single read is within one shard."""
        s = int(np.searchsorted(cum, i, side="right") - 1)
        v = np.ascontiguousarray(mmaps[s][i - cum[s]: j - cum[s]])
        return v.astype(np.float32, copy=False)   # FAISS search needs float32

    out = open(args.out_paf, "w")
    n_reads = n_pairs = n_chains = 0
    t0 = time.time()

    # Bounded in-flight futures preserve read order (FIFO) => byte-identical PAF.
    import collections
    inflight = collections.deque()
    max_inflight = max(2, W * 3)
    batch = []

    def drain_one():
        nonlocal n_pairs, n_chains
        s_, p_, c_ = inflight.popleft().get()
        out.write(s_)
        n_pairs += p_
        n_chains += c_

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
            order = np.argsort(t_gid, kind="stable")
            tg = np.ascontiguousarray(t_gid[order]).astype(np.int32, copy=False)
            to = np.ascontiguousarray(t_off[order]).astype(np.int32, copy=False)
            qo = np.ascontiguousarray(q_kept[order]).astype(np.int32, copy=False)
            ss = np.ascontiguousarray(s_kept[order]).astype(np.float32, copy=False)
            qlen = max(1, int(nsamp[r_gid]) // spk)
            job = (r_name, qlen, tg, to, qo, ss)

            if pool is not None:
                batch.append(job)
                if len(batch) >= args.chain_batch:
                    inflight.append(pool.apply_async(_chain_batch, (batch,)))
                    batch = []
                    while len(inflight) >= max_inflight:
                        drain_one()
            else:
                s_, p_, c_ = _chain_batch([job])
                out.write(s_)
                n_pairs += p_
                n_chains += c_

        if (n_reads % 20000) == 0:
            dt = time.time() - t0
            print(f"[map] reads={n_reads} pairs>={n_pairs} chains>={n_chains} "
                  f"({n_reads/max(dt,1e-9):.0f} reads/s)", flush=True)

    # flush the tail
    if pool is not None:
        if batch:
            inflight.append(pool.apply_async(_chain_batch, (batch,)))
            batch = []
        while inflight:
            drain_one()
        pool.close()
        pool.join()

    out.close()
    query_sec = time.time() - t0

    import resource
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB->GB on Linux

    stats = {
        "n_reads": n_reads, "n_windows": int(N),
        "n_pairs_reported": n_pairs, "n_chains": n_chains,
        "query_sec": round(query_sec, 1), "peak_rss_gb": round(peak_rss_gb, 2),
        "nprobe": args.nprobe, "topk": args.topk, "faiss_gpu": int(args.faiss_gpu),
        "gpu_id": int(args.gpu_id), "chain_workers": int(W),
    }
    with open(os.path.join(args.index_dir, "query_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[map] DONE reads={n_reads} windows={N} pairs={n_pairs} chains={n_chains} "
          f"query={query_sec:.1f}s peak_rss={peak_rss_gb:.2f}GB -> {args.out_paf}", flush=True)


if __name__ == "__main__":
    main()