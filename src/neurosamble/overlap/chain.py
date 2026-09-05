"""
Colinear chaining over Phase-1 read<->read anchors (Neurosamble Phase 2).

Phase 1 produced, for every canonical read-pair, an anchor dict
``{(q_off_samples, t_off_samples): score}`` (see ``overlap_probe.py``). This
module turns those scattered anchors into overlap CHAINS with a standard
minimap2-style colinear dynamic program, emitting ALL high-scoring chains per
pair (all-vs-all overlap reports many overlaps, so one pair can carry several).

Score model (deliberately simple, documented so the thresholds are legible):
  * Offsets are converted to bp via ``off // samples_per_kmer`` and sorted by q.
  * Two anchors chain only if BOTH q and t strictly increase (colinear,
    same-strand), the per-axis gap is within ``max_gap_bp``, and the colinearity
    deviation ``|Δq - Δt|`` is within the band ``bw_bp``.
  * Per-anchor reward is a CONSTANT ≈ the window span in bp
    (``WIN_SAMPLES // samples_per_kmer``); the transition subtracts a gap cost
    that is LINEAR in ``|Δq - Δt|`` (coefficient 1.0, i.e. 1 score unit per bp of
    deviation). So a chain score is roughly ``reward * n_anchors - Σ deviations``
    and ``min_chaining_score`` is expressed in those same (bp-like) units.

Torch-free (numpy not even required): safe to unit-test without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Phase-1 tiling window length in samples (encoder default; see overlap_index).
# Used only to size the constant per-anchor reward in bp.
WIN_SAMPLES = 2000


@dataclass
class Chain:
    """One colinear overlap chain between a canonical read-pair (all bp units)."""
    score: float
    n_anchors: int
    q_start: int
    q_end: int
    t_start: int
    t_end: int


def _best_chain(
    pts: List[Tuple[int, int]],
    anchor_reward: float,
    max_gap_bp: int,
    bw_bp: int,
) -> Tuple[float, List[int]]:
    """Best single colinear chain over ``pts`` (already sorted by q).

    Returns ``(score, indices_into_pts)``. Standard O(n^2) chaining DP:
    ``f[i] = anchor_reward + max_j ( f[j] - gap_cost(j->i) )`` over valid
    predecessors j (q and t strictly increasing, within gap/band limits).
    """
    n = len(pts)
    if n == 0:
        return 0.0, []
    f = [anchor_reward] * n
    pre = [-1] * n
    for i in range(n):
        qi, ti = pts[i]
        best = anchor_reward
        best_j = -1
        for j in range(i):
            qj, tj = pts[j]
            dq = qi - qj
            dt = ti - tj
            if dq <= 0 or dt <= 0:
                continue  # must be strictly colinear (same-strand, increasing)
            if dq > max_gap_bp or dt > max_gap_bp:
                continue  # gap too large to be one overlap
            dev = abs(dq - dt)
            if dev > bw_bp:
                continue  # outside the colinearity band
            sc = f[j] + anchor_reward - dev  # linear gap cost, coefficient 1.0
            if sc > best:
                best = sc
                best_j = j
        f[i] = best
        pre[i] = best_j

    end = max(range(n), key=lambda k: f[k])
    idxs: List[int] = []
    k = end
    while k != -1:
        idxs.append(k)
        k = pre[k]
    idxs.reverse()
    return f[end], idxs


def chain_anchors(
    anchors: Dict[Tuple[int, int], float],
    samples_per_kmer: int,
    min_num_anchors: int,
    min_chaining_score: float,
    max_gap_bp: int = 2500,
    bw_bp: int = 5000,
) -> List[Chain]:
    """Chain a single pair's anchors into all surviving overlap chains.

    ``anchors`` is the Phase-1 per-pair dict ``{(q_off_samples, t_off_samples):
    score}``. We repeatedly extract the best colinear chain, keep it if it clears
    both thresholds, remove its anchors, and repeat -- so every distinct
    high-scoring overlap between the pair is reported (all-vs-all).

    A chain is kept iff it has ``>= min_num_anchors`` anchors AND
    ``score >= min_chaining_score``. Returns chains in descending discovery order.
    """
    spk = max(1, int(samples_per_kmer))
    win_bp = max(1, WIN_SAMPLES // spk)
    anchor_reward = float(win_bp)

    # Convert to bp and sort by (q_bp, t_bp). Duplicate points are harmless.
    pts = sorted((q_off // spk, t_off // spk) for (q_off, t_off) in anchors.keys())

    chains: List[Chain] = []
    remaining = list(pts)
    while len(remaining) >= min_num_anchors:
        score, idxs = _best_chain(remaining, anchor_reward, max_gap_bp, bw_bp)
        if len(idxs) < min_num_anchors or score < min_chaining_score:
            break  # best remaining chain fails a threshold -> none better exists
        chain_pts = [remaining[k] for k in idxs]
        qs = [p[0] for p in chain_pts]
        ts = [p[1] for p in chain_pts]
        chains.append(
            Chain(
                score=float(score),
                n_anchors=len(idxs),
                q_start=min(qs),
                q_end=max(qs) + win_bp,   # window extends win_bp beyond its start
                t_start=min(ts),
                t_end=max(ts) + win_bp,
            )
        )
        used = set(idxs)
        remaining = [p for k, p in enumerate(remaining) if k not in used]

    return chains