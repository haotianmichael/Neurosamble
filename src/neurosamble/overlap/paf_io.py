"""
Minimal PAF writers for the RawHash head-to-head (Step 2c).

PAF is the domain-standard format that ``uncalled pafstats`` scores, so both
SquiggleSeek and RawHash2 are judged by the same tool on the same ground truth
(no custom _covers criterion). 12 columns:

    qname qlen qstart qend strand tname tlen tstart tend nmatch alen mapq

Ground truth comes from the squigulator read id (true coordinates are known) —
strictly better than RawHash's original truth (minimap2 on basecalled reads),
which would let the cascade define perfection.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def mapq_from_scores(scores: List[float]) -> List[int]:
    """Min-max normalize confidence scores to integer MAPQ in [0, 60]."""
    if not scores:
        return []
    s = np.asarray(scores, dtype=float)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        return [60] * len(scores)
    return [int(round((v - lo) / (hi - lo) * 60)) for v in s]


def _paf_line(qname, qlen, qstart, qend, strand, tname, tlen, tstart, tend,
              nmatch, alen, mapq) -> str:
    return "\t".join(str(x) for x in (
        qname, qlen, qstart, qend, strand, tname, tlen, tstart, tend, nmatch, alen, mapq))


def _read_span_bp(read, samples_per_kmer: int) -> int:
    return max(1, int(round(len(read.signal) / samples_per_kmer)))


def write_ground_truth_paf(eval_reads, reference_seq, samples_per_kmer, path):
    tlen = len(reference_seq)
    with open(path, "w") as f:
        for r in eval_reads:
            span = _read_span_bp(r, samples_per_kmer)
            tstart = r.reference_start
            tend = r.reference_end if r.reference_end is not None else tstart + span
            tname = r.reference_name or "ref"
            f.write(_paf_line(r.id, span, 0, span, r.strand, tname, tlen,
                              tstart, tend, span, span, 60) + "\n")
    return path


def write_mapping_paf(entries: List[Tuple], reference_seq, samples_per_kmer, path):
    """entries: list of (read, reported_tstart, mapq, strand, score). The cosine
    score is appended as a ``cs:f:`` tag so the head-to-head can sweep a
    confidence threshold. Unmapped reads are simply absent (counted as FN)."""
    tlen = len(reference_seq)
    with open(path, "w") as f:
        for r, tstart, mapq, strand, score in entries:
            span = _read_span_bp(r, samples_per_kmer)
            tname = r.reference_name or "ref"
            line = _paf_line(r.id, span, 0, span, strand, tname, tlen,
                             tstart, tstart + span, span, span, mapq)
            f.write(f"{line}\tcs:f:{score:.6f}\n")
    return path