"""
Sanitize an overlap PAF before feeding it to miniasm.

Two failure modes are handled:

1. Degenerate/malformed lines -- self-hits (qname==tname), zero-length or reversed
   intervals, <12 cols, non-numeric coords, or a strand that is not +/-.

2. Scale mismatch with the basecalled reads. Signal-domain overlappers report read
   lengths in a signal-derived unit (Neurosamble: ``signal_len // samples_per_kmer``
   ~= 1.28x bases; rawhash2: its own per-read base estimate), so their coordinates
   sit on a stretched scale vs the basecalled genome length. When ``--reads_fasta``
   is given we rescale by a SINGLE GLOBAL constant
   ``c = median(fasta_len[r] / native_len[r])`` over reads shared by the PAF and the
   FASTA. Multiplying every qlen/qs/qe/tlen/ts/te by the same ``c`` is a similarity
   transform: it preserves the overlap geometry EXACTLY (so miniasm's graph topology
   is unchanged) and only shifts the overall scale back into base space. This
   replaces the earlier PER-READ rescale, which set qlen=fasta_len line-by-line and
   -- because rawhash2's length/base ratio varies read-to-read -- distorted the
   geometry and shattered assemblies.

This only affects the assembly input (the ``*.clean.paf``); the overlap P/R/F1 is
scored on the original PAF, so those numbers are untouched.

Usage:
  python evaluate/sanitize_paf.py --in_paf in.paf --out_paf out.clean.paf [--reads_fasta reads.fa]
"""
from __future__ import annotations

import argparse
import statistics


def read_fasta_lengths(path):
    """name -> sequence length in bases (first header token as the id)."""
    lengths = {}
    name = None
    n = 0
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = n
                name = line[1:].split()[0]
                n = 0
            else:
                n += len(line.strip())
    if name is not None:
        lengths[name] = n
    return lengths


def read_native_lengths(in_paf):
    """read -> native PAF length (qlen when it is the query, tlen when target)."""
    native = {}
    with open(in_paf) as f:
        for line in f:
            if not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9:
                continue
            try:
                ql, tl = int(c[1]), int(c[6])
            except ValueError:
                continue
            q, t = c[0], c[5]
            if ql > native.get(q, 0):
                native[q] = ql
            if tl > native.get(t, 0):
                native[t] = tl
    return native


def global_scale(native_lengths, fasta_lengths):
    """Single global c = median(fasta_len[r] / native_len[r]) over shared reads."""
    ratios = [fasta_lengths[r] / nl
              for r, nl in native_lengths.items()
              if nl > 0 and r in fasta_lengths]
    if not ratios:
        return 1.0, 0
    return float(statistics.median(ratios)), len(ratios)


def sanitize(in_paf, out_paf, fasta_lengths=None):
    """Filter degenerate lines; if fasta_lengths given, apply one GLOBAL scale c."""
    scale = None
    if fasta_lengths is not None:
        native = read_native_lengths(in_paf)
        scale, n_used = global_scale(native, fasta_lengths)
        print(f"[sanitize] global scale c={scale:.6f} (median over {n_used} shared reads)",
              flush=True)

    kept = dropped = 0
    with open(in_paf) as fin, open(out_paf, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 12:
                dropped += 1
                continue
            qname, tname, strand = f[0], f[5], f[4]
            if qname == tname or strand not in ("+", "-"):
                dropped += 1
                continue
            try:
                qlen, qs, qe = int(f[1]), int(f[2]), int(f[3])
                tlen, ts, te = int(f[6]), int(f[7]), int(f[8])
            except ValueError:
                dropped += 1
                continue

            if scale is not None:
                # Similarity transform: same constant on every coordinate/length so
                # the overlap geometry is preserved. Never set qlen = fasta_len.
                qlen, qs, qe = round(qlen * scale), round(qs * scale), round(qe * scale)
                tlen, ts, te = round(tlen * scale), round(ts * scale), round(te * scale)
                f[1], f[2], f[3] = str(qlen), str(qs), str(qe)
                f[6], f[7], f[8] = str(tlen), str(ts), str(te)

            if not (0 <= qs < qe <= qlen and 0 <= ts < te <= tlen):
                dropped += 1
                continue
            fout.write("\t".join(f) + "\n")
            kept += 1
    return kept, dropped


def parse_args():
    p = argparse.ArgumentParser(description="Sanitize an overlap PAF for miniasm")
    p.add_argument("--in_paf", required=True)
    p.add_argument("--out_paf", required=True)
    p.add_argument("--reads_fasta", default=None,
                   help="apply a single GLOBAL scale c=median(fasta_len/native_len) to "
                        "all coords (similarity transform; preserves overlap geometry)")
    return p.parse_args()


def main():
    args = parse_args()
    fasta_lengths = read_fasta_lengths(args.reads_fasta) if args.reads_fasta else None
    kept, dropped = sanitize(args.in_paf, args.out_paf, fasta_lengths)
    print(f"[sanitize] {args.in_paf}: kept={kept} dropped={dropped} -> {args.out_paf}"
          + (" (globally rescaled to base space)" if fasta_lengths else ""), flush=True)


if __name__ == "__main__":
    main()
