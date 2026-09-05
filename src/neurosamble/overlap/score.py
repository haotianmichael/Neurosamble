"""
Dependency-free scorer for the Neurosamble overlap -> assembly pipeline.

ESA's pipeline shelled out to UNCALLED's ``pafstats.py`` (via ``$SCRIPTS_DIR``)
for overlap accuracy and to ``compute_aun.py`` / ``evaluate_gfa.py`` for
assembly contiguity. Those scripts are not vendored in this repo, so this module
replaces them with a small self-contained scorer (Python stdlib only -- no
faiss / numpy / torch), covering the two numbers that matter for a head-to-head:

1. Overlap-level precision / recall / F1 at the READ-PAIR level. A tool overlap
   is a true positive iff the same (query, target) read-pair -- order-canonicalized
   so (A,B) == (B,A) -- also appears in the truth PAF (``minimap2 -x ava-ont``).
   Self-hits (qname == tname) are ignored on both sides.

     precision = |tool_pairs & truth_pairs| / |tool_pairs|
     recall    = |tool_pairs & truth_pairs| / |truth_pairs|
     F1        = 2PR / (P + R)

2. Assembly contiguity from a miniasm GFA: number of unitigs, longest unitig,
   N50, and auN = Sum(len^2) / Sum(len). Sequence lengths come from the GFA
   ``S`` lines (the inline sequence, or an ``LN:i:`` tag when the sequence is ``*``).

CLI:
  python -m neurosamble.overlap.score --tool_paf tool.paf --truth_paf truth.paf [--gfa asm.gfa]
"""
from __future__ import annotations

import argparse
import json


# --------------------------------------------------------------------------- #
# Overlap P/R/F1 (read-pair level)
# --------------------------------------------------------------------------- #
def read_pairs(paf_path):
    """Set of order-canonicalized (query, target) read-pairs in a PAF.

    Column 0 is qname, column 5 is tname (standard PAF). Self-hits and malformed
    lines (<6 columns) are skipped. Canonicalizing each pair with ``frozenset``-
    like sorted tuples makes (A,B) and (B,A) the same overlap, so the count is of
    distinct read-pairs, not directed lines.
    """
    pairs = set()
    with open(paf_path) as f:
        for line in f:
            if not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 6:
                continue
            q, t = c[0], c[5]
            if q == t:
                continue
            pairs.add((q, t) if q < t else (t, q))
    return pairs


def overlap_prf(tool_paf, truth_paf):
    """Precision / recall / F1 of ``tool_paf`` against ``truth_paf`` (read-pairs)."""
    tool = read_pairs(tool_paf)
    truth = read_pairs(truth_paf)
    tp = len(tool & truth)
    n_tool = len(tool)
    n_truth = len(truth)
    precision = tp / n_tool if n_tool else 0.0
    recall = tp / n_truth if n_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "n_tool_pairs": n_tool,
        "n_truth_pairs": n_truth,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# --------------------------------------------------------------------------- #
# Assembly contiguity (miniasm GFA)
# --------------------------------------------------------------------------- #
def gfa_segment_lengths(gfa_path):
    """Lengths of every ``S`` (segment/unitig) line in a GFA.

    A miniasm ``S`` line is ``S<TAB>name<TAB>seq[<TAB>tags...]``. The length is
    ``len(seq)`` when the sequence is present, else the ``LN:i:<n>`` tag value
    (miniasm writes ``LN:i`` alongside the inline sequence anyway).
    """
    lengths = []
    with open(gfa_path) as f:
        for line in f:
            if not line.startswith("S\t") and not line.startswith("S "):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 3:
                continue
            seq = c[2]
            n = None
            if seq and seq != "*":
                n = len(seq)
            else:
                for tag in c[3:]:
                    if tag.startswith("LN:i:"):
                        try:
                            n = int(tag[5:])
                        except ValueError:
                            n = None
                        break
            if n is not None and n > 0:
                lengths.append(n)
    return lengths


def contiguity(gfa_path):
    """Unitig count, longest, N50, and auN from a miniasm GFA."""
    lengths = gfa_segment_lengths(gfa_path)
    if not lengths:
        return {"n_unitigs": 0, "total_bp": 0, "longest": 0, "n50": 0, "aun": 0.0}
    total = sum(lengths)
    longest = max(lengths)
    aun = sum(l * l for l in lengths) / total

    # N50: shortest length L such that unitigs >= L cover >= half the assembly.
    cum = 0
    n50 = 0
    for l in sorted(lengths, reverse=True):
        cum += l
        if cum * 2 >= total:
            n50 = l
            break
    return {
        "n_unitigs": len(lengths),
        "total_bp": total,
        "longest": longest,
        "n50": n50,
        "aun": aun,
    }


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Dependency-free overlap P/R/F1 + miniasm GFA contiguity scorer")
    p.add_argument("--tool_paf", required=True, help="overlap PAF produced by the tool")
    p.add_argument("--truth_paf", required=True, help="minimap2 -x ava-ont truth PAF")
    p.add_argument("--gfa", default=None, help="optional miniasm GFA for contiguity")
    p.add_argument("--json", default=None, help="optional path to also dump results as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    result = {"overlap": overlap_prf(args.tool_paf, args.truth_paf)}

    o = result["overlap"]
    print("==== overlap accuracy (read-pair level) ====")
    print(f"  tool_pairs   = {o['n_tool_pairs']}")
    print(f"  truth_pairs  = {o['n_truth_pairs']}")
    print(f"  true_positive= {o['tp']}")
    print(f"  precision    = {o['precision']*100:6.2f}%")
    print(f"  recall       = {o['recall']*100:6.2f}%")
    print(f"  f1           = {o['f1']*100:6.2f}%")

    if args.gfa:
        c = contiguity(args.gfa)
        result["contiguity"] = c
        print("==== assembly contiguity (miniasm GFA) ====")
        print(f"  n_unitigs = {c['n_unitigs']}")
        print(f"  total_bp  = {c['total_bp']}")
        print(f"  longest   = {c['longest']}")
        print(f"  N50       = {c['n50']}")
        print(f"  auN       = {c['aun']:.1f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[score] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
