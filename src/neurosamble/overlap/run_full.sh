#!/usr/bin/env bash
# =============================================================================
# Neurosamble -- FULL-SCALE all-vs-all overlap -> assembly head-to-head.
#
# Ported from ESA evaluate/run_neurosamble_full.sh onto the neurosamble.overlap
# package. The Neurosamble map step is the scale path:
#   NUM_GPUS-sharded encode -> CPU IVF index (checkpointed) -> streaming query.
# Rawsamble / minimap2-truth / miniasm / scoring mirror the reference pipeline.
# All outputs live under OUTDIR; encode+index are checkpointed so a re-run
# RESUMES rather than recomputes. The encoder is NOT loaded during query.
#
# Stages:
#   (a) encode   : torchrun -m neurosamble.overlap.encode        (NUM_GPUS GPUs)
#   (b) index    : python  -m neurosamble.overlap.index_ivf      (CPU, checkpointed)
#   (c) map      : python  -m neurosamble.overlap.map_full  -> neurosamble.paf
#   (d) rawsamble: rawhash2 -x ava                          -> rawsamble.paf
#   (e) truth    : minimap2 -x ava-ont --for-only reads reads-> mm2_overlaps.paf
#   (f) assemble : sanitize_paf + miniasm for each PAF      -> <tag>.gfa
#   (g) score    : python  -m neurosamble.overlap.score
#
# Positional args:
#   1 OUTDIR       output root (checkpoints live here; not timestamped)
#   2 REAL_BLOW5   full blow5/slow5 (ALL reads)  -- encode + rawsamble input
#   3 READS_FASTA  full basecalled reads (truth + miniasm sequences)
#   4 PORE         (optional) ONT pore model for rawhash2 -p (needed if DO_RAWSAMBLE=1)
#
# Required env: LOAD_ENCODER=<encoder .pt>  (default: data/real_encoder_v1.pt)
# =============================================================================
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 OUTDIR REAL_BLOW5 READS_FASTA [PORE]" >&2
  echo "       (env: LOAD_ENCODER=<encoder.pt>; defaults to data/real_encoder_v1.pt)" >&2
  exit 2
fi

OUTDIR="$1"; REAL_BLOW5="$2"; READS_FASTA="$3"; PORE="${4:-}"

# --- repo layout: this script lives at src/neurosamble/overlap/run_full.sh ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../src/neurosamble/overlap
SRC="$(cd "$HERE/../.." && pwd)"                          # .../src
REPO="$(cd "$SRC/.." && pwd)"                             # repo root
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"

# --- encoder (default: finetune output under P.OUT_DIR / data) --------------- #
DEFAULT_ENCODER="${NEUROSAMBLE_OUT_DIR:-$REPO/data}/real_encoder_v1.pt"
LOAD_ENCODER="${LOAD_ENCODER:-$DEFAULT_ENCODER}"
if [[ ! -s "$LOAD_ENCODER" ]]; then
  echo "[full] LOAD_ENCODER not found: $LOAD_ENCODER (set LOAD_ENCODER=<encoder.pt>)" >&2
  exit 2
fi

# --- knobs (same defaults as ESA) ------------------------------------------- #
NUM_GPUS="${NUM_GPUS:-2}"
NPROBE="${NPROBE:-64}"
TOPK="${TOPK:-10}"
INDEX_TYPE="${INDEX_TYPE:-ivfflat}"
DO_ASSEMBLY="${DO_ASSEMBLY:-1}"
DO_RAWSAMBLE="${DO_RAWSAMBLE:-1}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 8)}"
SPK="${SAMPLES_PER_KMER:-9}"

WIN="${WIN:-2000}"
STRIDE="${STRIDE:-1000}"
MIN_NUM_ANCHORS="${MIN_NUM_ANCHORS:-5}"
MIN_CHAINING_SCORE="${MIN_CHAINING_SCORE:-40}"
MAX_GAP_BP="${MAX_GAP_BP:-2500}"
BW_BP="${BW_BP:-5000}"

FAISS_GPU="${FAISS_GPU:-0}"
QUERY_BATCH="${QUERY_BATCH:-65536}"
GPU_TEMP_MB="${GPU_TEMP_MB:-8192}"

# --- external binaries (configurable) --------------------------------------- #
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
MINIMAP2="${MINIMAP2:-minimap2}"
MINIASM="${MINIASM:-miniasm}"
RAWHASH2="${RAWHASH2:-rawhash2}"
RAWHASH_PRESET="${RAWHASH_PRESET:-}"     # e.g. "--r10" for R10.4.1 data (else R9 defaults)

mkdir -p "$OUTDIR" "$OUTDIR/encode" "$OUTDIR/index"
NEURO_PAF="$OUTDIR/neurosamble.paf"
RAW_PAF="$OUTDIR/rawsamble.paf"
TRUTH_PAF="$OUTDIR/mm2_overlaps.paf"

echo "[full] OUTDIR=$OUTDIR NUM_GPUS=$NUM_GPUS NPROBE=$NPROBE TOPK=$TOPK INDEX_TYPE=$INDEX_TYPE"
echo "[full] FAISS_GPU=$FAISS_GPU DO_ASSEMBLY=$DO_ASSEMBLY DO_RAWSAMBLE=$DO_RAWSAMBLE THREADS=$THREADS SPK=$SPK"
echo "[full] LOAD_ENCODER=$LOAD_ENCODER win=$WIN stride=$STRIDE"
echo "[full] chaining: mna=$MIN_NUM_ANCHORS mcs=$MIN_CHAINING_SCORE gap=$MAX_GAP_BP bw=$BW_BP"

# --------------------------------------------------------------------------- #
# 1) Neurosamble scale path: encode (NUM_GPUS) -> IVF (CPU) -> streaming query
# --------------------------------------------------------------------------- #
if [[ "${REUSE_NEURO_PAF:-0}" == "1" && -s "$NEURO_PAF" ]]; then
  echo "[full] REUSE_NEURO_PAF=1 and $NEURO_PAF present -> skip encode/index/query"
else
  echo "[full] === (a) encode (${NUM_GPUS}-GPU sharded) ==="
  "$TORCHRUN" --nproc_per_node="$NUM_GPUS" -m neurosamble.overlap.encode \
    --real_reads "$REAL_BLOW5" --load_encoder "$LOAD_ENCODER" \
    --out_dir "$OUTDIR/encode" --win "$WIN" --stride "$STRIDE" \
    2>&1 | tee "$OUTDIR/encode.log"

  echo "[full] === (b) IVF index build (CPU, checkpointed) ==="
  "$PYTHON" -m neurosamble.overlap.index_ivf \
    --encode_dir "$OUTDIR/encode" --out_dir "$OUTDIR/index" \
    --index_type "$INDEX_TYPE" --threads "$THREADS" \
    2>&1 | tee "$OUTDIR/index.log"

  echo "[full] === (c) streaming query + chaining ==="
  "$PYTHON" -m neurosamble.overlap.map_full \
    --index_dir "$OUTDIR/index" --encode_dir "$OUTDIR/encode" \
    --out_paf "$NEURO_PAF" --nprobe "$NPROBE" --topk "$TOPK" \
    --threads "$THREADS" --samples_per_kmer "$SPK" --faiss_gpu "$FAISS_GPU" \
    --min_num_anchors "$MIN_NUM_ANCHORS" --min_chaining_score "$MIN_CHAINING_SCORE" \
    --max_gap_bp "$MAX_GAP_BP" --bw_bp "$BW_BP" \
    --query_batch "$QUERY_BATCH" --gpu_temp_mb "$GPU_TEMP_MB" \
    2>&1 | tee "$OUTDIR/query.log"
fi

# --------------------------------------------------------------------------- #
# 2) (d) Rawsamble on the SAME full blow5
# --------------------------------------------------------------------------- #
if [[ "$DO_RAWSAMBLE" != "0" ]]; then
  echo "[full] === (d) Rawsamble (rawhash2 -x ava) ==="
  if [[ "${REUSE_RAW_PAF:-0}" == "1" && -s "$RAW_PAF" ]]; then
    echo "[full] reuse existing rawsamble.paf (topk-independent): $RAW_PAF"
  elif [[ -z "$PORE" ]]; then
    echo "[full] no PORE model given -> skipping rawsamble baseline"
  else
    "$RAWHASH2" -x ava $RAWHASH_PRESET -t "$THREADS" -p "$PORE" -d "$OUTDIR/rawsamble_idx" "$REAL_BLOW5" \
      2>&1 | tee "$OUTDIR/rawsamble_index.log"
    "$RAWHASH2" -x ava $RAWHASH_PRESET -t "$THREADS" "$OUTDIR/rawsamble_idx" "$REAL_BLOW5" \
      > "$RAW_PAF" 2> "$OUTDIR/rawsamble_map.log"
  fi
fi

# --------------------------------------------------------------------------- #
# 3) (e) Overlap truth (minimap2 ava-ont, forward-only) on the full FASTA
# --------------------------------------------------------------------------- #
echo "[full] === (e) minimap2 ava-ont overlap truth ==="
if [[ "${REUSE_TRUTH_PAF:-0}" == "1" && -s "$TRUTH_PAF" ]]; then
  echo "[full] reuse existing mm2_overlaps.paf (topk-independent truth): $TRUTH_PAF"
else
  "$MINIMAP2" -x ava-ont --for-only -t "$THREADS" "$READS_FASTA" "$READS_FASTA" \
    > "$TRUTH_PAF" 2> "$OUTDIR/mm2_overlaps.log"
fi

# --------------------------------------------------------------------------- #
# 4) (f) Assembly + (g) scoring
# --------------------------------------------------------------------------- #
declare -A GFA_OF
if [[ "$DO_ASSEMBLY" != "0" ]]; then
  echo "[full] === (f) miniasm assembly ==="
  # Every tool sanitizes with --reads_fasta -> ONE global scale c=median(fasta_len/native_len),
  # a similarity transform that preserves each tool's overlap geometry and only shifts the
  # overall scale into base space. mm2 gets miniasm -f reads.fasta (real contig sequences);
  # the signal-domain tools assemble without -f (lengths from the rescaled PAF).
  for tag in neurosamble rawsamble mm2; do
    case "$tag" in
      neurosamble) PAF="$NEURO_PAF" ;;
      rawsamble)   PAF="$RAW_PAF" ;;
      mm2)         PAF="$TRUTH_PAF" ;;
    esac
    [[ -s "$PAF" ]] || { echo "[full] $tag: $PAF missing/empty; skip assembly"; continue; }
    CLEAN="$OUTDIR/${tag}.clean.paf"
    GFA="$OUTDIR/${tag}.gfa"
    "$PYTHON" -m neurosamble.overlap.sanitize_paf --in_paf "$PAF" \
      --out_paf "$CLEAN" --reads_fasta "$READS_FASTA" \
      2>&1 | tee -a "$OUTDIR/sanitize.log"
    if [[ "$tag" == "mm2" ]]; then
      "$MINIASM" -f "$READS_FASTA" "$CLEAN" > "$GFA" 2> "$OUTDIR/${tag}_miniasm.log" || true
    else
      "$MINIASM" "$CLEAN" > "$GFA" 2> "$OUTDIR/${tag}_miniasm.log" || true
      if [[ ! -s "$GFA" ]]; then
        # Fallback: some miniasm builds need -f. Build a placeholder FASTA from the
        # NATIVE clean.paf read lengths (never reads.fasta, so no rescaling).
        echo "[full] $tag: miniasm w/o -f gave empty gfa; retrying with placeholder FASTA" \
          | tee -a "$OUTDIR/${tag}_miniasm.log"
        PLACE="$OUTDIR/${tag}.placeholder.fasta"
        "$PYTHON" -c '
import sys
clean, out = sys.argv[1], sys.argv[2]
L = {}
with open(clean) as f:
    for line in f:
        c = line.rstrip("\n").split("\t")
        if len(c) < 9:
            continue
        try:
            ql, tl = int(c[1]), int(c[6])
        except ValueError:
            continue
        if ql > L.get(c[0], 0):
            L[c[0]] = ql
        if tl > L.get(c[5], 0):
            L[c[5]] = tl
with open(out, "w") as w:
    for name, n in L.items():
        w.write(">" + name + "\n" + "N" * n + "\n")
' "$CLEAN" "$PLACE" 2>&1 | tee -a "$OUTDIR/${tag}_miniasm.log" || true
        "$MINIASM" -f "$PLACE" "$CLEAN" > "$GFA" 2> "$OUTDIR/${tag}_miniasm.log" || true
      fi
    fi
    [[ -s "$GFA" ]] && GFA_OF[$tag]="$GFA"
  done
fi

# --------------------------------------------------------------------------- #
# 5) (g) Scoring: overlap P/R/F1 vs truth (+ contiguity if a GFA exists)
# --------------------------------------------------------------------------- #
echo "[full] === (g) scoring (overlap P/R/F1 + contiguity) ==="
for tag in neurosamble rawsamble; do
  case "$tag" in
    neurosamble) PAF="$NEURO_PAF" ;;
    rawsamble)   PAF="$RAW_PAF" ;;
  esac
  [[ -s "$PAF" ]] || { echo "[full] $tag: $PAF missing/empty; skip scoring"; continue; }
  echo "---- $tag ----" | tee -a "$OUTDIR/score.out"
  GFA_ARG=()
  [[ -n "${GFA_OF[$tag]:-}" ]] && GFA_ARG=(--gfa "${GFA_OF[$tag]}")
  "$PYTHON" -m neurosamble.overlap.score \
    --tool_paf "$PAF" --truth_paf "$TRUTH_PAF" "${GFA_ARG[@]}" \
    --json "$OUTDIR/${tag}_score.json" \
    2>&1 | tee -a "$OUTDIR/score.out"
done

echo ""
echo "############################ NEUROSAMBLE OVERLAP SUMMARY ############################"
cat "$OUTDIR/score.out" 2>/dev/null || true
echo "####################################################################################"
echo "[full] DONE. Outputs under: $OUTDIR"