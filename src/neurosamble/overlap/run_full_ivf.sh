#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 OUTDIR REAL_BLOW5 READS_FASTA [PORE]" >&2; exit 2; fi
OUTDIR="$1"; REAL_BLOW5="$2"; READS_FASTA="$3"; PORE="${4:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; SRC="$(cd "$HERE/../.." && pwd)"; REPO="$(cd "$SRC/.." && pwd)"
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
DEFAULT_ENCODER="${NEUROSAMBLE_OUT_DIR:-$REPO/data}/real_encoder_v1.pt"
LOAD_ENCODER="${LOAD_ENCODER:-$DEFAULT_ENCODER}"
[[ -s "$LOAD_ENCODER" ]] || { echo "[full] LOAD_ENCODER not found: $LOAD_ENCODER" >&2; exit 2; }
NUM_GPUS="${NUM_GPUS:-2}"; TOPK="${TOPK:-64}"; NPROBE="${NPROBE:-384}"
INDEX_TYPE="${INDEX_TYPE:-ivfflat}"
DO_ASSEMBLY="${DO_ASSEMBLY:-1}"; DO_RAWSAMBLE="${DO_RAWSAMBLE:-1}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 8)}"; SPK="${SAMPLES_PER_KMER:-9}"
WIN="${WIN:-2000}"; STRIDE="${STRIDE:-1000}"
MIN_NUM_ANCHORS="${MIN_NUM_ANCHORS:-5}"; MIN_CHAINING_SCORE="${MIN_CHAINING_SCORE:-40}"
MAX_GAP_BP="${MAX_GAP_BP:-2500}"; BW_BP="${BW_BP:-5000}"
CHAIN_WORKERS="${CHAIN_WORKERS:-0}"; ENCODE_BATCH="${ENCODE_BATCH:-2048}"
FAISS_GPU="${FAISS_GPU:-1}"; QUERY_BATCH="${QUERY_BATCH:-65536}"; GPU_TEMP_MB="${GPU_TEMP_MB:-8192}"
QUERY_GPU_ID="${QUERY_GPU_ID:--1}"
PYTHON="${PYTHON:-python}"; TORCHRUN="${TORCHRUN:-torchrun}"
MINIMAP2="${MINIMAP2:-minimap2}"; MINIASM="${MINIASM:-miniasm}"
MINIASM_MIN_SPAN="${MINIASM_MIN_SPAN:-500}"
RAWHASH2="${RAWHASH2:-rawhash2}"; RAWHASH_PRESET="${RAWHASH_PRESET:-}"
mkdir -p "$OUTDIR" "$OUTDIR/encode" "$OUTDIR/index"
NEURO_PAF="$OUTDIR/neurosamble.paf"; RAW_PAF="$OUTDIR/rawsamble.paf"; TRUTH_PAF="$OUTDIR/mm2_overlaps.paf"
encode_sec=0; index_sec=0; query_sec=0; raw_sec=0; truth_sec=0; asm_sec=0; score_sec=0
_now(){ date +%s; }
echo "[full] OUTDIR=$OUTDIR NUM_GPUS=$NUM_GPUS TOPK=$TOPK NPROBE=$NPROBE INDEX_TYPE=$INDEX_TYPE FAISS_GPU=$FAISS_GPU"
echo "[full] LOAD_ENCODER=$LOAD_ENCODER win=$WIN stride=$STRIDE miniasm_s=$MINIASM_MIN_SPAN preset='$RAWHASH_PRESET'"
if [[ "${REUSE_NEURO_PAF:-0}" == "1" && -s "$NEURO_PAF" ]]; then
  echo "[full] REUSE_NEURO_PAF=1 -> skip encode/index/query"
else
  if [[ "${REUSE_ENCODE:-0}" == "1" && -s "$OUTDIR/encode/encode_manifest.json" ]]; then
    echo "[full] REUSE_ENCODE=1 -> skip encode"
  else
    echo "[full] === (a) encode (${NUM_GPUS}-GPU, fp16) ==="; _t=$(_now)
    "$TORCHRUN" --nproc_per_node="$NUM_GPUS" -m neurosamble.overlap.encode \
      --real_reads "$REAL_BLOW5" --load_encoder "$LOAD_ENCODER" \
      --out_dir "$OUTDIR/encode" --win "$WIN" --stride "$STRIDE" \
      --encode_batch "$ENCODE_BATCH" --fp16_out 2>&1 | tee "$OUTDIR/encode.log"
    encode_sec=$(( $(_now) - _t )); echo "[TIME] encode = ${encode_sec}s" | tee -a "$OUTDIR/timing.log"
  fi
  if [[ "${REUSE_INDEX:-0}" == "1" && -s "$OUTDIR/index/ivf.index" ]]; then
    echo "[full] REUSE_INDEX=1 -> skip index build"
  else
    echo "[full] === (b) IVF index build ($INDEX_TYPE, CPU) ==="; _t=$(_now)
    "$PYTHON" -m neurosamble.overlap.index_ivf \
      --encode_dir "$OUTDIR/encode" --out_dir "$OUTDIR/index" \
      --index_type "$INDEX_TYPE" --threads "$THREADS" 2>&1 | tee "$OUTDIR/index.log"
    index_sec=$(( $(_now) - _t )); echo "[TIME] index = ${index_sec}s" | tee -a "$OUTDIR/timing.log"
  fi
  echo "[full] === (c) faiss IVF query + parallel chaining ==="; _t=$(_now)
  "$PYTHON" -m neurosamble.overlap.map_full \
    --index_dir "$OUTDIR/index" --encode_dir "$OUTDIR/encode" \
    --out_paf "$NEURO_PAF" --nprobe "$NPROBE" --topk "$TOPK" \
    --threads "$THREADS" --samples_per_kmer "$SPK" --faiss_gpu "$FAISS_GPU" \
    --min_num_anchors "$MIN_NUM_ANCHORS" --min_chaining_score "$MIN_CHAINING_SCORE" \
    --max_gap_bp "$MAX_GAP_BP" --bw_bp "$BW_BP" \
    --query_batch "$QUERY_BATCH" --gpu_temp_mb "$GPU_TEMP_MB" \
    --gpu_id "$QUERY_GPU_ID" --chain_workers "$CHAIN_WORKERS" 2>&1 | tee "$OUTDIR/query.log"
  query_sec=$(( $(_now) - _t )); echo "[TIME] query = ${query_sec}s" | tee -a "$OUTDIR/timing.log"
fi
if [[ "$DO_RAWSAMBLE" != "0" ]]; then
  echo "[full] === (d) Rawsamble (rawhash2 -x ava) ==="
  if [[ "${REUSE_RAW_PAF:-0}" == "1" && -s "$RAW_PAF" ]]; then echo "[full] reuse rawsamble.paf";
  elif [[ -z "$PORE" ]]; then echo "[full] no PORE -> skip rawsamble";
  else
    _t=$(_now)
    "$RAWHASH2" -x ava $RAWHASH_PRESET -t "$THREADS" -p "$PORE" -d "$OUTDIR/rawsamble_idx" "$REAL_BLOW5" 2>&1 | tee "$OUTDIR/rawsamble_index.log"
    "$RAWHASH2" -x ava $RAWHASH_PRESET -t "$THREADS" "$OUTDIR/rawsamble_idx" "$REAL_BLOW5" > "$RAW_PAF" 2> "$OUTDIR/rawsamble_map.log"
    raw_sec=$(( $(_now) - _t )); echo "[TIME] rawsamble = ${raw_sec}s" | tee -a "$OUTDIR/timing.log"
  fi
fi
echo "[full] === (e) minimap2 ava-ont truth ==="
if [[ "${REUSE_TRUTH_PAF:-0}" == "1" && -s "$TRUTH_PAF" ]]; then echo "[full] reuse mm2_overlaps.paf";
else
  _t=$(_now)
  "$MINIMAP2" -x ava-ont --for-only -t "$THREADS" "$READS_FASTA" "$READS_FASTA" > "$TRUTH_PAF" 2> "$OUTDIR/mm2_overlaps.log"
  truth_sec=$(( $(_now) - _t )); echo "[TIME] truth = ${truth_sec}s" | tee -a "$OUTDIR/timing.log"
fi
declare -A GFA_OF
if [[ "$DO_ASSEMBLY" != "0" ]]; then
  echo "[full] === (f) miniasm assembly (Rawsamble-style: raw paf, miniasm default) ==="; _t=$(_now)
  for tag in neurosamble rawsamble mm2; do
    case "$tag" in neurosamble) PAF="$NEURO_PAF";; rawsamble) PAF="$RAW_PAF";; mm2) PAF="$TRUTH_PAF";; esac
    [[ -s "$PAF" ]] || { echo "[full] $tag: $PAF missing; skip"; continue; }
    GFA="$OUTDIR/${tag}.gfa"
    "$MINIASM" "$PAF" > "$GFA" 2> "$OUTDIR/${tag}_miniasm.log" || true
    [[ -s "$GFA" ]] && GFA_OF[$tag]="$GFA"
  done
  asm_sec=$(( $(_now) - _t )); echo "[TIME] assembly = ${asm_sec}s" | tee -a "$OUTDIR/timing.log"
fi
echo "[full] === (g) scoring ==="; _t=$(_now)
for tag in neurosamble rawsamble mm2; do
  case "$tag" in neurosamble) PAF="$NEURO_PAF";; rawsamble) PAF="$RAW_PAF";; mm2) PAF="$TRUTH_PAF";; esac
  [[ -s "$PAF" ]] || { echo "[full] $tag: $PAF missing; skip scoring"; continue; }
  echo "---- $tag ----" | tee -a "$OUTDIR/score.out"
  GFA_ARG=(); [[ -n "${GFA_OF[$tag]:-}" ]] && GFA_ARG=(--gfa "${GFA_OF[$tag]}")
  "$PYTHON" -m neurosamble.overlap.score --tool_paf "$PAF" --truth_paf "$TRUTH_PAF" "${GFA_ARG[@]}" \
    --json "$OUTDIR/${tag}_score.json" 2>&1 | tee -a "$OUTDIR/score.out"
done
score_sec=$(( $(_now) - _t )); echo "[TIME] scoring = ${score_sec}s" | tee -a "$OUTDIR/timing.log"
echo "[full] === (h) chained-read % (reference-based, Rawsamble evaluate_gfa) ==="; _t=$(_now)
RH_SCRIPTS="${RAWHASH_SCRIPTS:-/home/nfs/mahaotian/ESA/Rawhash2/test/scripts}"
if [[ -z "${REF:-}" || ! -s "${REF:-}" ]]; then
  echo "[full] REF 未设置或不存在(${REF:-}); 跳过 chained%"
  for tag in neurosamble rawsamble mm2; do echo "NA" > "$OUTDIR/${tag}_chained.val"; done
else
  TRUEMAP="$OUTDIR/true_mappings.paf"
  if [[ "${REUSE_TRUEMAP:-0}" == "1" && -s "$TRUEMAP" ]]; then echo "[full] reuse true_mappings.paf";
  else "$MINIMAP2" -X --for-only -x map-ont -t "$THREADS" -o "$TRUEMAP" "$REF" "$READS_FASTA" 2> "$OUTDIR/truemap.log" || true; fi
  for tag in neurosamble rawsamble mm2; do
    if [[ -n "${GFA_OF[$tag]:-}" && -s "$TRUEMAP" ]]; then
      "$PYTHON" "$RH_SCRIPTS/evaluate_gfa.py" "${GFA_OF[$tag]}" "$TRUEMAP" > "$OUTDIR/${tag}_chained.tsv" 2>> "$OUTDIR/chained.log" || true
      CH=$(awk -F'\t' '$1=="TOTAL"{print $5}' "$OUTDIR/${tag}_chained.tsv" 2>/dev/null)
      echo "${CH:-NA}" > "$OUTDIR/${tag}_chained.val"
      echo "[full] $tag chained fraction = ${CH:-NA}"
    else
      echo "NA" > "$OUTDIR/${tag}_chained.val"; echo "[full] $tag chained: NA (无 gfa 或 truemap)"
    fi
  done
fi
chained_sec=$(( $(_now) - _t )); echo "[TIME] chained = ${chained_sec}s" | tee -a "$OUTDIR/timing.log"
{ echo "stage,seconds"; echo "encode,$encode_sec"; echo "index,$index_sec"; echo "query,$query_sec"
  echo "rawsamble,$raw_sec"; echo "truth,$truth_sec"; echo "assembly,$asm_sec"; echo "scoring,$score_sec"; } > "$OUTDIR/timing_summary.csv"
echo ""; echo "############ NEUROSAMBLE (IVF full-precision) SUMMARY ############"
cat "$OUTDIR/score.out" 2>/dev/null || true
echo "---- TIMING (s) ----"; cat "$OUTDIR/timing_summary.csv"
echo "[full] DONE. Outputs under: $OUTDIR"
