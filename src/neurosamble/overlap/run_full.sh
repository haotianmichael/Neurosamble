#!/usr/bin/env bash
# =============================================================================
# Neurosamble -- FULL-SCALE all-vs-all overlap -> assembly (RaBitQ retrieval).
# Same pipeline & TIMING/summary as the faiss version; retrieval = GPU IVF-RaBitQ.
# RUN FROM py310fp16 (torch/mamba/minimap2/miniasm/rawhash2); the RaBitQ build+search
# step runs via `conda run -n cuvsbuild`. ALL logs live under OUTDIR (never /tmp).
# args: 1 OUTDIR  2 REAL_BLOW5  3 READS_FASTA  4 PORE(optional)
# env:  LOAD_ENCODER, RQ_BIN, RQ_NLIST(2048) RQ_NPROBE(512) RQ_BITS(4) RQ_SEARCH_BATCH(2048)
# =============================================================================
set -euo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 OUTDIR REAL_BLOW5 READS_FASTA [PORE]" >&2; exit 2; fi
OUTDIR="$1"; REAL_BLOW5="$2"; READS_FASTA="$3"; PORE="${4:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; SRC="$(cd "$HERE/../.." && pwd)"; REPO="$(cd "$SRC/.." && pwd)"
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"

DEFAULT_ENCODER="${NEUROSAMBLE_OUT_DIR:-$REPO/data}/real_encoder_v1.pt"
LOAD_ENCODER="${LOAD_ENCODER:-$DEFAULT_ENCODER}"
[[ -s "$LOAD_ENCODER" ]] || { echo "[full] LOAD_ENCODER not found: $LOAD_ENCODER" >&2; exit 2; }

NUM_GPUS="${NUM_GPUS:-1}"; TOPK="${TOPK:-64}"
DO_ASSEMBLY="${DO_ASSEMBLY:-1}"; DO_RAWSAMBLE="${DO_RAWSAMBLE:-1}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 8)}"; SPK="${SAMPLES_PER_KMER:-9}"
WIN="${WIN:-2000}"; STRIDE="${STRIDE:-1000}"
MIN_NUM_ANCHORS="${MIN_NUM_ANCHORS:-5}"; MIN_CHAINING_SCORE="${MIN_CHAINING_SCORE:-40}"
MAX_GAP_BP="${MAX_GAP_BP:-2500}"; BW_BP="${BW_BP:-5000}"
CHAIN_WORKERS="${CHAIN_WORKERS:-0}"; ENCODE_BATCH="${ENCODE_BATCH:-2048}"
RQ_BIN="${RQ_BIN:-$REPO/src/rabitq_gpu/build/rabitq_search}"
RQ_NLIST="${RQ_NLIST:-2048}"; RQ_NPROBE="${RQ_NPROBE:-512}"; RQ_BITS="${RQ_BITS:-4}"
RQ_SEARCH_BATCH="${RQ_SEARCH_BATCH:-2048}"; RQ_GPU="${RQ_GPU:-0}"
CUVS_ENV="${CUVS_ENV:-cuvsbuild}"; CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
CUVS_LIB="${CUVS_LIB:-$CONDA_BASE/envs/$CUVS_ENV/lib}"
PYTHON="${PYTHON:-python}"; TORCHRUN="${TORCHRUN:-torchrun}"
MINIMAP2="${MINIMAP2:-minimap2}"; MINIASM="${MINIASM:-miniasm}"
MINIASM_MIN_SPAN="${MINIASM_MIN_SPAN:-500}"  # miniasm -s: RS overlaps ~530bp median; default 2000 drops them all
RAWHASH2="${RAWHASH2:-rawhash2}"; RAWHASH_PRESET="${RAWHASH_PRESET:-}"

mkdir -p "$OUTDIR" "$OUTDIR/encode" "$OUTDIR/index"
NEURO_PAF="$OUTDIR/neurosamble.paf"; RAW_PAF="$OUTDIR/rawsamble.paf"; TRUTH_PAF="$OUTDIR/mm2_overlaps.paf"

# --- timing accumulators ---
encode_sec=0; rabitq_build_sec=0; rabitq_search_sec=0; chain_sec=0
raw_sec=0; truth_sec=0; asm_sec=0; score_sec=0
_now(){ date +%s; }

echo "[full] OUTDIR=$OUTDIR NUM_GPUS=$NUM_GPUS TOPK=$TOPK RaBitQ(nlist=$RQ_NLIST nprobe=$RQ_NPROBE bits=$RQ_BITS batch=$RQ_SEARCH_BATCH)"
echo "[full] LOAD_ENCODER=$LOAD_ENCODER win=$WIN stride=$STRIDE RQ_BIN=$RQ_BIN CUVS_ENV=$CUVS_ENV"
echo "[full] chaining: mna=$MIN_NUM_ANCHORS mcs=$MIN_CHAINING_SCORE gap=$MAX_GAP_BP bw=$BW_BP workers=$CHAIN_WORKERS"

if [[ "${REUSE_NEURO_PAF:-0}" == "1" && -s "$NEURO_PAF" ]]; then
  echo "[full] REUSE_NEURO_PAF=1 -> skip encode/rabitq/chain"
else
  # (a) encode
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

  read RQ_N RQ_D EMB_REL < <("$PYTHON" - "$OUTDIR/encode/encode_manifest.json" << 'PY'
import json,sys
m=json.load(open(sys.argv[1])); sh=m["shards"]
assert len(sh)==1, f"expected 1 shard, got {len(sh)}"
print(sum(s["n_rows"] for s in sh), m["D"], sh[0]["emb_file"])
PY
)
  EMB_FILE="${EMB_FILE:-$OUTDIR/encode/$EMB_REL}"
  echo "[full] N=$RQ_N D=$RQ_D emb=$EMB_FILE"
  # --- ensure index/windows.npy + index/read_ids.txt exist (RaBitQ path skips index_ivf) ---
  if [[ ! -s "$OUTDIR/index/windows.npy" || ! -s "$OUTDIR/index/read_ids.txt" ]]; then
    echo "[full] building index/windows.npy + read_ids.txt from encode shards"
    "$PYTHON" "$HERE/_merge_index.py" "$OUTDIR/encode" "$OUTDIR/index" 2>&1 | tee -a "$OUTDIR/merge.log"
  fi

  RQ_NBR="$OUTDIR/index/neighbors.i64"; RQ_DIST="$OUTDIR/index/dists.f32"; RQ_IDX="$OUTDIR/index/rabitq.index"

  # (b) RaBitQ build + search (cuvsbuild). rabitq prints its own [TIME] build/search; we also wall-time it.
  if [[ "${REUSE_RQ:-0}" == "1" && -s "$RQ_NBR" ]]; then
    echo "[full] REUSE_RQ=1 -> skip rabitq_search"
  else
    echo "[full] === (b) RaBitQ build + search (env=$CUVS_ENV) ==="; _t=$(_now)
    LD_LIBRARY_PATH="$CUVS_LIB:${LD_LIBRARY_PATH:-}" CUDA_VISIBLE_DEVICES="$RQ_GPU" \
      conda run -n "$CUVS_ENV" --no-capture-output \
      "$RQ_BIN" --emb "$EMB_FILE" --emb_dtype fp16 --n "$RQ_N" --d "$RQ_D" \
        --nlist "$RQ_NLIST" --nprobe "$RQ_NPROBE" --topk "$TOPK" --bits "$RQ_BITS" \
        --batch "$RQ_SEARCH_BATCH" --index "$RQ_IDX" ${RQ_EXTRA:-} \
        --out_nbr "$RQ_NBR" --out_dist "$RQ_DIST" 2>&1 | tee "$OUTDIR/rabitq.log"
    rabitq_total_sec=$(( $(_now) - _t )); echo "[TIME] rabitq(build+search) = ${rabitq_total_sec}s" | tee -a "$OUTDIR/timing.log"
    # split from rabitq.log [TIME] lines if present
    rabitq_build_sec=$(grep -oE 'build\+reorg = [0-9.]+' "$OUTDIR/rabitq.log" | grep -oE '[0-9.]+' | tail -1 || echo 0)
    rabitq_search_sec=$(grep -oE 'search\+write = [0-9.]+' "$OUTDIR/rabitq.log" | grep -oE '[0-9.]+' | tail -1 || echo 0)
    echo "[TIME] rabitq build=${rabitq_build_sec}s search=${rabitq_search_sec}s" | tee -a "$OUTDIR/timing.log"
  fi

  # (c) chaining (parallel, from neighbors)
  echo "[full] === (c) chaining (parallel, from RaBitQ neighbors) ==="; _t=$(_now)
  "$PYTHON" -m neurosamble.overlap.map_full \
    --index_dir "$OUTDIR/index" --encode_dir "$OUTDIR/encode" \
    --out_paf "$NEURO_PAF" --topk "$TOPK" \
    --neighbors_bin "$RQ_NBR" --dists_bin "$RQ_DIST" --samples_per_kmer "$SPK" \
    --min_num_anchors "$MIN_NUM_ANCHORS" --min_chaining_score "$MIN_CHAINING_SCORE" \
    --max_gap_bp "$MAX_GAP_BP" --bw_bp "$BW_BP" --chain_workers "$CHAIN_WORKERS" \
    2>&1 | tee "$OUTDIR/query.log"
  chain_sec=$(( $(_now) - _t )); echo "[TIME] chaining = ${chain_sec}s" | tee -a "$OUTDIR/timing.log"
fi

# (d) rawsamble
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

# (e) truth
echo "[full] === (e) minimap2 ava-ont truth ==="
if [[ "${REUSE_TRUTH_PAF:-0}" == "1" && -s "$TRUTH_PAF" ]]; then echo "[full] reuse mm2_overlaps.paf";
else
  _t=$(_now)
  "$MINIMAP2" -x ava-ont --for-only -t "$THREADS" "$READS_FASTA" "$READS_FASTA" > "$TRUTH_PAF" 2> "$OUTDIR/mm2_overlaps.log"
  truth_sec=$(( $(_now) - _t )); echo "[TIME] truth = ${truth_sec}s" | tee -a "$OUTDIR/timing.log"
fi

# (f) assembly
declare -A GFA_OF
if [[ "$DO_ASSEMBLY" != "0" ]]; then
  echo "[full] === (f) miniasm assembly ==="; _t=$(_now)
  for tag in neurosamble rawsamble mm2; do
    case "$tag" in neurosamble) PAF="$NEURO_PAF";; rawsamble) PAF="$RAW_PAF";; mm2) PAF="$TRUTH_PAF";; esac
    [[ -s "$PAF" ]] || { echo "[full] $tag: $PAF missing; skip"; continue; }
    CLEAN="$OUTDIR/${tag}.clean.paf"; GFA="$OUTDIR/${tag}.gfa"
    "$PYTHON" -m neurosamble.overlap.sanitize_paf --in_paf "$PAF" --out_paf "$CLEAN" --reads_fasta "$READS_FASTA" 2>&1 | tee -a "$OUTDIR/sanitize.log"
    if [[ "$tag" == "mm2" ]]; then
      "$MINIASM" -s "$MINIASM_MIN_SPAN" -f "$READS_FASTA" "$CLEAN" > "$GFA" 2> "$OUTDIR/${tag}_miniasm.log" || true
    else
      "$MINIASM" -s "$MINIASM_MIN_SPAN" "$CLEAN" > "$GFA" 2> "$OUTDIR/${tag}_miniasm.log" || true
      if [[ ! -s "$GFA" ]]; then
        PLACE="$OUTDIR/${tag}.placeholder.fasta"
        "$PYTHON" -c '
import sys
clean,out=sys.argv[1],sys.argv[2]; L={}
for line in open(clean):
    c=line.rstrip("\n").split("\t")
    if len(c)<9: continue
    try: ql,tl=int(c[1]),int(c[6])
    except ValueError: continue
    if ql>L.get(c[0],0): L[c[0]]=ql
    if tl>L.get(c[5],0): L[c[5]]=tl
open(out,"w").write("".join(">%s\n%s\n"%(k,"N"*n) for k,n in L.items()))
' "$CLEAN" "$PLACE" 2>&1 | tee -a "$OUTDIR/${tag}_miniasm.log" || true
        "$MINIASM" -s "$MINIASM_MIN_SPAN" -f "$PLACE" "$CLEAN" > "$GFA" 2> "$OUTDIR/${tag}_miniasm.log" || true
      fi
    fi
    [[ -s "$GFA" ]] && GFA_OF[$tag]="$GFA"
  done
  asm_sec=$(( $(_now) - _t )); echo "[TIME] assembly = ${asm_sec}s" | tee -a "$OUTDIR/timing.log"
fi

# (g) scoring
echo "[full] === (g) scoring ==="; _t=$(_now)
for tag in neurosamble rawsamble; do
  case "$tag" in neurosamble) PAF="$NEURO_PAF";; rawsamble) PAF="$RAW_PAF";; esac
  [[ -s "$PAF" ]] || { echo "[full] $tag: $PAF missing; skip scoring"; continue; }
  echo "---- $tag ----" | tee -a "$OUTDIR/score.out"
  GFA_ARG=(); [[ -n "${GFA_OF[$tag]:-}" ]] && GFA_ARG=(--gfa "${GFA_OF[$tag]}")
  "$PYTHON" -m neurosamble.overlap.score --tool_paf "$PAF" --truth_paf "$TRUTH_PAF" "${GFA_ARG[@]}" \
    --json "$OUTDIR/${tag}_score.json" 2>&1 | tee -a "$OUTDIR/score.out"
done
score_sec=$(( $(_now) - _t )); echo "[TIME] scoring = ${score_sec}s" | tee -a "$OUTDIR/timing.log"

# --- summary csv ---
SUMCSV="$OUTDIR/timing_summary.csv"
{
  echo "stage,seconds"
  echo "encode,$encode_sec"
  echo "rabitq_build,$rabitq_build_sec"
  echo "rabitq_search,$rabitq_search_sec"
  echo "chaining,$chain_sec"
  echo "rawsamble,$raw_sec"
  echo "truth,$truth_sec"
  echo "assembly,$asm_sec"
  echo "scoring,$score_sec"
} > "$SUMCSV"

echo ""
echo "############################ NEUROSAMBLE (RaBitQ) SUMMARY ############################"
cat "$OUTDIR/score.out" 2>/dev/null || true
echo "---- TIMING (s) ----"; cat "$SUMCSV"
echo "#####################################################################################"
echo "[full] DONE. Outputs under: $OUTDIR"