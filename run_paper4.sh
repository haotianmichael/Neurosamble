#!/usr/bin/env bash
# =============================================================================
# RaBitQ (RQ) on E. coli R9.4 -> experiment/4.0/   (单卡 GPU0,不分块,不走 UVM 换页)
#
#   复用(只读) experiment/3.0/ecoli_r9:
#     encode/ 两卡编码的 2 个 fp16 分片 -> 按 rank 顺序合并成单分片(与 windows.npy 行序一致,不重编码)
#     rawsamble.paf / mm2_overlaps.paf -> 拷贝进 4.0(RS 基线与真值同一份,不重跑)
#   新跑: RaBitQ 流式建索引 + 全量检索(1 次) -> chaining -> 统一装配打分 -> 汇总
#   参数: bits=4, QUANT4, strategy=none(同 RQ_yeast);topk=256(RQ 快速路径上限);
#         nlist=4*sqrt(N)(同 IVF), nprobe=384(同 IVF), batch=1024
#   可重入: 合并好的 encode / 已建好的 rabitq.index / 已有打分 都会复用或跳过。
# 前置: 已按补丁重编 libcuvs(流式建库修复 + topk 上限 256)和 rabitq_search。
# 用法: bash run_paper4.sh
# =============================================================================
set -uo pipefail
cd /home/nfs/mahaotian/ESA/Neurosamble
source /home/mahaotian/miniconda3/etc/profile.d/conda.sh; conda activate py310fp16
export PYTHONPATH="$PWD/src"

RQ="src/neurosamble/overlap/run_full.sh"
ENC="data/encoder/mamba/real_encoder_v1.pt"
PORE_R9="data/ecoli/pore_r9.4_6mer.model"
RHS="${RAWHASH_SCRIPTS:-/home/nfs/mahaotian/ESA/Rawhash2/test/scripts}"
CD="/home/nfs/mahaotian/ESA/CALL_ESA/data"
MINIMAP2=minimap2; MINIASM=miniasm; PYTHON=python; THREADS=112

SRC3="experiment/3.0/ecoli_r9"                 # 只读
ROOT="experiment/4.0"; NAME="RQ_ecoli_r9"
OUT="$ROOT/$NAME"; LOG="$ROOT/$NAME.log"
FASTA="data/ecoli/reads.fasta"; BLOW5="data/ecoli/ecoli_R9.blow5"; REF="$CD/d2_ecoli_r94/ref.fa"
mkdir -p "$ROOT/accuracy" "$OUT/encode" "$OUT/index"

TOPK=256; NPROBE=384; BITS=4; BATCH=8192

# ============================================================================
#  PHASE 1 —— 检索(复用 3.0 的编码,RQ 建索引 + 检索 + chaining)
# ============================================================================
if [[ -s "$ROOT/accuracy/$NAME/neurosamble_score.json" ]]; then
  echo "===== RQ $NAME 已完成(有 accuracy),跳过检索 =====" | tee -a "$LOG"
else
  echo "===== RQ $NAME $(date '+%F %T') bits=$BITS QUANT4 topk=$TOPK nprobe=$NPROBE batch=$BATCH GPU0 =====" | tee -a "$LOG"
  for f in "$SRC3/encode/encode_manifest.json" "$SRC3/rawsamble.paf" "$SRC3/mm2_overlaps.paf"; do
    [[ -s "$f" ]] || { echo "  !! 缺 $f,无法复用 3.0 结果" | tee -a "$LOG"; exit 1; }
  done

  # (1) 合并两卡编码 -> 单分片 fp16(拷贝,不改 3.0)
  if [[ ! -s "$OUT/encode/encode_manifest.json" ]]; then
    echo "  合并 $SRC3/encode -> $OUT/encode(单分片)" | tee -a "$LOG"
    "$PYTHON" - "$SRC3/encode" "$OUT/encode" << 'PY' 2>&1 | tee -a "$LOG"
import json, os, sys, shutil
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(src, "encode_manifest.json")))
assert m.get("dtype", "float32") == "float16", m.get("dtype")
sh = sorted(m["shards"], key=lambda s: s["rank"])
D = int(m["D"]); N = 0
tmp = os.path.join(dst, "embeddings_shard0.f16.tmp")
with open(tmp, "wb") as o:
    for s in sh:
        p = os.path.join(src, s["emb_file"])
        assert os.path.getsize(p) == int(s["n_rows"]) * D * 2, (p, os.path.getsize(p))
        with open(p, "rb") as f: shutil.copyfileobj(f, o, 1 << 24)
        N += int(s["n_rows"])
os.replace(tmp, os.path.join(dst, "embeddings_shard0.f16"))
np.save(os.path.join(dst, "windows_shard0.npy"),
        np.concatenate([np.load(os.path.join(src, s["win_file"])) for s in sh], 0))
with open(os.path.join(dst, "read_ids_shard0.txt"), "w") as o:
    for s in sh:
        for line in open(os.path.join(src, s["rid_file"])):
            o.write(line if line.endswith("\n") else line + "\n")
m["shards"] = [{"rank": 0, "n_rows": N, "emb_file": "embeddings_shard0.f16",
                "win_file": "windows_shard0.npy", "rid_file": "read_ids_shard0.txt"}]
m["total_n_windows"] = N
json.dump(m, open(os.path.join(dst, "encode_manifest.json"), "w"), indent=2)
print("[merge] N=%d D=%d from %d shards" % (N, D, len(sh)))
PY
    [[ -s "$OUT/encode/encode_manifest.json" ]] || { echo "  !! 合并失败" | tee -a "$LOG"; exit 1; }
  fi

  # (2) RS 基线与真值:拷贝 3.0 的同一份(不重跑)
  for f in rawsamble.paf mm2_overlaps.paf; do
    [[ -s "$OUT/$f" ]] || cp "$SRC3/$f" "$OUT/$f"
  done

  # (3) nlist = 4*sqrt(N),与 index_ivf.py 完全相同
  NLIST=$("$PYTHON" -c "import json,math;m=json.load(open('$OUT/encode/encode_manifest.json'));print(int(min(4*math.sqrt(m['total_n_windows']),65536)))")
  echo "  nlist=$NLIST" | tee -a "$LOG"

  # (4) RQ 建索引 + 检索(单卡、单次全量)+ chaining;不重跑 encode/RS/truth
  rm -f "$OUT/neurosamble.paf"
  RQ_BITS="$BITS" RQ_NLIST="$NLIST" RQ_NPROBE="$NPROBE" TOPK="$TOPK" RQ_SEARCH_BATCH="$BATCH" RQ_GPU=0 \
  RQ_EXTRA="--search_mode QUANT4 --strategy none --reorder_scale 1.45 --stream 1 --train_per_cluster 128" \
  NUM_GPUS=1 CHAIN_WORKERS=0 THREADS="$THREADS" \
  REUSE_ENCODE=1 DO_RAWSAMBLE=0 REUSE_TRUTH_PAF=1 DO_ASSEMBLY=0 \
  LOAD_ENCODER="$ENC" \
  bash "$RQ" "$OUT" "$BLOW5" "$FASTA" "$PORE_R9" >> "$LOG" 2>&1 \
    && echo "  OK RQ $NAME" | tee -a "$LOG" || { echo "  FAILED RQ $NAME (见 $LOG)" | tee -a "$LOG"; exit 1; }
fi

# ============================================================================
#  PHASE 2 —— 统一装配 + 打分 + chained%(论文口径: NS sanitize, RS/mm2 raw, 默认 -s)
# ============================================================================
O="$ROOT/accuracy/$NAME"; mkdir -p "$O"; ALOG="$O/accuracy.log"
if [[ -s "$O/neurosamble_score.json" ]]; then
  echo "===== ACC $NAME 已完成,跳过 =====" | tee -a "$ALOG"
else
  echo "===== ACC $NAME $(date '+%F %T') =====" | tee -a "$ALOG"
  N_="$OUT/neurosamble.paf"; R_="$OUT/rawsamble.paf"; M_="$OUT/mm2_overlaps.paf"
  [[ -s "$N_" && -s "$M_" ]] || { echo "  缺 $N_ 或 $M_" | tee -a "$ALOG"; exit 1; }
  "$PYTHON" -m neurosamble.overlap.sanitize_paf --in_paf "$N_" --out_paf "$O/ns.clean.paf" --reads_fasta "$FASTA" >> "$ALOG" 2>&1
  "$MINIASM" "$O/ns.clean.paf" > "$O/neurosamble.gfa" 2>>"$ALOG" || true
  "$MINIASM" "$R_" > "$O/rawsamble.gfa" 2>>"$ALOG" || true
  "$MINIASM" "$M_" > "$O/mm2.gfa" 2>>"$ALOG" || true
  "$PYTHON" -m neurosamble.overlap.score --tool_paf "$R_" --truth_paf "$M_" --gfa "$O/rawsamble.gfa" --json "$O/rawsamble_score.json" >> "$ALOG" 2>&1
  "$PYTHON" -m neurosamble.overlap.score --tool_paf "$M_" --truth_paf "$M_" --gfa "$O/mm2.gfa" --json "$O/mm2_score.json" >> "$ALOG" 2>&1
  if [[ -s "$REF" && -s "$RHS/evaluate_gfa.py" ]]; then
    "$MINIMAP2" -X --for-only -x map-ont -t "$THREADS" -o "$O/true_mappings.paf" "$REF" "$FASTA" 2>>"$ALOG" || true
    for tag in neurosamble rawsamble mm2; do
      if [[ -s "$O/${tag}.gfa" && -s "$O/true_mappings.paf" ]]; then
        "$PYTHON" "$RHS/evaluate_gfa.py" "$O/${tag}.gfa" "$O/true_mappings.paf" > "$O/${tag}_chained.tsv" 2>>"$ALOG" || true
        awk -F'\t' '$1=="TOTAL"{print $5}' "$O/${tag}_chained.tsv" 2>/dev/null > "$O/${tag}_chained.val" || echo NA > "$O/${tag}_chained.val"
      else echo NA > "$O/${tag}_chained.val"; fi
    done
  else for tag in neurosamble rawsamble mm2; do echo NA > "$O/${tag}_chained.val"; done; fi
  # NS 的 score.json 最后写:它存在即表示本数据集打分全部完成(可重入判据)
  "$PYTHON" -m neurosamble.overlap.score --tool_paf "$N_" --truth_paf "$M_" --gfa "$O/neurosamble.gfa" --json "$O/neurosamble_score.json" >> "$ALOG" 2>&1
  echo "  OK ACC $NAME -> $O" | tee -a "$ALOG"
fi

# ============================================================================
#  PHASE 3 —— 汇总(RQ 4.0 + 同数据集 IVF 3.0 对照,只读 3.0)
# ============================================================================
echo "########## 汇总 ##########"
"$PYTHON" - "$ROOT" "$NAME" "experiment/3.0" << 'PY' | tee "$ROOT/ALL_SUMMARY.txt"
import csv, json, os, re, sys
root, name, root3 = sys.argv[1], sys.argv[2], sys.argv[3]
def L(p):
    try: return json.load(open(p))
    except Exception: return {}
def V(p):
    try:
        v = open(p).read().strip(); return round(float(v) * 100, 1) if v not in ("", "NA") else ""
    except Exception: return ""
def tim(r, d):
    t = {}; p = os.path.join(r, d, "timing_summary.csv")
    if os.path.exists(p):
        for line in open(p):
            k, _, v = line.strip().partition(",")
            if v.replace(".", "", 1).isdigit(): t[k] = v
    lp = os.path.join(r, d + ".log")
    if os.path.exists(lp):
        for line in open(lp, errors="ignore"):
            m = re.search(r"\[TIME\]\s+(encode|index|query|rawsamble)\s*=\s*([0-9.]+)", line)
            if m and float(m.group(2)) > 0: t[m.group(1)] = m.group(2)
    return t
ov = lambda o, k: o.get("overlap", {}).get(k, ""); co = lambda o, k: o.get("contiguity", {}).get(k, "")
t_rq, t_ivf = tim(root, name), tim(root3, "ecoli_r9")
rows = []
def add(label, accdir, tag, bld, ret, chn):
    s = L(os.path.join(accdir, tag + "_score.json"))
    if not s: return
    rows.append([label, tag, ov(s, "precision"), ov(s, "recall"), ov(s, "f1"),
                 V(os.path.join(accdir, tag + "_chained.val")), co(s, "n50"), co(s, "aun"),
                 co(s, "longest"), co(s, "n_unitigs"), bld, ret, chn])
A4 = os.path.join(root, "accuracy", name); A3 = os.path.join(root3, "accuracy", "ecoli_r9")
add(name, A4, "neurosamble", t_rq.get("rabitq_build", ""), t_rq.get("rabitq_search", ""), t_rq.get("chaining", ""))
add("ecoli_r9_IVF(3.0)", A3, "neurosamble", t_ivf.get("index", ""), t_ivf.get("query", ""), "(含在search)")
add("ecoli_r9", A4, "rawsamble", "", t_ivf.get("rawsamble", ""), "")
add("ecoli_r9", A4, "mm2", "", "", "")
hdr = ["experiment", "method", "P", "R", "F1", "Chained%", "N50", "auN", "Longest", "Unitigs", "build_s", "search_s", "chain_s"]
with open(os.path.join(root, "ALL_SUMMARY.csv"), "w") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
print("== %s/ALL_SUMMARY.csv ==" % root)
W = [18, 11, 7, 7, 7, 8, 10, 10, 10, 8, 8, 9, 12]
fmt = lambda r: "  ".join(str(x)[:w].ljust(w) for x, w in zip(r, W))
print(fmt(hdr))
for r in rows: print(fmt(r))
PY
echo "[paper4] DONE -> $ROOT  (汇总: $ROOT/ALL_SUMMARY.csv;日志: $LOG, $ROOT/accuracy/$NAME/accuracy.log)"
