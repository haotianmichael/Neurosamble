#!/usr/bin/env bash
# =============================================================================
# 论文实验一次性重跑 -> experiment/3.0/   (完整、口径统一、可直接写论文)
#   检索: IVF(5 数据集) NS+RS+mm2  ;  RQ(yeast) NS+RS+mm2
#   装配统一按论文口径: NS 做 sanitize、RS/mm2 用 raw、miniasm 默认 -s、算 chained%。
#   每个实验单独 log;末尾汇总 ALL_SUMMARY。
#
# 可重入:数据集若已出 accuracy/<name>/neurosamble_score.json 就整体跳过;
#         encode/index 已在则复用,只补缺失阶段——不毁已成功的数据。
# 用法: bash run_paper3.sh
# =============================================================================
set -uo pipefail
cd /home/nfs/mahaotian/ESA/Neurosamble
source /home/mahaotian/miniconda3/etc/profile.d/conda.sh; conda activate py310fp16
export PYTHONPATH="$PWD/src"

IVF="src/neurosamble/overlap/run_full_ivf.sh"
RQ="src/neurosamble/overlap/run_full.sh"
ENC="data/encoder/mamba/real_encoder_v1.pt"
PORE_R9="data/ecoli/pore_r9.4_6mer.model"
# R10 的 RS 必须用官方 uncalled 模型(9mer 模型会让 rawhash2 堆崩);已拷到 data 下
UNCALLED_R10="${UNCALLED_R10:-data/ecoli_r10/pore_r10_uncalled.model}"
RAWHASH2="${RAWHASH2:-/home/nfs/mahaotian/ESA/Rawhash2/src/rawhash2}"
RHS="${RAWHASH_SCRIPTS:-/home/nfs/mahaotian/ESA/Rawhash2/test/scripts}"
CD="/home/nfs/mahaotian/ESA/CALL_ESA/data"
ROOT="experiment/3.0"; mkdir -p "$ROOT" "$ROOT/accuracy"
MINIMAP2=minimap2; MINIASM=miniasm; PYTHON=python; THREADS=112

# ============================================================================
#  PHASE 1 —— 检索(DO_ASSEMBLY=0,装配留到 Phase 2 统一做;可重入)
# ============================================================================
run_ivf(){  # name blow5 fasta pore preset topk nprobe qb gt
  local name="$1" blow5="$2" fasta="$3" pore="$4" preset="$5" topk="$6" nprobe="$7" qb="$8" gt="$9"
  local OUT="$ROOT/$name" LOG="$ROOT/$name.log"
  if [[ -s "$ROOT/accuracy/$name/neurosamble_score.json" ]]; then
    echo "===== IVF $name 已完成(有 accuracy),跳过 =====" | tee -a "$LOG"; return; fi
  echo "===== IVF $name $(date '+%F %T') topk=$topk nprobe=$nprobe qb=$qb gt=$gt =====" | tee -a "$LOG"
  if [[ ! -s "$blow5" || ! -s "$fasta" ]]; then echo "  SKIP $name: 缺 $blow5/$fasta" | tee -a "$LOG"; return; fi
  mkdir -p "$OUT/encode" "$OUT/index"
  local re_enc=0 re_idx=0
  [[ -s "$OUT/encode/encode_manifest.json" ]] && re_enc=1   # 复用已完成的 encode
  [[ -s "$OUT/index/ivf.index" ]] && re_idx=1               # 复用已建好的 index
  rm -f "$OUT/neurosamble.paf"                              # 清掉可能的半成品,重出 query
  NUM_GPUS=2 ENCODE_BATCH=2048 TOPK="$topk" NPROBE="$nprobe" INDEX_TYPE=ivfflat FAISS_GPU=1 \
  CHAIN_WORKERS=0 THREADS="$THREADS" DO_RAWSAMBLE=1 DO_ASSEMBLY=0 \
  QUERY_BATCH="$qb" GPU_TEMP_MB="$gt" \
  REUSE_ENCODE="$re_enc" REUSE_INDEX="$re_idx" REUSE_NEURO_PAF=0 REUSE_RAW_PAF=0 REUSE_TRUTH_PAF=0 \
  LOAD_ENCODER="$ENC" RAWHASH2="$RAWHASH2" RAWHASH_PRESET="$preset" \
  bash "$IVF" "$OUT" "$blow5" "$fasta" "$pore" >> "$LOG" 2>&1 \
    && echo "  OK IVF $name" | tee -a "$LOG" || echo "  FAILED IVF $name (见 $LOG)" | tee -a "$LOG"
}

run_rq_yeast(){  # 从 IVF yeast 的 encode(2-shard)合并成单 shard,不重复 encode
  local OUT="$ROOT/RQ_yeast" LOG="$ROOT/RQ_yeast.log"
  local blow5="data/yeast/yeast_R9.blow5" fasta="data/yeast/reads.fasta"
  if [[ -s "$ROOT/accuracy/RQ_yeast/neurosamble_score.json" ]]; then
    echo "===== RQ yeast 已完成,跳过 =====" | tee -a "$LOG"; return; fi
  local SRC_ENC="$ROOT/yeast/encode" DST_ENC="$OUT/encode" reuse=0 ngpu=1
  echo "===== RQ yeast $(date '+%F %T') bits4 QUANT4 nprobe512 topk64 batch8192 =====" | tee -a "$LOG"
  mkdir -p "$OUT/index"
  if [[ -s "$SRC_ENC/encode_manifest.json" ]]; then
    rm -rf "$DST_ENC"; mkdir -p "$DST_ENC"
    "$PYTHON" - "$SRC_ENC" "$DST_ENC" << 'PY'
import json,os,sys,shutil; import numpy as np
src,dst=sys.argv[1],sys.argv[2]; m=json.load(open(os.path.join(src,"encode_manifest.json")))
sh=sorted(m["shards"],key=lambda s:s["rank"])
with open(os.path.join(dst,"embeddings_shard0.f32"),"wb") as o:
    for s in sh:
        with open(os.path.join(src,s["emb_file"]),"rb") as f: shutil.copyfileobj(f,o,1<<24)
wins=[np.load(os.path.join(src,s.get("win_file") or "windows_shard%d.npy"%s["rank"])) for s in sh]
np.save(os.path.join(dst,"windows_shard0.npy"),np.concatenate(wins,0))
with open(os.path.join(dst,"read_ids_shard0.txt"),"w") as o:
    for s in sh:
        for line in open(os.path.join(src,"read_ids_shard%d.txt"%s["rank"])): o.write(line if line.endswith("\n") else line+"\n")
N=int(sum(int(s["n_rows"]) for s in sh))
m["shards"]=[{"rank":0,"n_rows":N,"emb_file":"embeddings_shard0.f32","win_file":"windows_shard0.npy","rid_file":"read_ids_shard0.txt"}]
m["total_n_windows"]=N; json.dump(m,open(os.path.join(dst,"encode_manifest.json"),"w"),indent=2)
open(os.path.join(dst,"shard0.done"),"w").write("merged N=%d\n"%N); print("[merge] N=%d"%N)
PY
    reuse=1; echo "  合并 IVF yeast encode -> 单 shard(同源 embedding)" | tee -a "$LOG"
  else echo "  IVF yeast encode 不存在,RQ 现场单-GPU encode" | tee -a "$LOG"; fi
  rm -f "$OUT/neurosamble.paf"
  RQ_BITS=4 RQ_NLIST=2048 RQ_NPROBE=512 TOPK=64 RQ_SEARCH_BATCH=8192 RQ_GPU=0 \
  RQ_EXTRA="--search_mode QUANT4 --strategy none --reorder_scale 1.45" \
  NUM_GPUS="$ngpu" ENCODE_BATCH=2048 CHAIN_WORKERS=0 THREADS="$THREADS" \
  DO_RAWSAMBLE=1 DO_ASSEMBLY=0 REUSE_ENCODE="$reuse" \
  LOAD_ENCODER="$ENC" RAWHASH2="$RAWHASH2" RAWHASH_PRESET="" \
  bash "$RQ" "$OUT" "$blow5" "$fasta" "$PORE_R9" >> "$LOG" 2>&1 \
    && echo "  OK RQ yeast" | tee -a "$LOG" || echo "  FAILED RQ yeast (见 $LOG)" | tee -a "$LOG"
}

echo "########## PHASE 1: 检索 ##########"
#        name         blow5                        fasta                        pore            preset  topk nprobe  qb    gt
run_ivf green_algae data/green_algae/reads.blow5  data/green_algae/reads.fasta "$PORE_R9"       ""      16   384    8192  8192
run_ivf yeast       data/yeast/yeast_R9.blow5     data/yeast/reads.fasta       "$PORE_R9"       ""      64   384    8192  8192
run_ivf human       data/human/reads.blow5        data/human/reads.fasta       "$PORE_R9"       ""      64   384    8192  8192
run_ivf ecoli_r10   data/ecoli_r10/reads.blow5    data/ecoli_r10/reads.fasta   "$UNCALLED_R10"  "--r10" 64   384    1024  4096
run_ivf ecoli_r9    data/ecoli/ecoli_R9.blow5     data/ecoli/reads.fasta       "$PORE_R9"       ""      600  384    1024  4096
run_rq_yeast

# ============================================================================
#  PHASE 2 —— 统一装配 + 打分 + chained% (论文口径: NS sanitize, RS/mm2 raw, 默认 -s)
# ============================================================================
acc(){  # name src_dir fasta ref
  local name="$1" src="$2" fasta="$3" ref="$4"
  local O="$ROOT/accuracy/$name"; mkdir -p "$O"; local LOG="$O/accuracy.log"
  local N="$src/neurosamble.paf" R="$src/rawsamble.paf" M="$src/mm2_overlaps.paf"
  echo "===== ACC $name $(date '+%F %T') =====" | tee -a "$LOG"
  [[ -s "$M" ]] || { echo "  缺 truth $M,跳过" | tee -a "$LOG"; return; }
  if [[ -s "$N" ]]; then
    "$PYTHON" -m neurosamble.overlap.sanitize_paf --in_paf "$N" --out_paf "$O/ns.clean.paf" --reads_fasta "$fasta" >> "$LOG" 2>&1
    "$MINIASM" "$O/ns.clean.paf" > "$O/neurosamble.gfa" 2>>"$LOG" || true
    "$PYTHON" -m neurosamble.overlap.score --tool_paf "$N" --truth_paf "$M" --gfa "$O/neurosamble.gfa" --json "$O/neurosamble_score.json" >> "$LOG" 2>&1
  fi
  if [[ -s "$R" ]]; then
    "$MINIASM" "$R" > "$O/rawsamble.gfa" 2>>"$LOG" || true
    "$PYTHON" -m neurosamble.overlap.score --tool_paf "$R" --truth_paf "$M" --gfa "$O/rawsamble.gfa" --json "$O/rawsamble_score.json" >> "$LOG" 2>&1
  fi
  "$MINIASM" "$M" > "$O/mm2.gfa" 2>>"$LOG" || true
  "$PYTHON" -m neurosamble.overlap.score --tool_paf "$M" --truth_paf "$M" --gfa "$O/mm2.gfa" --json "$O/mm2_score.json" >> "$LOG" 2>&1
  if [[ -s "$ref" && -s "$RHS/evaluate_gfa.py" ]]; then
    "$MINIMAP2" -X --for-only -x map-ont -t "$THREADS" -o "$O/true_mappings.paf" "$ref" "$fasta" 2>>"$LOG" || true
    for tag in neurosamble rawsamble mm2; do
      if [[ -s "$O/${tag}.gfa" && -s "$O/true_mappings.paf" ]]; then
        "$PYTHON" "$RHS/evaluate_gfa.py" "$O/${tag}.gfa" "$O/true_mappings.paf" > "$O/${tag}_chained.tsv" 2>>"$LOG" || true
        awk -F'\t' '$1=="TOTAL"{print $5}' "$O/${tag}_chained.tsv" 2>/dev/null > "$O/${tag}_chained.val" || echo NA > "$O/${tag}_chained.val"
      else echo NA > "$O/${tag}_chained.val"; fi
    done
  else for tag in neurosamble rawsamble mm2; do echo NA > "$O/${tag}_chained.val"; done; fi
  echo "  OK ACC $name -> $O" | tee -a "$LOG"
}
echo "########## PHASE 2: 统一装配打分 ##########"
acc green_algae "$ROOT/green_algae" data/green_algae/reads.fasta "$CD/d3_green_algae_r94/ref.fa"
acc yeast       "$ROOT/yeast"       data/yeast/reads.fasta       "$CD/d3_yeast_r94/ref.fa"
acc human       "$ROOT/human"       data/human/reads.fasta       "$CD/d4_human_na12878_r94/ref.fa"
acc ecoli_r10   "$ROOT/ecoli_r10"   data/ecoli_r10/reads.fasta   "$CD/d6_ecoli_r104/ref.fa"
acc ecoli_r9    "$ROOT/ecoli_r9"    data/ecoli/reads.fasta       "$CD/d2_ecoli_r94/ref.fa"
acc RQ_yeast    "$ROOT/RQ_yeast"    data/yeast/reads.fasta       "$CD/d3_yeast_r94/ref.fa"

# ============================================================================
#  PHASE 3 —— 汇总 CSV
# ============================================================================
echo "########## PHASE 3: 汇总 ##########"
"$PYTHON" - "$ROOT" << 'PY' | tee "$ROOT/ALL_SUMMARY.txt"
import json,os,csv,sys,re
root=sys.argv[1]; accd=os.path.join(root,"accuracy")
def L(p):
    try: return json.load(open(p))
    except: return {}
def V(p):
    try:
        v=open(p).read().strip(); return round(float(v)*100,1) if v not in("","NA") else ""
    except: return ""
def tim(name):
    t={}; p=os.path.join(root,name,"timing_summary.csv")
    if os.path.exists(p):
        for line in open(p):
            k,_,v=line.strip().partition(",")
            if v.replace(".","",1).isdigit(): t[k]=v
    lp=os.path.join(root,name+".log")   # 用日志 [TIME] 的非零值回补(修复分次重跑把 timing_summary 清零)
    if os.path.exists(lp):
        for line in open(lp,errors="ignore"):
            m=re.search(r"\[TIME\]\s+(encode|index|query|rawsamble)\s*=\s*([0-9.]+)",line)
            if m and float(m.group(2))>0: t[m.group(1)]=m.group(2)
    return t
ov=lambda o,k:o.get("overlap",{}).get(k,""); co=lambda o,k:o.get("contiguity",{}).get(k,"")
rows=[]
for retdir in ("green_algae","yeast","human","ecoli_r10","ecoli_r9","RQ_yeast"):
    A=os.path.join(accd,retdir); t=tim(retdir)
    for tag in ("neurosamble","rawsamble","mm2"):
        s=L(os.path.join(A,tag+"_score.json"))
        if not s: continue
        if tag=="neurosamble":
            ret=(t.get("query") or t.get("rabitq_search") or ""); bld=(t.get("index") or t.get("rabitq_build") or "")
        elif tag=="rawsamble": ret=t.get("rawsamble",""); bld=""
        else: ret=""; bld=""
        rows.append([retdir,tag,ov(s,"precision"),ov(s,"recall"),ov(s,"f1"),
                     V(os.path.join(A,tag+"_chained.val")),co(s,"n50"),co(s,"aun"),
                     co(s,"longest"),co(s,"n_unitigs"),bld,ret])
hdr=["experiment","method","P","R","F1","Chained%","N50","auN","Longest","Unitigs","build_s","search_s"]
with open(os.path.join(root,"ALL_SUMMARY.csv"),"w") as f:
    w=csv.writer(f); w.writerow(hdr); w.writerows(rows)
print("== %s/ALL_SUMMARY.csv =="%root)
W=[12,11,7,7,7,8,10,10,10,8,8,9]
def fmt(r): return "  ".join(str(x)[:w].ljust(w) for x,w in zip(r,W))
print(fmt(hdr))
for r in rows: print(fmt(r))
PY
echo "[paper3] DONE -> $ROOT  (汇总: $ROOT/ALL_SUMMARY.csv;各实验 log: $ROOT/<name>.log)"