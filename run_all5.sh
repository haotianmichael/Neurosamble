#!/usr/bin/env bash
# 一口气 IVF 全精度跑 5 个数据集 -> experiment/2.0/<name>/, 末尾汇总 ALL_SUMMARY.csv
# 装配=Rawsamble 原样(原始 paf + miniasm 默认); 打分含 NS/RS/mm2 的 overlap P/R/F1 + contiguity + Chained read%
# 可重入(REUSE_*); 日志追加不覆盖; 各阶段耗时从 .run.log 的 [TIME] 抓取
set -uo pipefail
cd /home/nfs/mahaotian/ESA/Neurosamble
source /home/mahaotian/miniconda3/etc/profile.d/conda.sh; conda activate py310fp16
export PYTHONPATH="$PWD/src"

IVF="src/neurosamble/overlap/run_full_ivf.sh"
ENC="data/encoder/mamba/real_encoder_v1.pt"
PORE_R9="data/ecoli/pore_r9.4_6mer.model"                       # r9.4 通用(6-mer)
PORE_R10="${PORE_R10:-data/ecoli_r10/pore_r10_9mer.model}"      # r10 (9-mer)
RAWHASH2="${RAWHASH2:-/home/nfs/mahaotian/ESA/Rawhash2/src/rawhash2}"
QBATCH="${QUERY_BATCH:-8192}"                                    # 修 36GB OOM 的关键旋钮
GTEMP="${GPU_TEMP_MB:-8192}"
CD="/home/nfs/mahaotian/ESA/CALL_ESA/data"                       # 参考基因组来源(算 Chained%)
ROOT="experiment/2.0"; mkdir -p "$ROOT"

run_one(){  # name blow5 fasta pore preset topk nprobe ref
  local name="$1" blow5="$2" fasta="$3" pore="$4" preset="$5" topk="$6" nprobe="$7" ref="$8"
  echo "########## $(date '+%F %T')  $name  topk=$topk nprobe=$nprobe preset='$preset' qbatch=$QBATCH ##########"
  if [[ ! -s "$blow5" || ! -s "$fasta" ]]; then echo "  SKIP $name: 缺 blow5/fasta ($blow5 / $fasta)"; return; fi
  echo "===== RERUN $(date '+%F %T')  topk=$topk nprobe=$nprobe qbatch=$QBATCH =====" >> "$ROOT/$name.run.log"
  NUM_GPUS=2 ENCODE_BATCH=2048 TOPK="$topk" NPROBE="$nprobe" INDEX_TYPE=ivfflat FAISS_GPU=1 \
  CHAIN_WORKERS=0 THREADS=112 DO_RAWSAMBLE=1 DO_ASSEMBLY=1 \
  QUERY_BATCH="$QBATCH" GPU_TEMP_MB="$GTEMP" REF="$ref" \
  REUSE_ENCODE=1 REUSE_INDEX=1 REUSE_NEURO_PAF=1 REUSE_RAW_PAF=1 REUSE_TRUTH_PAF=1 REUSE_TRUEMAP=1 \
  LOAD_ENCODER="$ENC" RAWHASH2="$RAWHASH2" RAWHASH_PRESET="$preset" \
  bash "$IVF" "$ROOT/$name" "$blow5" "$fasta" "$pore" >> "$ROOT/$name.run.log" 2>&1 \
    && echo "  OK -> $ROOT/$name" || echo "  FAILED (见 $ROOT/$name.run.log 末尾)"
}

# ===== 配置表(小->大,按 coverage 调 topk;nprobe 统一 384) =====
#          name         blow5                          fasta                        pore        preset   topk nprobe  ref
run_one green_algae  data/green_algae/reads.blow5    data/green_algae/reads.fasta  "$PORE_R9"  ""       16   384    "$CD/d3_green_algae_r94/ref.fa"
run_one yeast        data/yeast/yeast_R9.blow5       data/yeast/reads.fasta        "$PORE_R9"  ""       64   384    "$CD/d3_yeast_r94/ref.fa"
run_one human        data/human/reads.blow5          data/human/reads.fasta        "$PORE_R9"  ""       64   384    "$CD/d4_human_na12878_r94/ref.fa"
run_one ecoli_r10    data/ecoli_r10/reads.blow5      data/ecoli_r10/reads.fasta    "$PORE_R10" "--r10"  64   384    "$CD/d6_ecoli_r104/ref.fa"
run_one ecoli_r9     data/ecoli/ecoli_R9.blow5       data/ecoli/reads.fasta        "$PORE_R9"  ""       600  384    "$CD/d2_ecoli_r94/ref.fa"

# ===== 汇总 (NS/RS/mm2: overlap P/R/F1 + contiguity + Chained%; 耗时从 .run.log 抓,最后一次为准) =====
python3 - << 'PY'
import json,glob,os,csv,re
root="experiment/2.0"; rows=[]
def L(p):
    try: return json.load(open(p))
    except Exception: return {}
def times(name):
    t={}; p=root+"/"+name+".run.log"
    if os.path.exists(p):
        for line in open(p,errors="ignore"):
            m=re.search(r'\[TIME\]\s+(encode|index|query)\s*=\s*(\d+)',line)
            if m: t[m.group(1)]=m.group(2)   # last wins
    return t
def chained(name,tag):
    p=root+"/"+name+"/"+tag+"_chained.val"
    try:
        v=open(p).read().strip()
        return round(float(v)*100,1) if v not in ("","NA") else ""
    except Exception: return ""
ov=lambda o,k:o.get("overlap",{}).get(k,"")
co=lambda o,k:o.get("contiguity",{}).get(k,"")
for d in sorted(glob.glob(root+"/*/")):
    name=os.path.basename(d.rstrip("/"))
    ns=L(d+"neurosamble_score.json"); rs=L(d+"rawsamble_score.json"); mm=L(d+"mm2_score.json"); tim=times(name)
    rows.append([name,
        ov(ns,"precision"),ov(ns,"recall"),ov(ns,"f1"),co(ns,"n50"),co(ns,"aun"),co(ns,"longest"),co(ns,"n_unitigs"),chained(name,"neurosamble"),
        ov(rs,"precision"),ov(rs,"recall"),ov(rs,"f1"),co(rs,"n50"),co(rs,"aun"),co(rs,"longest"),co(rs,"n_unitigs"),chained(name,"rawsamble"),
        co(mm,"n50"),co(mm,"aun"),co(mm,"longest"),co(mm,"n_unitigs"),chained(name,"mm2"),
        tim.get("encode",""),tim.get("index",""),tim.get("query","")])
hdr=["dataset",
     "NS_P","NS_R","NS_F1","NS_N50","NS_auN","NS_longest","NS_utg","NS_chain%",
     "RS_P","RS_R","RS_F1","RS_N50","RS_auN","RS_longest","RS_utg","RS_chain%",
     "MM2_N50","MM2_auN","MM2_longest","MM2_utg","MM2_chain%",
     "enc_s","idx_s","qry_s"]
with open(root+"/ALL_SUMMARY.csv","w") as f:
    w=csv.writer(f); w.writerow(hdr); w.writerows(rows)
print("== experiment/2.0/ALL_SUMMARY.csv ==")
for r in [hdr]+rows: print("\t".join(str(x) for x in r))
PY
echo "ALL DONE."
