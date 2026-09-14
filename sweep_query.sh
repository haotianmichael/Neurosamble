#!/usr/bin/env bash
set -euo pipefail
cd /home/nfs/mahaotian/ESA/Neurosamble
source /home/mahaotian/miniconda3/etc/profile.d/conda.sh; conda activate py310fp16
export PYTHONPATH="$PWD/src"
python -c "import sys,triton; print('[env]',sys.executable,'| triton',triton.__version__)"

# ---- paths ----
BLOW5=/home/nfs/mahaotian/ESA/CALL_ESA/data/d2_ecoli_r94/ecoli_R9.blow5
ENC=/home/nfs/mahaotian/ESA/CALL_ESA/experiments/stage4b_finetune/real_encoder_v1.pt
PORE=/home/nfs/mahaotian/ESA/Neurosamble/data/pore_r9.4_6mer.model
SHARED=experiment/full_mamba_fp16_shared          # 单卡 batch5120 的 encode+index 存这里
OLD=experiment/full_mamba_fp16_topk32             # 复用它已有的 truth/rawsamble,省得重跑
TOPKS="32 64 96 128"

# ================= 第0步:单卡 batch5120 建 encode + index(只做一次) =================
# 只想要交叉点、不想再 encode 的话:把下面 if 到 fi 整段删掉,并把 SHARED 改成 OLD 即可。
if [[ ! -s "$SHARED/index/ivf.index" ]]; then
  mkdir -p "$SHARED"
  echo "==================== (0a) encode  single-GPU batch=5120 fp16 ===================="
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m neurosamble.overlap.encode \
    --real_reads "$BLOW5" --load_encoder "$ENC" \
    --out_dir "$SHARED/encode" --win 2000 --stride 1000 \
    --encode_batch 5120 --fp16_out 2>&1 | tee "$SHARED/encode.log"
  echo "==================== (0b) IVF index build (ivfsq, CPU) ===================="
  python -m neurosamble.overlap.index_ivf \
    --encode_dir "$SHARED/encode" --out_dir "$SHARED/index" \
    --index_type ivfsq --threads 112 2>&1 | tee "$SHARED/index.log"
fi

# ---- truth + rawsamble 与 topk 无关:复用旧目录的,拷进 SHARED ----
cp -n "$OLD/mm2_overlaps.paf" "$SHARED/" 2>/dev/null || true
cp -n "$OLD/rawsamble.paf"    "$SHARED/" 2>/dev/null || true
[[ -s "$SHARED/mm2_overlaps.paf" ]] || { echo "‼ 缺 truth PAF($SHARED/mm2_overlaps.paf)"; exit 2; }

# ================= 每个 topk 只重跑 query + score =================
for K in $TOPKS; do
  D=experiment/full_mamba_fp16_topk$K; mkdir -p "$D"
  echo "==================== topk=$K -> $D ===================="
  python -m neurosamble.overlap.map_full \
    --index_dir "$SHARED/index" --encode_dir "$SHARED/encode" \
    --out_paf "$D/neurosamble.paf" --nprobe 384 --topk "$K" \
    --samples_per_kmer 9 --min_num_anchors 5 --min_chaining_score 40 \
    --max_gap_bp 2500 --bw_bp 5000 \
    --faiss_gpu 1 --gpu_id -1 --chain_workers 0 \
    --query_batch 16384 --gpu_temp_mb 12288 \
    2>&1 | tee "$D/query.log"
  cp "$SHARED/index/query_stats.json" "$D/query_stats.json" 2>/dev/null || true
  python -m neurosamble.overlap.score \
    --tool_paf "$D/neurosamble.paf" --truth_paf "$SHARED/mm2_overlaps.paf" \
    --json "$D/neurosamble_score.json" 2>&1 | tee "$D/score.log"
  echo "---- topk=$K done: $(python -c "import json;d=json.load(open('$D/neurosamble_score.json')).get('overlap',{});print('P=%.4f R=%.4f F1=%.4f'%(d.get('precision',0),d.get('recall',0),d.get('f1',0)))" 2>/dev/null) ----"
done
echo "ALL TOPK DONE -> experiment/full_mamba_fp16_topk{32,64,96,128}"
