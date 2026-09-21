#!/usr/bin/env bash
# 复用已有 PAF 重新装配(不跑 encode/检索/rawhash2)。修正口径: NS 走 sanitize 校坐标, RS/mm2 用 raw.
# 全部结果 -> experiment/2.0/accuracy/
set -uo pipefail
cd /home/nfs/mahaotian/ESA/Neurosamble
source /home/mahaotian/miniconda3/etc/profile.d/conda.sh; conda activate py310fp16
export PYTHONPATH="$PWD/src"

SRC="experiment/2.0"                          # 已有 paf 来源(只读)
OUT="experiment/2.0/accuracy"; mkdir -p "$OUT"
MINIASM=miniasm; PYTHON=python
RHS=/home/nfs/mahaotian/ESA/Rawhash2/test/scripts   # evaluate_gfa.py(算 Chained%)

declare -A FASTA
FASTA[green_algae]=data/green_algae/reads.fasta
FASTA[yeast]=data/yeast/reads.fasta
FASTA[human]=data/human/reads.fasta
FASTA[ecoli_r10]=data/ecoli_r10/reads.fasta
FASTA[ecoli_r9]=data/ecoli/reads.fasta

chained_of(){  # gfa truemap -> fraction(TOTAL 行第5列)
  local gfa="$1" tm="$2"
  [[ -s "$gfa" && -s "$tm" ]] || { echo ""; return; }
  "$PYTHON" "$RHS/evaluate_gfa.py" "$gfa" "$tm" 2>/dev/null | awk -F'\t' '$1=="TOTAL"{print $5}'
}

for name in green_algae yeast human ecoli_r10 ecoli_r9; do
  S="$SRC/$name"; O="$OUT/$name"; mkdir -p "$O"
  NPAF="$S/neurosamble.paf"; RPAF="$S/rawsamble.paf"; MPAF="$S/mm2_overlaps.paf"; TM="$S/true_mappings.paf"
  if [[ ! -s "$NPAF" || ! -s "$RPAF" || ! -s "$MPAF" ]]; then echo "SKIP $name: 缺 paf"; continue; fi
  echo "########## $(date '+%F %T')  $name ##########"

  "$PYTHON" -m neurosamble.overlap.sanitize_paf --in_paf "$NPAF" --out_paf "$O/ns.clean.paf" --reads_fasta "${FASTA[$name]}" > "$O/ns.sanitize.log" 2>&1
  "$MINIASM" "$O/ns.clean.paf" > "$O/neurosamble.gfa" 2> "$O/ns.miniasm.log" || true
  "$MINIASM" "$RPAF" > "$O/rawsamble.gfa" 2> "$O/rs.miniasm.log" || true
  "$MINIASM" "$MPAF" > "$O/mm2.gfa"       2> "$O/mm2.miniasm.log" || true

  "$PYTHON" -m neurosamble.overlap.score --tool_paf "$NPAF" --truth_paf "$MPAF" --gfa "$O/neurosamble.gfa" --json "$O/neurosamble_score.json" > "$O/ns.score.log"  2>&1
  "$PYTHON" -m neurosamble.overlap.score --tool_paf "$RPAF" --truth_paf "$MPAF" --gfa "$O/rawsamble.gfa"   --json "$O/rawsamble_score.json"   > "$O/rs.score.log"  2>&1
  "$PYTHON" -m neurosamble.overlap.score --tool_paf "$MPAF" --truth_paf "$MPAF" --gfa "$O/mm2.gfa"         --json "$O/mm2_score.json"         > "$O/mm2.score.log" 2>&1
  chained_of "$O/neurosamble.gfa" "$TM" > "$O/ns_chained.val"
  chained_of "$O/rawsamble.gfa"   "$TM" > "$O/rs_chained.val"
  chained_of "$O/mm2.gfa"         "$TM" > "$O/mm2_chained.val"
  echo "  OK -> $O"
done

python3 - <<'PY'
import json,glob,os,csv
OUT="experiment/2.0/accuracy"; rows=[]
def L(p):
    try: return json.load(open(p))
    except Exception: return {}
def V(p):
    try:
        v=open(p).read().strip(); return round(float(v)*100,1) if v else ""
    except Exception: return ""
ov=lambda o,k:o.get("overlap",{}).get(k,"")
co=lambda o,k:o.get("contiguity",{}).get(k,"")
for d in sorted(glob.glob(OUT+"/*/")):
    name=os.path.basename(d.rstrip("/"))
    ns=L(d+"neurosamble_score.json"); rs=L(d+"rawsamble_score.json"); mm=L(d+"mm2_score.json")
    rows.append([name,
        ov(ns,"precision"),ov(ns,"recall"),ov(ns,"f1"),co(ns,"n50"),co(ns,"aun"),co(ns,"longest"),co(ns,"n_unitigs"),V(d+"ns_chained.val"),
        ov(rs,"precision"),ov(rs,"recall"),ov(rs,"f1"),co(rs,"n50"),co(rs,"aun"),co(rs,"longest"),co(rs,"n_unitigs"),V(d+"rs_chained.val"),
        co(mm,"n50"),co(mm,"aun"),co(mm,"longest"),co(mm,"n_unitigs"),V(d+"mm2_chained.val")])
hdr=["dataset","NS_P","NS_R","NS_F1","NS_N50","NS_auN","NS_longest","NS_utg","NS_chain%",
     "RS_P","RS_R","RS_F1","RS_N50","RS_auN","RS_longest","RS_utg","RS_chain%",
     "MM2_N50","MM2_auN","MM2_longest","MM2_utg","MM2_chain%"]
with open(OUT+"/ALL_SUMMARY.csv","w") as f:
    w=csv.writer(f); w.writerow(hdr); w.writerows(rows)
print("== experiment/2.0/accuracy/ALL_SUMMARY.csv ==")
for r in [hdr]+rows: print("\t".join(str(x) for x in r))
PY
echo "ALL DONE (accuracy)."
