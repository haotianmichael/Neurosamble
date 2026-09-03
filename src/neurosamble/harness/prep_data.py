"""Extract N real ecoli reads' signals + ground-truth coords from blow5 + truth.paf."""
import pickle, random, numpy as np, pyslow5

DATA = "/home/nfs/mahaotian/ESA/CALL_ESA/data/d2_ecoli_r94"
BLOW5 = f"{DATA}/ecoli_R9.blow5"
PAF   = f"{DATA}/truth.paf"
REF   = f"{DATA}/ref.fa"
OUT   = "/home/nfs/mahaotian/ESA/Neurosamble/data/ecoli_pairs.pkl"

N          = 5000     # 子采样条数（基线够用；要更多再加）
TRIM       = 1500     # 丢掉 read 开头的 adapter/stall 采样
INPUT_LEN  = 2000     # 每条取的样本数（= input_signal_len）
random.seed(42)

# --- paf: read_id -> (tstart, tend, strand, tname)，每条取对齐最长的那条 ---
best = {}
with open(PAF) as f:
    for line in f:
        c = line.split("\t")
        if len(c) < 12: continue
        qid, strand, tname = c[0], c[4], c[5]
        tstart, tend = int(c[7]), int(c[8])
        alen = tend - tstart
        if qid not in best or alen > best[qid][4]:
            best[qid] = (tstart, tend, strand, tname, alen)
print(f"[paf] {len(best)} aligned reads")

# --- ref: name -> seq（CFT073 单 contig）---
ref = {}; name=None; buf=[]
with open(REF) as f:
    for line in f:
        if line.startswith(">"):
            if name: ref[name]="".join(buf)
            name=line[1:].split()[0]; buf=[]
        else: buf.append(line.strip())
    if name: ref[name]="".join(buf)
ref_name = next(iter(ref)); ref_seq = ref[ref_name]
print(f"[ref] {ref_name} len={len(ref_seq)}")

# --- 子采样 read_id，从 blow5 抽信号 ---
want = set(random.sample([q for q in best if best[q][3]==ref_name], min(N, len(best))))
s = pyslow5.Open(BLOW5, "r")
sigs, coords, ends, strands = [], [], [], []
for rid in want:
    try:
        rd = s.get_read(rid, pA=True)      # pA=True -> 电流值
    except Exception:
        continue
    if rd is None: continue
    sig = np.asarray(rd["signal"], dtype=np.float32)[TRIM:TRIM+INPUT_LEN]
    if sig.size < 100: continue
    ts, te, st, _, _ = best[rid]
    sigs.append(sig); coords.append(ts); ends.append(te); strands.append(st)
s.close()
print(f"[blow5] extracted {len(sigs)} reads")

pickle.dump({"query_signals":sigs,"query_coords":coords,"query_ends":ends,
             "query_strands":strands,"reference_seq":ref_seq}, open(OUT,"wb"))
print(f"[out] {OUT}")