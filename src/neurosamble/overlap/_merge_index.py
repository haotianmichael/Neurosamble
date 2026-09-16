import json, os, sys
import numpy as np
enc, idx = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(enc, "encode_manifest.json")))
shards = m["shards"]
os.makedirs(idx, exist_ok=True)
wins = []
for sh in shards:
    wf = sh.get("win_file") or ("windows_shard%d.npy" % sh["rank"])
    wins.append(np.load(os.path.join(enc, wf)))
W = np.concatenate(wins, 0) if len(wins) > 1 else wins[0]
np.save(os.path.join(idx, "windows.npy"), W)
with open(os.path.join(idx, "read_ids.txt"), "w") as out:
    for sh in shards:
        rf = "read_ids_shard%d.txt" % sh["rank"]
        with open(os.path.join(enc, rf)) as f:
            for line in f:
                out.write(line if line.endswith("\n") else line + "\n")
print("[merge] windows.npy rows=%d ; read_ids.txt written" % W.shape[0])
