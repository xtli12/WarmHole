"""Build an entry library (Mouth B) from successful no-switch rollouts of the new instruction."""
import glob
import json
import sys

import numpy as np

exp_dir, pair_key, out, H = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tag = sys.argv[5] if len(sys.argv) > 5 else "lib"   # only held-out library rollouts (initial states 20-39)
rows = [json.loads(l) for p in glob.glob(f"{exp_dir}/results_*.jsonl") for l in open(p) if l.strip()]
states, segs, n_ep = [], [], 0
for r in rows:
    if r.get("pair") != pair_key or r.get("outcome") != "success" or r.get("tag") != tag:
        continue
    z = np.load(f"{exp_dir}/traj/{r['id']}.npz")
    ee, gr, act = z["ee"], z["grip"], z["act"]
    n_ep += 1
    for t in range(len(act)):
        seg = act[t:t + H]
        if len(seg) < H:
            seg = np.concatenate([seg, np.repeat(seg[-1:], H - len(seg), axis=0)])
        states.append(np.r_[ee[t], gr[t, 0] - gr[t, 1]])
        segs.append(seg)
np.savez(out, states=np.asarray(states, np.float32), segs=np.asarray(segs, np.float32))
print(f"{out}: {n_ep} episodes, {len(states)} entry states")
