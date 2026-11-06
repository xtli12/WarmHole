"""Round i smoke (read-only): per episode the switch state and how the teacher's transit went (11-01 22:05).
usage: ismk_look.py RUN_DIR"""
import glob
import json
import os
import sys

import numpy as np

BASE = np.array([-0.66, 0.0])
d = sys.argv[1]
rows = {}
for p in glob.glob(os.path.join(d, "results_*.jsonl")):
    for ln in open(p):
        r = json.loads(ln)
        rows[r["id"]] = r
for rid, r in sorted(rows.items(), key=lambda kv: (kv[1].get("pair"), int(kv[1]["ts"]), kv[0])):
    f = os.path.join(d, "steps", rid + ".npz")
    if not os.path.exists(f):
        print(rid, "no step record")
        continue
    z = np.load(f)
    src, sw, ee, new, old = z["src"].astype(int), z["sw"].astype(bool), z["ee"].astype(float), z["new_obj"].astype(float), z["old_obj"].astype(float)
    s = int(np.where(sw)[0][0])
    tr = np.where(src == -2)[0]
    e = int(tr[-1]) if len(tr) else s
    w = (r.get("worm") or [{}])[0]
    tgt = np.asarray(w.get("target") or [np.nan] * 3, float)
    path = ee[s:e + 1]
    dmin = float(np.linalg.norm(path - tgt, axis=1).min()) if len(path) else float("nan")
    held = old[s, 2] - old[0, 2] > 0.02
    print(f"{r.get('pair')} ts {int(r['ts']):2d} {r.get('outcome'):8s} | switch: ee ({ee[s,0]:+.3f}, {ee[s,1]:+.3f}, {ee[s,2]:.3f}) reach {100*np.linalg.norm(ee[s,:2]-BASE):.0f} cm,"
          f" rel old ({100*(ee[s,0]-old[s,0]):+.1f}, {100*(ee[s,1]-old[s,1]):+.1f}, {100*(ee[s,2]-old[s,2]):+.1f}) held {int(held)}"
          f" | corridor {w.get('corridor_clear')} | transit {len(tr)} steps, target ({tgt[0]:+.3f}, {tgt[1]:+.3f}, {tgt[2]:.3f}),"
          f" end ({ee[e,0]:+.3f}, {ee[e,1]:+.3f}, {ee[e,2]:.3f}) err {100*np.linalg.norm(ee[e]-tgt):.1f} cm, min {100*dmin:.1f} cm,"
          f" z max {ee[s:e+1,2].max():.3f} | after: closest to new bowl {100*np.linalg.norm(ee[e:,:2]-new[e:,:2],axis=1).min():.1f} cm")
    # the z / reach path of the transit every 10 steps
    k = np.arange(s, e + 1, 10)
    print("    transit path (reach cm, z): " + " ".join(f"({100*np.linalg.norm(ee[i,:2]-BASE):.0f},{ee[i,2]:.3f})" for i in k))
