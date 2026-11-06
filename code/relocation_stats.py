"""Where does each method hand the robot back to the base policy, and does that predict success?

For every committed switch (t_s in 12-42) the hand-over step is the first step after the bridge
(Wormhole v1/v2) or after the scripted retreat, and t_s for the other methods. We measure the distance of
the end effector at hand-over to the approach phase of successful executions of the new instruction
(the entry memory, states before the gripper first closes).
usage: relocation_stats.py LIB_DIR OUT_CSV RUN_DIR [RUN_DIR ...]"""
import csv
import glob
import json
import os
import sys

import numpy as np

lib_dir, out_csv = sys.argv[1], sys.argv[2]
COMM = {12, 22, 32, 42}
libs = {}


def manifold(bb, d):
    if (bb, d) not in libs:
        z = np.load(os.path.join(lib_dir, f"{bb}_sp{d}_v2.npz"))
        m = z["t"] <= z["t_close"]
        libs[(bb, d)] = z["pos"][m].astype(np.float64)
    return libs[(bb, d)]


out = []
for run in sys.argv[3:]:
    for p in glob.glob(os.path.join(run, "results_*.jsonl")):
        for ln in open(p):
            r = json.loads(ln)
            if r.get("error") or r.get("outcome") == "done_before_switch" or r.get("ts") not in COMM:
                continue
            if not r.get("variant"):
                continue
            bb = r["id"].split("_")[0]
            d = "AB" if r["pair"] == "A->B" else "BA"
            z = np.load(os.path.join(run, "traj", r["id"] + ".npz"))
            ts = r["ts"]
            flag = None
            if "bridge" in z.files and z["bridge"].any():
                flag = z["bridge"]
            elif "retreat" in z.files and z["retreat"].any():
                flag = z["retreat"]
            h = int(np.where(flag)[0][-1]) + 1 if flag is not None else ts
            h = min(h, len(z["ee"]) - 1)
            M = manifold(bb, d)
            dist = float(np.sqrt(((M - z["ee"][h]) ** 2).sum(axis=1)).min())
            out.append(dict(bb=bb, variant=r["variant"], pair=r["pair"], ts=ts, init=r["init"], outcome=r["outcome"],
                            handover=h - ts, d_new=round(dist, 4)))
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print(out_csv, len(out))
for bb in sorted({o["bb"] for o in out}):
    for v in sorted({o["variant"] for o in out if o["bb"] == bb}):
        rs = [o for o in out if o["bb"] == bb and o["variant"] == v]
        dn = np.array([o["d_new"] for o in rs])
        sc = np.array([o["outcome"] == "success" for o in rs])
        bins = [(0, 0.03), (0.03, 0.08), (0.08, 0.15), (0.15, 1)]
        txt = " ".join(f"[{a:.2f},{b:.2f}): {sc[(dn >= a) & (dn < b)].mean() if ((dn >= a) & (dn < b)).any() else float('nan'):.2f} (n={((dn >= a) & (dn < b)).sum()})" for a, b in bins)
        print(f"{bb} {v:9s} n={len(rs):3d} succ={sc.mean():.2f} median d_new={np.median(dn):.3f} | {txt}")
