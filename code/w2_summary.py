"""Summary of Wormhole v2 runs. usage: w2_summary.py RUN_DIR [RUN_DIR ...]
Per backbone / variant / direction: early success (t_s=2), committed success (t_s in 12-42) with the
capture and freeze rates, completion time after the switch (median over successes), switch
discontinuity and transition jerk (as in the paper), and for v2 the traversal length."""
import collections
import glob
import json
import math
import os
import sys

import numpy as np

COMM = {12, 22, 32, 42}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def metrics(z, ts, H=20):
    a = z["act"][:, :6].astype(np.float64)
    if ts + 1 >= len(a):
        return float("nan"), float("nan")
    disc = float(np.linalg.norm(a[ts] - a[ts - 1]))
    seg = a[ts:ts + H]
    jerk = float(np.mean(np.linalg.norm(seg[2:] - 2 * seg[1:-1] + seg[:-2], axis=1))) if len(seg) >= 3 else float("nan")
    return disc, jerk


rows = []
for d in sys.argv[1:]:
    for p in glob.glob(os.path.join(d, "results_*.jsonl")):
        for ln in open(p):
            if ln.strip():
                r = json.loads(ln)
                r["_dir"] = d
                rows.append(r)
groups = collections.defaultdict(list)
for r in rows:
    if r.get("error") or r.get("outcome") == "done_before_switch" or r.get("invalid") or not r.get("variant"):
        continue
    bb = r["id"].split("_")[0]
    groups[(bb, r["variant"], r["pair"])].append(r)
    groups[(bb, r["variant"], "both")].append(r)
print("bb variant dir | early | committed [95% CI] | capture freeze | completion(med) | disc jerk | tunnel steps")
for key in sorted(groups):
    rs = groups[key]
    er = [r for r in rs if r["ts"] == 2]
    cr = [r for r in rs if r["ts"] in COMM]
    if not cr:
        continue
    s = sum(r["outcome"] == "success" for r in cr)
    lo, hi = wilson(s, len(cr))
    comp, disc, jerk, steps = [], [], [], []
    for r in cr:
        z = np.load(os.path.join(r["_dir"], "traj", r["id"] + ".npz"))
        dd, jj = metrics(z, r["ts"])
        disc.append(dd)
        jerk.append(jj)
        if r["outcome"] == "success":
            comp.append(r["t_new"] - r["ts"])
        for w in r.get("worm") or []:
            if "steps" in w:
                steps.append(w["steps"])
    e = 100 * np.mean([r["outcome"] == "success" for r in er]) if er else float("nan")
    print(f"{key[0]} {key[1]:8s} {key[2]:5s} | {e:5.1f} ({len(er)}) | {100*s/len(cr):5.1f} [{100*lo:.0f},{100*hi:.0f}] ({len(cr)}) | "
          f"{100*np.mean([r['outcome']=='capture' for r in cr]):5.1f} {100*np.mean([r['outcome']=='timeout' for r in cr]):5.1f} | "
          f"{(np.median(comp) if comp else float('nan')):6.1f} | {np.nanmean(disc):.2f} {np.nanmean(jerk):.3f} | "
          f"{(np.median(steps) if steps else float('nan')):.0f}")
# per switch step for the committed regime
print("\ncommitted success by t_s (both directions)")
for key in sorted(k for k in groups if k[2] == "both"):
    rs = groups[key]
    line = []
    for ts in sorted(COMM):
        x = [r["outcome"] == "success" for r in rs if r["ts"] == ts]
        line.append(f"{ts}:{sum(x)}/{len(x)}")
    print(key[0], key[1], " ".join(line))
