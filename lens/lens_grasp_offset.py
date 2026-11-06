"""Where does the policy itself grasp the bowl?  From no-switch rollouts (trajectory files with ee, act, old_obj):
the end-effector position when the gripper is first commanded to close, relative to the instructed bowl (whose
position is recorded as old_obj), only for rollouts that reached their goal.  Printed per backbone and recording
(pair:which, e.g. "B->A:new" = scene 9 executing the instruction that is new in the B->A switch).
With --json OUT: {"pooled": {bb: [dx, dy]}, "scene": {bb: {"A->B": [dx, dy], "B->A": [dx, dy]}}, "n": {...}} in metres,
where "scene" is the own grasp offset for the object of the new instruction of each switch direction.
usage: lens_grasp_offset.py RUN_DIR [RUN_DIR ...] [--json OUT]"""
import argparse
import collections
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--json", default="")
a = ap.parse_args()
g = collections.defaultdict(list)
for d in a.runs:
    res = {}
    for p in glob.glob(os.path.join(d, "results_*.jsonl")):
        for ln in open(p):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            res[r["id"]] = r
    for eid, r in res.items():
        if r.get("error") or r.get("outcome") != "success" or r.get("ts") is not None:
            continue
        f = os.path.join(d, "traj", eid + ".npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        act, ee, obj = z["act"], z["ee"], z["old_obj"]
        c = np.where(act[:, 6] > 0)[0]
        if not len(c) or np.isnan(obj).any():
            continue
        k = int(c[0])
        g[(eid.split("_")[0], str(r.get("pair") or r.get("scene")))].append(np.r_[ee[k] - obj[k], k, ee[0] - obj[0]])
out = {"pooled": {}, "scene": {}, "n": {}}
for key in sorted(g, key=str):
    x = np.stack(g[key])
    off = x[:, :3] * 100
    print(f"{key[0]:5s} {key[1]:22s} n {len(x):3d} | grasp - bowl (cm): dx {off[:, 0].mean():5.1f}+-{off[:, 0].std():3.1f} "
          f"dy {off[:, 1].mean():5.1f}+-{off[:, 1].std():3.1f} dz {off[:, 2].mean():5.1f}+-{off[:, 2].std():3.1f} "
          f"|dxy| {np.linalg.norm(off[:, :2], axis=1).mean():4.1f} | step {x[:, 3].mean():5.1f} | start - bowl "
          f"dx {100 * x[:, 4].mean():5.1f} dy {100 * x[:, 5].mean():5.1f} dz {100 * x[:, 6].mean():5.1f}")
    bb, pair = key
    if pair.endswith(":new"):
        out["scene"].setdefault(bb, {})[pair.split(":")[0]] = [round(float(v), 4) for v in x[:, :2].mean(0)]
        out["n"][f"{bb} {pair}"] = int(len(x))
for bb in sorted(set(k[0] for k in g)):
    x = np.concatenate([np.stack(g[k]) for k in g if k[0] == bb])
    out["pooled"][bb] = [round(float(v), 4) for v in x[:, :2].mean(0)]
    out["n"][f"{bb} pooled"] = int(len(x))
    print(f"{bb:5s} pooled over all recordings: n {len(x)}, dx {100 * x[:, 0].mean():5.1f} dy {100 * x[:, 1].mean():5.1f} cm")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
    print(json.dumps(out))
