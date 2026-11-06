"""Round i data (read-only): how well does the teacher do from the students' own switch states, by how far out the arm
is at the switch?  (handover 7.15; reads results_*.jsonl and steps/<id>.npz of the relabeling runs pi05_i<student>)

c1 (11-01 22:15).  Per run and direction: buckets of the end effector's horizontal distance from the robot base at the
switch (< 64, 64-68, 68-72, 72-75, >= 75 cm) x t_s groups (12-27, 32-42, 47-62): episodes, teacher successes, median
scripted-transit steps, share of transits that reached the hand-back point (< 2 cm), median arrival error, median
inward speed of the transit over its first 30 steps after the lift (cm per step toward the base).
usage: i_reach.py RUNS NAME [NAME ...]"""
import collections
import glob
import json
import os
import sys

import numpy as np

BASE = np.array([-0.66, 0.0])
EDGES = [0.64, 0.68, 0.72, 0.75]
LAB = ["<64", "64-68", "68-72", "72-75", ">=75"]


def bucket(x):
    return int(np.searchsorted(EDGES, x, side="right"))


def tsg(ts):
    return "12-27" if ts <= 27 else "32-42" if ts <= 42 else "47-62"


def main():
    runs, names = sys.argv[1], sys.argv[2:]
    for name in names:
        rows = {}
        for p in glob.glob(os.path.join(runs, f"pi05_{name}_p*", "results_*.jsonl")):
            for ln in open(p):
                try:
                    r = json.loads(ln)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("error") or r.get("outcome") is None:
                    continue
                r["_d"] = os.path.dirname(p)
                rows[r["id"]] = r
        tab = collections.defaultdict(list)
        for r in rows.values():
            f = os.path.join(r["_d"], "steps", r["id"] + ".npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            src, sw, ee = z["src"].astype(int), z["sw"].astype(bool), z["ee"].astype(float)
            s = int(np.where(sw)[0][0])
            tr = np.where(src == -2)[0]
            reach = np.linalg.norm(ee[:, :2] - BASE, axis=1)
            w = (r.get("worm") or [{}])[0]
            # inward speed: reach decrease per step over the 30 steps after the end effector stopped rising (the lift)
            if len(tr) > 5:
                zt = ee[tr, 2]
                k0 = tr[int(np.argmax(zt))] if r.get("pair") == "B->A" else tr[0]
                k1 = min(k0 + 30, tr[-1])
                v = 100 * (reach[k0] - reach[k1]) / max(1, k1 - k0)
            else:
                v = float("nan")
            tab[(r.get("pair"), bucket(reach[s]), tsg(int(r["ts"])))].append(
                (r["outcome"] == "success", len(tr), w.get("arrive_err"), v))
        print(f"== {name}: {sum(len(v) for v in tab.values())} episodes with step records")
        for p in ("A->B", "B->A"):
            for b in range(len(LAB)):
                for g in ("12-27", "32-42", "47-62"):
                    xs = tab.get((p, b, g), [])
                    if not xs:
                        continue
                    ok = sum(x[0] for x in xs)
                    st = np.median([x[1] for x in xs])
                    ae = [x[2] for x in xs if x[2] is not None]
                    arr = np.mean([a < 0.02 for a in ae]) if ae else float("nan")
                    vv = [x[3] for x in xs if np.isfinite(x[3])]
                    print(f"  {p} reach {LAB[b]:>5s} cm, t_s {g}: n {len(xs):4d}, teacher successes {ok:4d} ({100 * ok / len(xs):5.1f}%)"
                          f" | transit steps median {st:.0f}, reached the hand-back point {arr:.2f}, arrival error median"
                          f" {100 * (np.median(ae) if ae else float('nan')):.1f} cm | inward speed {np.median(vv) if vv else float('nan'):.2f} cm/step")


if __name__ == "__main__":
    main()
