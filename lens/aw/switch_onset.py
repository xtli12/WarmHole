"""What does the policy do in the first steps after a B->A switch, in episodes that later freeze vs succeed?  (read-only)

c1 (11-01 21:30, after round h).  For every B->A switch episode (t_s >= 12) of the given result sets (traj/<id>.npz:
ee, act, new_obj, old_obj, prompt_new), classes as in fail_anatomy (S success, T1 never within 6 cm of the new bowl,
T2 / T3 / C).  Reports per class (medians): the end-effector height above the old (cabinet) bowl and its horizontal
distance from the robot base at the switch; the mean commanded translation over the first 10 steps after the switch
(dz, and the horizontal component toward the new bowl), the displacement over the first 20 / 40 steps (dz, toward the
new bowl), and the share of episodes whose first-10-step mean dz is negative (moving down).  With --teach DIR...: the
same numbers for the teacher's switch episodes (scripted transit from the switch).
usage: switch_onset.py RUNS NAME [NAME ...] [--teach DIR ...]"""
import collections
import glob
import json
import os
import sys

import numpy as np

BASE = np.array([-0.66, 0.0])          # robot base (x, y) in the LIBERO table frame (approx.; only relative use)


def rows_of(paths):
    out = {}
    for d in paths:
        for p in glob.glob(os.path.join(d, "results_*.jsonl")):
            for ln in open(p):
                try:
                    r = json.loads(ln)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("error") or r.get("outcome") is None:
                    continue
                r["_d"] = d
                out[r["id"]] = r
    return out


def onset(r):
    f = os.path.join(r["_d"], "traj", r["id"] + ".npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    if "prompt_new" not in z.files:
        return None
    sw = np.where(z["prompt_new"] > 0)[0]
    if len(sw) == 0:
        return None
    s = int(sw[0])
    ee, act, new, old = z["ee"].astype(float), z["act"].astype(float), z["new_obj"].astype(float), z["old_obj"].astype(float)
    if s + 41 >= len(ee):
        return None
    u = new[s, :2] - ee[s, :2]
    u = u / (np.linalg.norm(u) + 1e-9)
    dxy = np.linalg.norm(ee[s:, :2] - new[s:, :2], axis=1)
    lifted = bool((new[s:, 2] - new[s, 2]).max() > 0.03)
    if r["outcome"] == "success":
        c = "S"
    elif r["outcome"] == "capture":
        c = "C"
    elif not (dxy < 0.06).any():
        c = "T1"
    else:
        c = "T3" if lifted else "T2"
    a10 = act[s:s + 10, :3].mean(0)
    return dict(c=c, h_old=ee[s, 2] - old[s, 2], reach=float(np.linalg.norm(ee[s, :2] - BASE)), az10=a10[2],
                anew10=float(a10[:2] @ u), dz20=ee[s + 20, 2] - ee[s, 2], dn20=float((ee[s + 20, :2] - ee[s, :2]) @ u),
                dz40=ee[s + 40, 2] - ee[s, 2], dn40=float((ee[s + 40, :2] - ee[s, :2]) @ u), down=bool(a10[2] < 0),
                ts=int(r.get("ts") or 0))


def table(label, eps):
    by = collections.defaultdict(list)
    for e in eps:
        by[e["c"]].append(e)
    print(f"== {label}: {len(eps)} B->A switch episodes (t_s >= 12)")
    for c in ("S", "T1", "T2", "T3", "C"):
        xs = by.get(c, [])
        if not xs:
            continue
        m = lambda k: float(np.median([x[k] for x in xs]))  # noqa: E731
        print(f"  {c:2s} n={len(xs):4d} | at the switch: {100 * m('h_old'):+.1f} cm above the old bowl, reach {100 * m('reach'):.0f} cm"
              f" | first 10 steps: dz {m('az10'):+.3f}, toward new {m('anew10'):+.3f}, moving down {np.mean([x['down'] for x in xs]):.2f}"
              f" | after 20 steps: dz {100 * m('dz20'):+.1f} cm, toward new {100 * m('dn20'):+.1f} cm | after 40: dz {100 * m('dz40'):+.1f},"
              f" toward new {100 * m('dn40'):+.1f} cm | t_s median {m('ts'):.0f}")


def main():
    args = sys.argv[1:]
    teach = []
    if "--teach" in args:
        i = args.index("--teach")
        teach, args = args[i + 1:], args[:i]
    runs, names = args[0], args[1:]
    for n in names:
        rs = rows_of(sorted(glob.glob(os.path.join(runs, f"pi05_{n}_p*"))))
        eps = [onset(r) for r in rs.values() if r.get("pair") == "B->A" and int(r.get("ts") or 0) >= 12
               and r.get("variant") in ("flush", None)]
        table(n, [e for e in eps if e])
    if teach:
        rs = rows_of(teach)
        eps = [onset(r) for r in rs.values() if (r.get("pair") == "B->A" or "_spBA_" in str(r.get("id"))) and int(r.get("ts") or 0) >= 12]
        table("teacher (wo_off_h15)", [e for e in eps if e])


if __name__ == "__main__":
    main()
