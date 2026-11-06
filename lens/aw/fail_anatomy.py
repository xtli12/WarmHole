"""Where do the self-trained policies get stuck after an instruction switch?  (read-only)

c2 of fail_anatomy (11-01): for every switch episode with t_s >= 12 of the given result sets, reads the trajectory
(traj/<id>.npz: ee, old_obj, new_obj, prompt_new) and reports, per model x direction x regime (committed 12-42 / late
52-62):
  outcome classes as in c1 (T1 never within 6 cm horizontally of the new bowl, T2 approached but never lifted it,
  T3 lifted but not placed, C capture) and "still" (end effector moved < 2 cm over the last 40 steps);
  for T1 failures: the end-effector position relative to the new bowl at the end (dx, dy, dz in cm, medians) and the
  closest horizontal approach;
  for successes: steps from the switch to the first approach within 6 cm, and to success.
With --teach RUN_DIR [...]: the hand-back point of the teacher (last scripted step, src == -2) relative to the new
bowl, and the mean |action| (first 3 dims) over the last 5 scripted steps vs the first 5 policy steps after it.
usage: fail_anatomy.py RUNS_DIR NAME [NAME ...] [--teach DIR ...]"""
import collections
import glob
import json
import os
import sys

import numpy as np


def load(runs, name):
    rows = {}
    for d in sorted(glob.glob(os.path.join(runs, f"pi05_{name}_p*"))):
        for p in glob.glob(os.path.join(d, "results_*.jsonl")):
            for ln in open(p):
                try:
                    r = json.loads(ln)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("error") or r.get("outcome") is None:
                    continue
                r["_dir"] = d
                rows[r["id"]] = r
    return rows


def anatomy(r):
    f = os.path.join(r["_dir"], "traj", r["id"] + ".npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    ee, old, new = z["ee"], z["old_obj"], z["new_obj"]
    if "prompt_new" in z.files:
        sw = np.where(z["prompt_new"] > 0)[0]
        if len(sw) == 0:
            return None
        s = int(sw[0])
    else:
        s = int(r["ts"])
    if s >= len(ee):
        return None
    post = slice(s, len(ee))
    dxy = np.linalg.norm(ee[post, :2] - new[post, :2], axis=1)
    near = np.where(dxy < 0.06)[0]
    lifted = bool((new[post, 2] - new[s, 2]).max() > 0.03)
    tail = ee[max(s, len(ee) - 40):]
    path40 = float(np.linalg.norm(np.diff(tail, axis=0), axis=1).sum()) if len(tail) > 2 else 0.0
    rel_end = (ee[-1] - new[-1]) * 100.0
    held = bool(old[s, 2] - old[0, 2] > 0.02)
    return dict(approach=len(near) > 0, t_near=int(near[0]) if len(near) else None, min_dxy=float(dxy.min()) * 100,
                lifted=lifted, still=path40 < 0.02, path40=path40 * 100, rel_end=rel_end, held=held)


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def models(runs, names):
    for name in names:
        rows = load(runs, name)
        tab = collections.defaultdict(collections.Counter)
        t1 = collections.defaultdict(list)
        succ = collections.defaultdict(list)
        for r in rows.values():
            ts = r.get("ts")
            if ts is None or int(ts) < 12 or not r.get("switched", True) or r.get("variant") not in ("flush", None):
                continue
            key = (r.get("pair"), "late" if int(ts) >= 52 else "comm")
            tab[key]["n"] += 1
            a = anatomy(r)
            if a is None:
                tab[key]["noTraj"] += 1
                continue
            if r["outcome"] == "success":
                tab[key]["S"] += 1
                if a["t_near"] is not None and r.get("t_new"):
                    succ[key].append((a["t_near"], int(r["t_new"]) - int(ts)))
                continue
            c = ("C" if r["outcome"] == "capture" else "T1" if not a["approach"] else "T2" if not a["lifted"] else "T3")
            tab[key][c] += 1
            if a["still"]:
                tab[key][c + "still"] += 1
            if a["held"]:
                tab[key][c + "held"] += 1
            if c == "T1":
                t1[key].append(a)
        print(f"== {name}: {sum(t['n'] for t in tab.values())} switch episodes (t_s >= 12)")
        for key in sorted(tab, key=str):
            t = tab[key]
            line = f"  {key[0]:5s} {key[1]:4s} n={t['n']:3d} S={t['S']:3d} | " + " ".join(
                f"{k}={v}" for k, v in sorted(t.items()) if k not in ("n", "S"))
            if succ[key]:
                line += (f" | successes: first within 6 cm after {med([x[0] for x in succ[key]]):.0f} steps,"
                         f" success after {med([x[1] for x in succ[key]]):.0f}")
            print(line)
            if t1[key]:
                rel = np.array([a["rel_end"] for a in t1[key]])
                print(f"      T1 end position rel. new bowl (cm, median): dx {med(rel[:, 0]):+.1f} dy {med(rel[:, 1]):+.1f}"
                      f" dz {med(rel[:, 2]):+.1f}; closest horizontal {med([a['min_dxy'] for a in t1[key]]):.1f} cm;"
                      f" path over last 40 steps {med([a['path40'] for a in t1[key]]):.1f} cm")


def teacher(dirs):
    stats = collections.defaultdict(list)
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "steps", "*.npz"))):
            z = np.load(f)
            src = z["src"]
            if int(z["ts"]) < 12:
                continue
            k = np.where(src == -2)[0]
            if len(k) == 0:
                continue
            e = int(k[-1])
            if e + 6 >= len(src):
                continue
            new = z["new_obj"]
            rel = (z["ee"][e] - new[e]) * 100.0
            a = np.abs(z["act"][:, :3])
            pair = "A->B" if "_spAB_" in f else "B->A"
            stats[pair].append((rel, a[max(0, e - 4):e + 1].mean(), a[e + 1:e + 6].mean(), e - int(z["ts"])))
    for pair, xs in sorted(stats.items()):
        rel = np.array([x[0] for x in xs])
        print(f"== teacher {pair}: {len(xs)} switch episodes (t_s >= 12); hand-back point rel. new bowl (cm, median):"
              f" dx {med(rel[:, 0]):+.1f} dy {med(rel[:, 1]):+.1f} dz {med(rel[:, 2]):+.1f};"
              f" |action| last 5 scripted steps {med([x[1] for x in xs]):.3f}, first 5 policy steps {med([x[2] for x in xs]):.3f};"
              f" transit length {med([x[3] for x in xs]):.0f} steps")


def main():
    args = sys.argv[1:]
    teach = []
    if "--teach" in args:
        i = args.index("--teach")
        teach, args = args[i + 1:], args[:i]
    if args:
        models(args[0], args[1:])
    if teach:
        teacher(teach)


if __name__ == "__main__":
    main()
