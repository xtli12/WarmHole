"""Where exactly is the B->A "freeze" pose, and how do the policies get there?  (read-only, CPU)

c1 (11-01, after the round-f smoke): fail_anatomy_c2 found that the T1 failures (never within 6 cm of the new bowl) end at
about (+9, +2, +23.5) cm from the new bowl for every model.  This reads the same trajectories (traj/<id>.npz: ee, grip,
act, old_obj, new_obj, prompt_new, bridge) and reports, per model x direction:
  the robot's start pose ee[0] and the bowls (absolute, m, medians);
  for T1 failures: the end pose (absolute), its distance to the start pose and to the pose at the switch, when the arm
    stopped (first step after which it stays within 3 cm of its end pose), the path from the switch to that step, the
    highest point after the switch, the mean |translation action| / |rotation action| / gripper command over the last
    30 steps, and the gripper opening at the end;
  for successes: the fraction whose post-switch path passes within 4 cm of the T1 end pose of the same model and
    direction, and the median number of steps they spend there;
  with --teach DIR ...: the scripted transit of the teacher (bridge == 1): start, highest point, hand-back point
    (absolute), and whether it passes within 4 cm of the median T1 end pose of the first model.
usage: freeze_anatomy.py RUNS_DIR NAME [NAME ...] [--teach DIR ...]"""
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


def pair_of(r):
    p = r.get("pair")
    if p:
        return p
    rid = str(r.get("id", ""))
    return "A->B" if "_spAB_" in rid else "B->A" if "_spBA_" in rid else None


def traj(d, rid):
    f = os.path.join(d, "traj", rid + ".npz")
    return np.load(f) if os.path.exists(f) else None


def switch_index(z, ts):
    if "prompt_new" in z.files:
        sw = np.where(z["prompt_new"] > 0)[0]
        return int(sw[0]) if len(sw) else None
    return int(ts)


def med(xs, q=50):
    xs = [x for x in xs if x is not None and np.all(np.isfinite(x))]
    return np.percentile(np.array(xs, dtype=float), q, axis=0) if len(xs) else np.nan


def fmt(v, s=100.0, p="%+.1f"):
    v = np.atleast_1d(np.asarray(v, dtype=float)) * s
    return "(" + ", ".join(p % x for x in v) + ")"


def episode(r):
    z = traj(r["_dir"], r["id"])
    if z is None:
        return None
    s = switch_index(z, r.get("ts"))
    ee, new, old = z["ee"].astype(float), z["new_obj"].astype(float), z["old_obj"].astype(float)
    if s is None or s >= len(ee) - 2:
        return None
    post = slice(s, len(ee))
    dxy = np.linalg.norm(ee[post, :2] - new[post, :2], axis=1)
    lifted = bool((new[post, 2] - new[s, 2]).max() > 0.03)
    end = ee[-1]
    far = np.where(np.linalg.norm(ee[s:] - end, axis=1) >= 0.03)[0]
    k = s + (int(far[-1]) + 1 if len(far) else 0)
    act = z["act"].astype(float) if "act" in z.files else None
    tail = slice(max(s, len(ee) - 30), len(ee))
    e = dict(s=s, n=len(ee), home=ee[0], at_switch=ee[s], end=end, new=new[s], old=old[s], new_end=new[-1],
             approach=bool((dxy < 0.06).any()), lifted=lifted, k=k,
             path_to_stop=float(np.linalg.norm(np.diff(ee[s:k + 1], axis=0), axis=1).sum()) if k > s else 0.0,
             zmax=float(ee[s:, 2].max()), ee=ee, held=bool(old[s, 2] - old[0, 2] > 0.02))
    if act is not None and len(act) == len(ee):
        e.update(a_tr=float(np.abs(act[tail, :3]).mean()), a_rot=float(np.abs(act[tail, 3:6]).mean()),
                 a_grip=float(act[tail, 6].mean()), a_tr_first=float(np.abs(act[s:s + 10, :3]).mean()))
    if "grip" in z.files:
        e["grip_end"] = float(np.abs(z["grip"][-1]).sum())
    if "bridge" in z.files:
        e["bridge"] = z["bridge"].astype(int)
    return e


def report(runs, names, ref=None):
    for name in names:
        rows = load(runs, name)
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows.values():
            ts = r.get("ts")
            if ts is None or int(ts) < 12 or not r.get("switched", True) or r.get("variant") not in ("flush", None):
                continue
            e = episode(r)
            if e is None:
                continue
            cls = ("S" if r["outcome"] == "success" else "C" if r["outcome"] == "capture" else
                   "T1" if not e["approach"] else "T2" if not e["lifted"] else "T3")
            by[pair_of(r)][cls].append(e)
            by[pair_of(r)]["all"].append(e)
        print(f"== {name}")
        for pair in sorted(by, key=str):
            g = by[pair]
            al = g["all"]
            print(f"  {pair}: {len(al)} switch episodes (t_s >= 12): " +
                  " ".join(f"{c}={len(g[c])}" for c in ("S", "T1", "T2", "T3", "C")) +
                  f" | start pose {fmt(med([e['home'] for e in al]), 1, '%.3f')} m, new bowl {fmt(med([e['new'] for e in al]), 1, '%.3f')},"
                  f" old bowl {fmt(med([e['old'] for e in al]), 1, '%.3f')}")
            t1 = g["T1"]
            if t1:
                endp = med([e["end"] for e in t1])
                if ref is not None:
                    ref.setdefault(pair, endp)
                dh = [np.linalg.norm(e["end"] - e["home"]) for e in t1]
                ds = [np.linalg.norm(e["end"] - e["at_switch"]) for e in t1]
                print(f"    T1 end pose {fmt(endp, 1, '%.3f')} m (quartiles x {fmt(med([e['end'][0] for e in t1], 25), 1, '%.3f')}"
                      f"-{fmt(med([e['end'][0] for e in t1], 75), 1, '%.3f')}, z {fmt(med([e['end'][2] for e in t1], 25), 1, '%.3f')}"
                      f"-{fmt(med([e['end'][2] for e in t1], 75), 1, '%.3f')});"
                      f" rel. new bowl {fmt(med([e['end'] - e['new_end'] for e in t1]))} cm, rel. start pose {fmt(med([e['end'] - e['home'] for e in t1]))} cm")
                print(f"    T1: |end - start pose| median {100 * med(dh):.1f} cm (<3 cm: {sum(x < 0.03 for x in dh)}/{len(dh)});"
                      f" |end - pose at switch| {100 * med(ds):.1f} cm; stopped {med([e['k'] - e['s'] for e in t1]):.0f} steps after the switch"
                      f" (episode ends {med([e['n'] - 1 - e['s'] for e in t1]):.0f} after it); path switch->stop {100 * med([e['path_to_stop'] for e in t1]):.1f} cm;"
                      f" highest point after the switch z {med([e['zmax'] for e in t1]):.3f}; old bowl held at the switch {sum(e['held'] for e in t1)}/{len(t1)}")
                if "a_tr" in t1[0]:
                    print(f"    T1 actions over the last 30 steps: |translation| {med([e['a_tr'] for e in t1]):.3f}"
                          f" (first 10 after the switch {med([e['a_tr_first'] for e in t1]):.3f}), |rotation| {med([e['a_rot'] for e in t1]):.3f},"
                          f" gripper command {med([e['a_grip'] for e in t1]):+.2f}; gripper opening at the end {med([e.get('grip_end') for e in t1]):.3f}")
                for e in t1[:6]:
                    pts = e["ee"][e["s"]::20]
                    print("      path every 20 steps from the switch: " + " ".join(fmt(p, 1, "%.2f") for p in pts[:12]))
                ok = g["S"]
                if ok:
                    near = [np.linalg.norm(e["ee"][e["s"]:] - endp, axis=1) < 0.04 for e in ok]
                    frac = sum(x.any() for x in near) / len(ok)
                    dwell = [int(x.sum()) for x in near if x.any()]
                    print(f"    successes passing within 4 cm of the T1 end pose: {frac:.2f} ({sum(x.any() for x in near)}/{len(ok)}),"
                          f" steps there (median) {med(dwell) if dwell else float('nan'):.0f}; |translation| first 10 after the switch"
                          f" {med([e.get('a_tr_first') for e in ok]):.3f}")


def teacher(dirs, ref):
    rows = {}
    for d in dirs:
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
    by = collections.defaultdict(list)
    for r in rows.values():
        ts = r.get("ts")
        if ts is None or int(ts) < 12 or not r.get("switched", True):
            continue
        e = episode(r)
        if e is None or "bridge" not in e:
            continue
        b = np.where(e["bridge"] > 0)[0]
        if len(b) == 0:
            continue
        e["b0"], e["b1"], e["ok"] = int(b[0]), int(b[-1]), r["outcome"] == "success"
        by[pair_of(r)].append(e)
    for pair, es in sorted(by.items(), key=str):
        ee = [e["ee"] for e in es]
        hb = [e["ee"][e["b1"]] for e in es]
        st = [e["ee"][e["b0"]] for e in es]
        zt = [e["ee"][e["b0"]:e["b1"] + 1, 2].max() for e in es]
        line = (f"== teacher {pair}: {len(es)} switch episodes (t_s >= 12), success {sum(e['ok'] for e in es)}; scripted transit from"
                f" {fmt(med(st), 1, '%.3f')} (highest z {med(zt):.3f}) to the hand-back point {fmt(med(hb), 1, '%.3f')} m in"
                f" {med([e['b1'] - e['b0'] for e in es]):.0f} steps; start pose {fmt(med([e['home'] for e in es]), 1, '%.3f')}")
        if ref and pair in ref:
            endp = ref[pair]
            near = [np.linalg.norm(x[e["b0"]:e["b1"] + 1] - endp, axis=1) < 0.04 for x, e in zip(ee, es)]
            after = [np.linalg.norm(x[e["b1"]:] - endp, axis=1) < 0.04 for x, e in zip(ee, es)]
            pre = [np.linalg.norm(x[:e["s"]] - endp, axis=1) < 0.04 for x, e in zip(ee, es)]
            line += (f" | within 4 cm of the T1 end pose: before the switch {sum(x.any() for x in pre)}/{len(es)},"
                     f" scripted transit {sum(x.any() for x in near)}/{len(es)}, after the hand-back {sum(x.any() for x in after)}/{len(es)}")
        print(line)
        for e in es[:4]:
            pts = e["ee"][e["b0"]:e["b1"] + 1:8]
            print("      scripted path every 8 steps: " + " ".join(fmt(p, 1, "%.2f") for p in pts[:12]))


def main():
    args = sys.argv[1:]
    teach = []
    if "--teach" in args:
        i = args.index("--teach")
        teach, args = args[i + 1:], args[:i]
    ref = {}
    if args:
        report(args[0], args[1:], ref)
    if teach:
        teacher(teach, ref)


if __name__ == "__main__":
    main()
