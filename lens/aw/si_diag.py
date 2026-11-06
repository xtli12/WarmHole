"""Before round i: are the switch states of the failures covered by the teacher data, and why are T3 / C failures lost?
(read-only, CPU; reads traj/<id>.npz: ee, grip, old_obj, new_obj, prompt_new; never touches the test split)

c1 (11-01 21:50).  c2 (11-01 22:30): --variants V1,V2 selects the result rows by variant (default flush; dg0c for the
round-i relabeling runs, whose class is then the teacher's outcome from the student's own switch state).
part A  switch-state coverage.  For every switch episode (t_s >= 12) the state at the switch: end effector relative to
        the old bowl (cm), its height, whether the old bowl is held (lifted > 2 cm since the start), gripper opening.
        Per set x direction x class (S success, T1 never within 6 cm of the new bowl, T2 approached but never lifted it,
        T3 lifted it but not placed, C capture): medians, and the distance (cm) to the nearest switch state of the
        teacher data in the same direction (feature: relative position 3 + height + 10 x held); "far" = beyond the 90th
        percentile of the teacher's own leave-one-out nearest-neighbour distances.  Per t_s as well.
part B  T3 / C anatomy.  The plate position of each layout = median final position of the new bowl over the successes
        of all sets on that layout (scene, init).  For T3 and C (and S as reference): old bowl held at the switch; final
        horizontal distance of the old / new bowl to the plate; new bowl resting on the old bowl (within 6 cm
        horizontally and > 1.5 cm higher); new bowl dropped away from the plate (> 8 cm); steps from the switch to the
        capture.
usage: si_diag.py RUNS [--variants V1,V2] --sets NAME[+NAME] [...] --teach DIR [DIR ...]   (NAME+NAME pools run names into one set)"""
import collections
import glob
import json
import os
import sys

import numpy as np

BASE = np.array([-0.66, 0.0])
VARIANTS = ("flush", None)          # c2: --variants a,b (e.g. dg0c for the round-i relabeling runs)


def rows_of(paths, tag):
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
                if int(r.get("ts") or 0) < 12 or not r.get("switched", True):
                    continue
                if tag != "teacher" and r.get("variant") not in VARIANTS:
                    continue
                r["_d"], r["_set"] = d, tag
                out[r["id"]] = r
    return out


def pair_of(r):
    p = r.get("pair")
    if p:
        return p
    return "B->A" if "_spBA_" in r["id"] else "A->B" if "_spAB_" in r["id"] else "?"


def load(r):
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
    ee, new, old = z["ee"].astype(float), z["new_obj"].astype(float), z["old_obj"].astype(float)
    grip = z["grip"].astype(float) if "grip" in z.files else np.zeros((len(ee), 2))
    if s + 2 >= len(ee) or np.isnan(new).any() or np.isnan(old).any():
        return None
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
    rel = (ee[s] - old[s]) * 100.0
    held = float(old[s, 2] - old[0, 2] > 0.02)
    feat = np.array([rel[0], rel[1], rel[2], ee[s, 2] * 100.0, 10.0 * held])
    return dict(c=c, s=s, ts=int(r["ts"]), pair=pair_of(r), scene=int(r.get("scene") or -1), init=int(r.get("init") or -1),
                rel=rel, z=ee[s, 2], reach=float(np.linalg.norm(ee[s, :2] - BASE)), held=held,
                grip=float(grip[s, 0] - grip[s, 1]), feat=feat, new_end=new[-1], old_end=old[-1], old0=old[0], new0=new[0],
                new_max=float(new[s:, 2].max()), t_old=r.get("t_old"), set=r["_set"], id=r["id"])


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else float("nan")


def part_a(sets, teach):
    tfe = collections.defaultdict(list)
    for e in teach:
        tfe[e["pair"]].append(e["feat"])
    thr = {}
    for p, fs in tfe.items():
        F = np.asarray(fs)
        d = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nn = d.min(1)
        thr[p] = float(np.percentile(nn, 90))
        print(f"teacher {p}: {len(F)} switch states; leave-one-out nearest-neighbour distance median {np.median(nn):.1f} cm,"
              f" 90th percentile {thr[p]:.1f} cm")

    def nnd(e):
        F = np.asarray(tfe.get(e["pair"], []))
        return float(np.linalg.norm(F - e["feat"], axis=1).min()) if len(F) else float("nan")

    print("-- part A: state at the switch (medians; rel = end effector - old bowl, cm; NN = distance to the nearest teacher"
          " switch state, cm; far = share beyond the teacher's 90th percentile)")
    for name, eps in list(sets.items()) + [("teacher", teach)]:
        for p in ("A->B", "B->A"):
            by = collections.defaultdict(list)
            for e in eps:
                if e["pair"] == p:
                    by[e["c"]].append(e)
            for c in ("S", "T1", "T2", "T3", "C"):
                xs = by.get(c, [])
                if not xs:
                    continue
                d = [nnd(e) for e in xs] if name != "teacher" else [float("nan")]
                far = np.mean([x > thr.get(p, np.inf) for x in d]) if name != "teacher" else float("nan")
                rel = np.array([e["rel"] for e in xs])
                print(f"  {name:10s} {p} {c:2s} n={len(xs):4d} | rel {med(rel[:, 0]):+5.1f} {med(rel[:, 1]):+5.1f} {med(rel[:, 2]):+5.1f}"
                      f" | z {100 * med([e['z'] for e in xs]):.1f} cm, reach {100 * med([e['reach'] for e in xs]):.0f} cm,"
                      f" held {np.mean([e['held'] for e in xs]):.2f}, grip {med([e['grip'] for e in xs]):.3f}"
                      f" | NN {med(d):.1f} cm, far {far:.2f} | t_s median {med([e['ts'] for e in xs]):.0f}")
    print("-- part A by t_s (B->A; NN distance median / far share / held share; students pooled per set)")
    for name, eps in list(sets.items()):
        line = f"  {name:10s}"
        for ts in (12, 22, 32, 42, 52, 62):
            xs = [e for e in eps if e["pair"] == "B->A" and e["ts"] == ts]
            if not xs:
                continue
            d = [nnd(e) for e in xs]
            t1 = [nnd(e) for e in xs if e["c"] == "T1"]
            line += (f" | t{ts}: n {len(xs)} S {sum(e['c'] == 'S' for e in xs)} T1 {len(t1)} NN {med(d):.1f}"
                     f" (T1 {med(t1):.1f}) held {np.mean([e['held'] for e in xs]):.2f}")
        print(line)
    tl = "  teacher   "
    for ts in sorted({e["ts"] for e in teach}):
        xs = [e for e in teach if e["pair"] == "B->A" and e["ts"] == ts]
        if xs:
            tl += f" | t{ts}: n {len(xs)} S {sum(e['c'] == 'S' for e in xs)} held {np.mean([e['held'] for e in xs]):.2f}"
    print(tl)


def part_b(sets, teach):
    allx = [e for eps in sets.values() for e in eps] + teach
    plate, zs = {}, {}
    g = collections.defaultdict(list)
    for e in allx:
        if e["c"] == "S":
            g[(e["scene"], e["init"])].append(e["new_end"])
    for k, v in g.items():
        v = np.asarray(v)
        plate[k] = np.median(v[:, :2], axis=0)
        zs[k] = float(np.median(v[:, 2]))
    print(f"-- part B: plate position from the successes of {len(plate)} layouts")
    for name, eps in list(sets.items()) + [("teacher", teach)]:
        for p in ("A->B", "B->A"):
            for c in ("S", "T3", "C", "T2", "T1"):
                xs = [e for e in eps if e["pair"] == p and e["c"] == c and (e["scene"], e["init"]) in plate]
                if not xs:
                    continue
                o2p = [100 * float(np.linalg.norm(e["old_end"][:2] - plate[(e["scene"], e["init"])])) for e in xs]
                n2p = [100 * float(np.linalg.norm(e["new_end"][:2] - plate[(e["scene"], e["init"])])) for e in xs]
                stacked = [float(np.linalg.norm(e["new_end"][:2] - e["old_end"][:2]) < 0.06 and e["new_end"][2] > e["old_end"][2] + 0.015)
                           for e in xs]
                old_on = [x < 5.0 for x in o2p]
                dropped = [x > 8.0 for x in n2p]
                held = [e["held"] for e in xs]
                dz_new = [100 * (e["new_end"][2] - zs[(e["scene"], e["init"])]) for e in xs]
                line = (f"  {name:10s} {p} {c:2s} n={len(xs):4d} | held at switch {np.mean(held):.2f} | old bowl ends within 5 cm"
                        f" of the plate {np.mean(old_on):.2f} (median {med(o2p):.1f} cm) | new bowl to plate median {med(n2p):.1f} cm,"
                        f" > 8 cm {np.mean(dropped):.2f} | new bowl on the old {np.mean(stacked):.2f} | new bowl height vs"
                        f" successes {med(dz_new):+.1f} cm")
                if c == "C":
                    dt = [int(e["t_old"]) - e["s"] for e in xs if e["t_old"] is not None]
                    line += f" | capture after {med(dt):.0f} steps (min {min(dt) if dt else -1})"
                if c == "T3":
                    both = sum(1 for a, b in zip(old_on, held) if a)
                    line += f" | T3 with the old bowl on the plate: {both}; of those held at the switch {sum(1 for a, b in zip(old_on, held) if a and b)}"
                print(line)


def main():
    global VARIANTS
    a = sys.argv[1:]
    if "--variants" in a:
        k = a.index("--variants")
        VARIANTS = tuple(x for x in a[k + 1].split(",")) + (None,)
        a = a[:k] + a[k + 2:]
    runs = a[0]
    i, j = a.index("--sets"), a.index("--teach")
    names, tdirs = a[i + 1:j], a[j + 1:]
    sets = {}
    for n in names:
        dirs = sorted(d for part in n.split("+") for d in glob.glob(os.path.join(runs, f"pi05_{part}_p*")))
        if not dirs:
            print(f"(no run directories for {n})")
            continue
        eps = [load(r) for r in rows_of(dirs, n).values()]
        sets[n] = [e for e in eps if e]
    teach = [e for e in (load(r) for r in rows_of(tdirs, "teacher").values()) if e]
    print(f"sets: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()) + f"; teacher {len(teach)} switch episodes (t_s >= 12)")
    part_a(sets, teach)
    part_b(sets, teach)


if __name__ == "__main__":
    main()
