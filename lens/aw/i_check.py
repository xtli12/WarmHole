"""Checks of round i (handover 7.15, switch-state relabeling with a kinematics-aware teacher).  Read-only.
usage:
  i_check.py smoke RUNS NAME           smoke gates: no errors; in every recorded episode the teacher takes over exactly at
                                       the switch (src -4 = the student before it, then the scripted transit src -2
                                       from the first switched frame, then the teacher policy src >= 0); B->A
                                       transits carry the corridor (corridor_clear >= 0.03 in the first worm entry),
                                       A->B transits do not
  i_check.py dsum RUNS NAME [...]      outcomes by direction x t_s, successes, corridor and transit statistics
  i_check.py data DIR [...]            round-i datasets (aw_build_data_c7 --cat_as 6 --policy_max 40): consistent
                                       meta.npz, category 6 only, fw 3 / 1, at most 40 policy frames per episode,
                                       training layouts only
  i_check.py enough RUNS NAME MIN MINBA  at least MIN successful episodes, of them at least MINBA B->A (7.15 data rule)
Exit status 0 = passed."""
import collections
import glob
import json
import os
import sys

import numpy as np

RES = 224
BB = "pi05"


def rows(runs, name):
    out = {}
    for f in sorted(glob.glob(os.path.join(runs, f"{BB}_{name}_p*", "results_*.jsonl"))):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            r["_dir"] = os.path.dirname(f)
            out[r["id"]] = r
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float("nan")


def w0(r):
    w = r.get("worm") or []
    return w[0] if w else {}


def smoke(runs, name):
    ok = True
    m = rows(runs, name)
    err = [r["id"] for r in m.values() if r.get("error")]
    print(f"{name}: {len(m)} rows, errors {len(err)} {err[:3]}; outcomes {dict(collections.Counter(r.get('outcome') for r in m.values()))}")
    ok &= bool(m) and not err
    nrec = 0
    for r in m.values():
        if r.get("error"):
            continue
        ts = int(r["ts"])
        f = os.path.join(r["_dir"], "steps", r["id"] + ".npz")
        good = True
        if os.path.exists(f):
            nrec += 1
            z = np.load(f)
            src, sw = z["src"].astype(int), z["sw"].astype(bool)
            s = int(np.where(sw)[0][0])
            good &= s == ts and bool((src[:s] == -4).all()) and int(src[s]) == -2
            good &= bool(((src[s:] == -2) | (src[s:] >= 0)).all())
        else:
            good = False
        cc = w0(r).get("corridor_clear")
        if r.get("pair") == "B->A":
            good &= cc is not None and cc >= 0.03 - 1e-9
        else:
            good &= cc is None
        if not good:
            print(f"  NOT OK {r['id']}: ts {ts}, step record {os.path.exists(f)}, corridor_clear {cc}")
        ok &= bool(good)
    print(f"step records {nrec}; corridor_clear (B->A): {[round(w0(r).get('corridor_clear', -1), 3) for r in m.values() if r.get('pair') == 'B->A']}")
    print("SMOKE", "OK" if ok else "FAILED")
    return 0 if ok else 1


def dsum(runs, names):
    for name in names:
        rs = rows(runs, name)
        r = [x for x in rs.values() if not x.get("error")]
        print(f"{name}: {len(r)} episodes, errors {len(rs) - len(r)}; successes {sum(x['outcome'] == 'success' for x in r)};"
              f" outcomes {dict(collections.Counter(x.get('outcome') for x in r))}")
        for p in ("A->B", "B->A"):
            xs = [x for x in r if x.get("pair") == p]
            if not xs:
                continue
            by = collections.defaultdict(list)
            for x in xs:
                by[int(x["ts"])].append(x["outcome"] == "success")
            st = [w0(x).get("steps") for x in xs]
            ae = [w0(x).get("arrive_err") for x in xs]
            cc = [w0(x).get("corridor_clear") for x in xs]
            print(f"  {p}: {sum(x['outcome'] == 'success' for x in xs)}/{len(xs)} | by t_s "
                  + " ".join(f"{t}:{sum(v)}/{len(v)}" for t, v in sorted(by.items()))
                  + f" | transit steps median {med(st):.0f}, arrival error median {100 * med(ae):.1f} cm"
                  + (f" | corridor clear median {med(cc):.3f} (min {min(c for c in cc if c is not None):.3f})" if any(c is not None for c in cc) else ""))
    return 0


def data(dirs):
    ok = True
    for d in dirs:
        m = np.load(os.path.join(d, "meta.npz"))
        N = int(m["N"])
        eps = json.load(open(os.path.join(d, "episodes.json")))
        nimg = os.path.getsize(os.path.join(d, "img.u8")) // (RES * RES * 3)
        nwr = os.path.getsize(os.path.join(d, "wrist.u8")) // (RES * RES * 3)
        cat, fw, init, ep = m["cat"], m["fw"], m["init"], m["ep"]
        npol = np.bincount(ep[fw == 1.0], minlength=len(eps)) if N else np.zeros(1)
        good = (N > 0 and nimg == N and nwr == N and len(cat) == N and len(fw) == N
                and set(np.unique(cat).tolist()) == {6} and set(np.unique(fw).tolist()) <= {1.0, 3.0}
                and sum(e["T"] for e in eps) == N and int(npol.max()) <= 40
                and bool(np.all(((init >= 20) & (init <= 37)) | ((init >= 10000) & (init <= 10499)))))
        pairs = collections.Counter(e.get("pair") for e in eps)
        print(f"{d}: {N} frames from {len(eps)} episodes {dict(pairs)}; transit {int((fw == 3).sum())}, policy {int((fw == 1).sum())}"
              f" (max {int(npol.max())} per episode), weight mass {float(fw.sum()):.0f} | {'ok' if good else 'NOT OK'}")
        ok &= good
    print("DATA", "OK" if ok else "FAILED")
    return 0 if ok else 1


def enough(runs, name, mn, mnba):
    r = [x for x in rows(runs, name).values() if not x.get("error") and x.get("outcome") == "success"]
    nba = sum(x.get("pair") == "B->A" for x in r)
    print(f"{name}: {len(r)} successful episodes (needed {mn}), B->A {nba} (needed {mnba})")
    return 0 if len(r) >= mn and nba >= mnba else 1


def main():
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "smoke":
        return smoke(rest[0], rest[1])
    if cmd == "dsum":
        return dsum(rest[0], rest[1:])
    if cmd == "data":
        return data(rest)
    if cmd == "enough":
        return enough(rest[0], rest[1], int(rest[2]), int(rest[3]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
