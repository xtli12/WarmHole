"""Checks of round g (handover 7.13, counterfactual attractor relabeling).  Read-only.
usage:
  g_check.py smoke RUNS MAIN CF      smoke gates 1-2: conditional take-overs (bh_client_lens15 dg_cond) of pi05_MAIN and
                                     their counterfactual branches pi05_CF: no errors; a take-over only while in transit
                                     (dg_dxy > 0.07); a skipped episode ends at or before t_s + k; in the step records the
                                     frames before the take-over are the student's (src -4 before the switch, -3 after)
                                     and the frames from it on are the teacher's (src -2 or >= 0); every branch restored
                                     within 1 cm of its stored state, switched from its first frame, teacher frames only
  g_check.py dsum RUNS NAME [...]    conditional DAgger runs: take-overs by reason and k, successes after a take-over,
                                     distance / time at the take-over
  g_check.py csum RUNS NAME [...]    counterfactual branches: outcomes, restore error
  g_check.py data C DIR [...]        round-g datasets (aw_build_data_c6 --cat_as C): consistent meta.npz, categories {4, C}
                                     only, fw 3 / 1 on category C, training layouts only
  g_check.py enough RUNS NAME MIN    at least MIN successful episodes with a take-over (the data-volume rule of 7.13)
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


def smoke(runs, main, cf):
    ok = True
    m = rows(runs, main)
    c = rows(runs, cf)
    err = [r["id"] for r in list(m.values()) + list(c.values()) if r.get("error")]
    print(f"{main}: {len(m)} rows, {cf}: {len(c)} rows, errors {len(err)} {err[:3]}")
    ok &= bool(m) and not err
    reasons = collections.Counter(r.get("dg_reason") for r in m.values())
    print(f"take-over reasons {dict(reasons)}; outcomes {dict(collections.Counter(r.get('outcome') for r in m.values()))}")
    ok &= sum(reasons[k] for k in ("k", "freeze")) > 0
    for r in m.values():
        rs, k, ts = r.get("dg_reason"), int(r.get("dg_k") or 0), int(r["ts"])
        if rs in ("k", "freeze"):
            good = (r.get("dg_dxy") or 0) > 0.07 and r.get("snap") and os.path.exists(r["snap"])
            f = os.path.join(r["_dir"], "steps", r["id"] + ".npz")
            if os.path.exists(f):
                z = np.load(f)
                src, sw = z["src"].astype(int), z["sw"].astype(bool)
                s = int(np.where(sw)[0][0])
                tk = s + int(r["dg_since"])
                good &= s == ts and bool((src[:s] == -4).all()) and bool((src[s:tk] == -3).all())
                good &= bool(((src[tk:] == -2) | (src[tk:] >= 0)).all()) and bool((src[tk:] == -2).any())
                good &= (rs == "k" and int(r["dg_since"]) == k) or (rs == "freeze" and int(r["dg_since"]) < k)
            elif r.get("outcome") is not None:
                good = False
            if not good:
                print(f"  NOT OK take-over {r['id']}: dxy {r.get('dg_dxy')}, since {r.get('dg_since')}, k {k}, snap {r.get('snap')}")
            ok &= bool(good)
        elif rs == "none":
            good = r.get("outcome") == "skip" and int(r.get("steps", 10 ** 6)) <= ts + k
            good &= not os.path.exists(os.path.join(r["_dir"], "steps", r["id"] + ".npz"))
            if not good:
                print(f"  NOT OK skip {r['id']}: steps {r.get('steps')}, ts {ts}, k {k}, dxy {r.get('dg_dxy')}")
            ok &= bool(good)
    print(f"take-overs: dxy {[round(r['dg_dxy'], 3) for r in m.values() if r.get('dg_reason') in ('k', 'freeze')]}"
          f"; skips: dxy {[round(r.get('dg_dxy') or 0, 3) for r in m.values() if r.get('dg_reason') == 'none']}")
    if not c:
        print("no counterfactual branch: NOT OK")
        ok = False
    for r in c.values():
        f = os.path.join(r["_dir"], "steps", r["id"] + ".npz")
        good = r.get("resume_err") is not None and r["resume_err"] < 0.01 and int(r["ts"]) == 0 and r.get("branch_of") in m
        if os.path.exists(f):
            z = np.load(f)
            src, sw = z["src"].astype(int), z["sw"].astype(bool)
            good &= bool(sw.all()) and bool(((src == -2) | (src >= 0)).all()) and bool((src == -2).any())
        elif r.get("outcome") == "success":
            good = False
        if not good:
            print(f"  NOT OK branch {r['id']}: resume_err {r.get('resume_err')}, ts {r.get('ts')}, of {r.get('branch_of')}")
        ok &= bool(good)
    print(f"branches: outcomes {dict(collections.Counter(r.get('outcome') for r in c.values()))}; restore error (m)"
          f" max {max([r.get('resume_err') or 0 for r in c.values()] or [0]):.2e}")
    print("SMOKE", "OK" if ok else "FAILED")
    return 0 if ok else 1


def dsum(runs, names):
    for name in names:
        r = [x for x in rows(runs, name).values() if not x.get("error")]
        tk = [x for x in r if x.get("dg_reason") in ("k", "freeze")]
        line = f"{name}: {len(r)} episodes; " + ", ".join(
            f"{k} {v}" for k, v in sorted(collections.Counter(str(x.get('dg_reason')) for x in r).items()))
        line += f"; take-overs {len(tk)}, successful {sum(x['outcome'] == 'success' for x in tk)}"
        by = collections.defaultdict(list)
        for x in tk:
            by[(x.get("dg_reason"), int(x.get("dg_k") or 0))].append(x)
        line += " | " + "; ".join(f"{rs} k={k}: {len(xs)} ({sum(x['outcome'] == 'success' for x in xs)} ok, dxy {med([x['dg_dxy'] for x in xs]):.3f},"
                                  f" since {med([x['dg_since'] for x in xs]):.0f})" for (rs, k), xs in sorted(by.items(), key=str))
        late = [x for x in tk if int(x["ts"]) >= 52]
        line += f" | late t_s: {len(late)} take-overs; procedural layouts {sum(int(x['init']) >= 10000 for x in tk)}"
        print(line)
    return 0


def csum(runs, names):
    for name in names:
        r = [x for x in rows(runs, name).values() if not x.get("error")]
        e = [x for x in rows(runs, name).values() if x.get("error")]
        print(f"{name}: {len(r)} branches, {len(e)} errors; outcomes {dict(collections.Counter(x.get('outcome') for x in r))};"
              f" restore error max {max([x.get('resume_err') or 0 for x in r] or [0]):.2e} m")
    return 0


def data(c, dirs):
    ok = True
    for d in dirs:
        m = np.load(os.path.join(d, "meta.npz"))
        N = int(m["N"])
        eps = json.load(open(os.path.join(d, "episodes.json")))
        nimg = os.path.getsize(os.path.join(d, "img.u8")) // (RES * RES * 3)
        nwr = os.path.getsize(os.path.join(d, "wrist.u8")) // (RES * RES * 3)
        cat, fw, init = m["cat"], m["fw"], m["init"]
        sel = cat == c
        good = (N > 0 and nimg == N and nwr == N and len(cat) == N and len(fw) == N and bool(sel.any())
                and set(np.unique(cat).tolist()) <= {4, c} and set(np.unique(fw[sel]).tolist()) <= {1.0, 3.0}
                and sum(e["T"] for e in eps) == N
                and bool(np.all(((init >= 20) & (init <= 37)) | ((init >= 10000) & (init <= 10499)))))
        print(f"{d}: {N} frames from {len(eps)} episodes; category {c}: {int(sel.sum())} frames (transit {int((sel & (fw == 3)).sum())},"
              f" policy {int((sel & (fw == 1)).sum())}), weight mass {float(fw[sel].sum()):.0f}; student frames {int((cat == 4).sum())}"
              f" | {'ok' if good else 'NOT OK'}")
        ok &= good
    print("DATA", "OK" if ok else "FAILED")
    return 0 if ok else 1


def enough(runs, name, mn):
    r = [x for x in rows(runs, name).values() if not x.get("error") and x.get("dg_reason") in ("k", "freeze")]
    n = sum(x["outcome"] == "success" for x in r)
    print(f"{name}: {n} successful take-overs (needed {mn})")
    return 0 if n >= mn else 1


def main():
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "smoke":
        return smoke(rest[0], rest[1], rest[2])
    if cmd == "dsum":
        return dsum(rest[0], rest[1:])
    if cmd == "csum":
        return csum(rest[0], rest[1:])
    if cmd == "data":
        return data(int(rest[0]), rest[1:])
    if cmd == "enough":
        return enough(rest[0], rest[1], int(rest[2]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
