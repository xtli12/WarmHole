"""m_check.py c2 (11-02 14:30; c1 11-02 14:00; round m = round j on pi0): j_check.py c1 with the backbone prefix of the run directories taken from
the environment variable BBP (pi05 | pi0, default pi05), plus e_check's `runs` (completeness) and `clean` (move error
rows) for the same prefix.  Checks of round j (handover 7.16, kinematically safe wormhole exit).  Read-only except `filter` and `pilot` (writes
the pilot decision file).
usage:
  j_check.py smoke RUNS NAME                 smoke gates: no errors; every margin-variant episode logs margin_start /
                                             margin_min / margin_handback and a hand-over reason arrive / margin; no
                                             B->A margin hand-over [c2: reported only]; at least one A->B margin hand-over; at a margin
                                             hand-over the margin is below the threshold and was higher at the start
  j_check.py repro RUNS NAME ORIGDIR [...]   the original-variant episodes of NAME against the same episodes of the
                                             original teacher data (ORIGDIR = pi05_teach_p*): outcome and the largest
                                             end-effector deviation before the switch / over the episode (report only)
  j_check.py pilot RUNS OUT NAME:M [...] -- ORIGDIR [...]
                                             pilot of 7.16: per margin M, successes on its episodes, hand-over reasons,
                                             margin at the hand-back; the original teacher on the same conditions; the
                                             rule (more successes, tie -> the larger M; NOGO if the selected M has more
                                             than 1 success fewer than the original teacher) -> OUT ("M=1.2" or "NOGO")
  j_check.py filter ORIGDIR OUTDIR           a copy of a teacher run directory without its A->B switch episodes:
                                             results_*.jsonl filtered, steps/ and traj/ linked (no data copied)
  j_check.py count DIR [...]                 rows per kind (A->B / B->A switch, no-switch) and successes
  j_check.py data DIR [...]                  the round-j teacher dataset (aw_build_data_c7 without --cat_as): consistent
                                             meta.npz, categories 0-3
Exit status 0 = passed."""
import collections
import glob
import json
import os
import sys

import numpy as np

BB = os.environ.get("BBP", "pi05")


def rows_dir(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "results_*.jsonl"))):
        for ln in open(p):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            r["_dir"] = d
            out[r["id"]] = r
    return out


def rows(runs, name):
    out = {}
    for d in sorted(glob.glob(os.path.join(runs, f"{BB}_{name}_p*"))):
        out.update(rows_dir(d))
    return out


def w0(r):
    w = r.get("worm") or []
    return w[0] if w else {}


def kind(r):
    rid = r["id"]
    if "_spAB_" in rid and r.get("ts") is not None and str(r.get("variant", "")).startswith("wo_off"):
        return "A->B switch"
    if "_spBA_" in rid and r.get("ts") is not None and str(r.get("variant", "")).startswith("wo_off"):
        return "B->A switch"
    return "other"


def smoke(runs, name):
    ok = True
    m = rows(runs, name)
    err = [r["id"] for r in m.values() if r.get("error")]
    print(f"{name}: {len(m)} rows, errors {len(err)} {err[:3]}; outcomes {dict(collections.Counter(r.get('outcome') for r in m.values()))}")
    ok &= bool(m) and not err
    nmab = 0
    nmba = 0
    for r in m.values():
        if r.get("error") or "m" not in str(r.get("variant", "")).replace("wo_off_h15", ""):
            continue
        w = w0(r)
        good = all(k in w for k in ("margin_start", "margin_min", "margin_handback")) and w.get("handover") in ("arrive", "margin")
        thr = float(r.get("lens_margin") or 0) or float(str(r["variant"]).split("m")[-1]) / 10
        if w.get("handover") == "margin":
            good &= w["margin_handback"] < thr + 0.02 and w["margin_start"] > w["margin_handback"]
            if r.get("pair") == "B->A":
                # c2 (round m amendment, handover 7.19, 11-02 14:30): reported, not a failure.  On pi0 a B->A
                # traversal can start near the elbow limit (smoke 14:12: margin 1.33 at t_s = 42 -> hand-back at
                # 1.18); the pilot and the new teacher data of phases 2-3 use A->B switches only (--m_dirs AB), so a
                # B->A margin hand-over in the smoke does not enter any dataset.
                nmba += 1
                print(f"  B->A margin hand-over (reported, not a gate): {r['id']}: {w}")
            else:
                nmab += 1
        if not good:
            print(f"  NOT OK {r['id']}: {w}")
        ok &= bool(good)
    print(f"A->B margin hand-overs: {nmab}; B->A margin hand-overs (reported only): {nmba}; margins at the hand-back: "
          f"{[(r['pair'], w0(r).get('handover'), round(w0(r).get('margin_handback', -1), 2)) for r in m.values() if w0(r)]}")
    ok &= nmab >= 1
    print("SMOKE", "OK" if ok else "FAILED")
    return 0 if ok else 1


def orig_key(r):
    return (r.get("pair"), int(r["ts"]), int(r["init"]))


def orig_rows(dirs):
    out = {}
    for d in dirs:
        for r in rows_dir(d).values():
            if r.get("variant") == "wo_off_h15" and not r.get("error") and r.get("outcome") is not None:
                out[orig_key(r)] = r
    return out


def repro(runs, name, dirs):
    o = orig_rows(dirs)
    for r in sorted(rows(runs, name).values(), key=lambda x: x["id"]):
        if r.get("variant") != "wo_off_h15" or r.get("error"):
            continue
        q = o.get(orig_key(r))
        if q is None:
            print(f"  {r['id']}: no original episode")
            continue
        a = np.load(os.path.join(r["_dir"], "traj", r["id"] + ".npz"))["ee"]
        b = np.load(os.path.join(q["_dir"], "traj", q["id"] + ".npz"))["ee"]
        n = min(len(a), len(b))
        ts = int(r["ts"])
        d = np.linalg.norm(a[:n] - b[:n], axis=1)
        print(f"  {r['id']}: outcome {r['outcome']} (original {q['outcome']}), steps {r.get('steps')} / {q.get('steps')};"
              f" largest end-effector deviation before the switch {1000 * d[:ts].max():.2f} mm, over the episode {1000 * d.max():.1f} mm")
    return 0


def pilot(runs, out, arms, dirs):
    o = orig_rows(dirs)
    res = {}
    for arm in arms:
        name, m = arm.split(":")
        rs = [r for r in rows(runs, name).values() if not r.get("error") and r.get("outcome") is not None]
        succ = sum(r["outcome"] == "success" for r in rs)
        base = [o[orig_key(r)]["outcome"] == "success" for r in rs if orig_key(r) in o]
        hov = collections.Counter(w0(r).get("handover") for r in rs)
        mh = [w0(r).get("margin_handback") for r in rs if w0(r).get("margin_handback") is not None]
        mm = [w0(r).get("margin_min") for r in rs if w0(r).get("margin_min") is not None]
        st = [w0(r).get("steps") for r in rs if w0(r).get("steps") is not None]
        res[float(m)] = (succ, sum(base), len(rs), len(base))
        print(f"{name} (M = {m}): {succ}/{len(rs)} successes; original teacher on the same conditions {sum(base)}/{len(base)};"
              f" hand-over {dict(hov)}; margin at the hand-back median {np.median(mh) if mh else float('nan'):.2f},"
              f" minimum during the transit median {np.median(mm) if mm else float('nan'):.2f}; transit steps median"
              f" {np.median(st) if st else float('nan'):.0f}")
    best = max(res, key=lambda k: (res[k][0], k))
    succ, base, n, nb = res[best]
    if succ < base - 1:
        line = "NOGO"
        why = f"M = {best}: {succ} successes < original {base} - 1"
    else:
        line = f"M={best}"
        why = f"{succ} successes vs original {base} (allowed: at least {base - 1})"
    print(f"selection (7.16: more successes, tie -> larger M; NOGO if more than 1 success below the original): {line} ({why})")
    with open(out, "w") as f:
        f.write(line + "\n" + why + "\n")
    return 0 if line != "NOGO" else 1


def filt(orig, outd):
    os.makedirs(outd, exist_ok=True)
    n_in = n_out = 0
    for p in sorted(glob.glob(os.path.join(orig, "results_*.jsonl"))):
        lines = open(p).read().splitlines()
        keep = []
        for ln in lines:
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            n_in += 1
            if kind(r) == "A->B switch":
                continue
            keep.append(ln)
        n_out += len(keep)
        with open(os.path.join(outd, os.path.basename(p)), "w") as f:
            f.write("\n".join(keep) + ("\n" if keep else ""))
    for sub in ("steps", "traj"):
        dst = os.path.join(outd, sub)
        if not os.path.exists(dst):
            os.symlink(os.path.join(os.path.abspath(orig), sub), dst)
    print(f"{orig} -> {outd}: {n_in} rows, {n_out} kept (A->B switch episodes removed); steps/ and traj/ linked")
    return 0


def data(dirs, mn=50000):
    ok = True
    for d in dirs:
        m = np.load(os.path.join(d, "meta.npz"))
        N = int(m["N"])
        eps = json.load(open(os.path.join(d, "episodes.json")))
        RES = 224
        nimg = os.path.getsize(os.path.join(d, "img.u8")) // (RES * RES * 3)
        cat = m["cat"]
        c = {int(k): int((cat == k).sum()) for k in np.unique(cat)}
        good = N > mn and nimg == N and set(c) == {0, 1, 2, 3} and sum(e["T"] for e in eps) == N
        nab = sum("_spAB_" in e["id"] and e.get("ts") is not None and e["ts"] >= 0 for e in eps)
        print(f"{d}: {N} frames from {len(eps)} episodes (A->B switch {nab}); frames per category {c} | {'ok' if good else 'NOT OK'}")
        ok &= good
    print("DATA", "OK" if ok else "FAILED")
    return 0 if ok else 1


def count(dirs):
    for d in dirs:
        rs = list(rows_dir(d).values())
        c = collections.Counter(kind(r) for r in rs)
        s = collections.Counter(kind(r) for r in rs if r.get("outcome") == "success")
        print(f"{d}: {len(rs)} rows; " + ", ".join(f"{k} {c[k]} ({s[k]} successes)" for k in sorted(c)))
    return 0


def runs_(runs, specs):
    ok = True
    for s in specs:
        name, n, mn = (s.split(":") + [""])[:3]
        n = int(n)
        mn = int(mn) if mn else n
        r = rows(runs, name)
        e = sum(bool(x.get("error")) for x in r.values())
        v = len(r) - e
        oc = collections.Counter(x.get("outcome") for x in r.values() if not x.get("error"))
        good = v >= mn and len(r) <= n
        print(f"{name}: {len(r)}/{n} rows, {e} errors, {v} valid (needed {mn}), {dict(oc)}{'' if good else '  <-- INCOMPLETE'}")
        ok &= good
    return 0 if ok else 1


def clean(runs, names):
    for name in names:
        moved = 0
        for f in sorted(glob.glob(os.path.join(runs, f"{BB}_{name}_p*", "results_*.jsonl"))):
            keep, bad = [], []
            for ln in open(f):
                try:
                    r = json.loads(ln)
                except Exception:  # noqa: BLE001
                    bad.append(ln)
                    continue
                (bad if r.get("error") else keep).append(ln)
            if bad:
                with open(f.replace("results_", "errors_"), "a") as fe:
                    fe.writelines(bad)
                with open(f, "w") as fo:
                    fo.writelines(keep)
                moved += len(bad)
        print(f"{name}: {moved} error rows moved to errors_*.jsonl")
    return 0


def main():
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "runs":
        return runs_(rest[0], rest[1:])
    if cmd == "clean":
        return clean(rest[0], rest[1:])
    if cmd == "smoke":
        return smoke(rest[0], rest[1])
    if cmd == "repro":
        return repro(rest[0], rest[1], rest[2:])
    if cmd == "pilot":
        k = rest.index("--")
        return pilot(rest[0], rest[1], rest[2:k], rest[k + 1:])
    if cmd == "filter":
        return filt(rest[0], rest[1])
    if cmd == "count":
        return count(rest)
    if cmd == "data":
        mn = 50000
        if "--min" in rest:
            k = rest.index("--min")
            mn = int(rest[k + 1])
            rest = rest[:k] + rest[k + 2:]
        return data(rest, mn)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
