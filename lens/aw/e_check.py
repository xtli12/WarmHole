"""Checks of round e (handover 7.11, switch self-play).  Read-only except `clean`.
usage:
  e_check.py smoke RUNS              smoke rollouts pi05_spsmk (24 self-play: layouts 20 / r101, K = 1) and pi05_awrsmk
                                     (12 devR-type: r521, outside devR and self-play): fields, layouts, step records, layout reproducibility
  e_check.py equiv RUNS NEW OLD      episodes of NEW (bh_client_lens14) and OLD (lens12) with the same (pair, init, t_s):
                                     outcome, steps and end-effector path identical?  (information, not a gate)
  e_check.py val0 LOG LOG [...]      the 'VAL step 0' lines of training logs agree (|difference| <= 2e-4 x max(1, |value|))
  e_check.py data DIR [DIR ...]      self-play datasets (aw_build_data_c5 --self): meta.npz consistent, category 5, fw
  e_check.py runs RUNS NAME:N[:MIN] [...]  results of each run name: rows, errors, outcome counts; passes if every name
                                     has at least MIN (default N) valid rows and no more than N rows
  e_check.py spsum RUNS NAME [...]   self-play results: success by direction / segment / layout type, conditions by p_c
  e_check.py clean RUNS NAME [...]   move error rows of pi05_NAME_p*/results_*.jsonl to errors_*.jsonl (to re-run them)
Exit status 0 = passed."""
import collections
import glob
import json
import os
import re
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


def smoke(runs):
    ok = True
    sp, dr = rows(runs, "spsmk"), rows(runs, "awrsmk")
    err = [r["id"] for r in list(sp.values()) + list(dr.values()) if r.get("error")]
    print(f"self-play smoke {len(sp)}/24, devR smoke {len(dr)}/12, errors {len(err)} {err[:3]}")
    ok &= len(sp) == 24 and len(dr) == 12 and not err
    print("self-play outcomes", dict(collections.Counter(r.get("outcome") for r in sp.values())),
          "| devR outcomes", dict(collections.Counter(r.get("outcome") for r in dr.values())))
    inits = sorted({r.get("init") for r in sp.values()})
    print("self-play layouts", inits, "| devR layouts", sorted({r.get("init") for r in dr.values()}))
    ok &= inits == [20, 10101] and sorted({r.get("init") for r in dr.values()}) == [10521]
    bad = [r["id"] for r in sp.values() if not r.get("grp") or r.get("noise_tag") != "sp1"
           or r.get("grp") != f"{r['id'].split('_sp')[-1].split('_')[0]}_ts{r['ts']}_i{r['init']}"]
    print("self-play rows without the right grp / noise_tag:", len(bad), bad[:3])
    ok &= not bad
    nost = [r["id"] for r in sp.values() if not os.path.exists(os.path.join(r["_dir"], "steps", r["id"] + ".npz"))]
    print("self-play episodes without a step record:", len(nost))
    ok &= not nost
    # the same layout must start identically for every t_s (layout reproducible); official 20 and r101 must differ
    start = collections.defaultdict(list)
    for r in list(sp.values()) + list(dr.values()):
        f = os.path.join(r["_dir"], "traj", r["id"] + ".npz")
        if os.path.exists(f):
            z = np.load(f)
            start[(r.get("pair"), r.get("init"))].append(np.concatenate([z["old_obj"][0], z["new_obj"][0], z["ee"][0]]))
    spread = {k: float(np.max(np.ptp(np.stack(v), axis=0))) for k, v in start.items() if len(v) > 1}
    print("start-state spread over the t_s of one layout (m):", {f"{k[0]} {k[1]}": round(v, 6) for k, v in sorted(spread.items(), key=str)})
    ok &= bool(spread) and max(spread.values()) < 1e-4
    for p in sorted({k[0] for k in start}, key=str):
        a, b = start.get((p, 20)), start.get((p, 10101))
        if a and b:
            d = float(np.linalg.norm(a[0][:6] - b[0][:6]))
            print(f"{p}: bowls of official 20 vs r101 differ by {d:.3f} m")
            ok &= d > 0.01
    # step records: post-switch frames exist and carry the new instruction
    for r in list(sp.values())[:3]:
        z = np.load(os.path.join(r["_dir"], "steps", r["id"] + ".npz"))
        T, ts = len(z["act"]), int(z["ts"])
        print(f"  {r['id']}: {T} steps, t_s {ts}, switched frames {int(z['sw'].sum())}, img {z['img'].shape}, "
              f"outcome {r.get('outcome')}")
        ok &= ts == r["ts"] and int(z["sw"].sum()) == T - ts and z["img"].shape[1:] == (RES, RES, 3)
    print("SMOKE", "OK" if ok else "FAILED")
    return 0 if ok else 1


def equiv(runs, new, old):
    a, b = rows(runs, new), rows(runs, old)
    key = lambda r: (r.get("pair"), r.get("init"), r.get("ts"), r.get("variant"))  # noqa: E731
    bk = {key(r): r for r in b.values()}
    n = same_o = same_s = same_p = 0
    for r in a.values():
        s = bk.get(key(r))
        if s is None:
            continue
        n += 1
        same_o += r.get("outcome") == s.get("outcome")
        same_s += r.get("steps") == s.get("steps")
        fa = os.path.join(r["_dir"], "traj", r["id"] + ".npz")
        fb = os.path.join(s["_dir"], "traj", s["id"] + ".npz")
        if os.path.exists(fa) and os.path.exists(fb):
            ea, eb = np.load(fa)["ee"], np.load(fb)["ee"]
            m = min(len(ea), len(eb))
            d = float(np.abs(ea[:m] - eb[:m]).max()) if m else float("nan")
            same_p += len(ea) == len(eb) and d < 1e-5
            print(f"  {key(r)}: {r.get('outcome')} {r.get('steps')} vs {s.get('outcome')} {s.get('steps')}; "
                  f"max end-effector difference {d:.2e} m")
    print(f"lens14 ({new}) vs lens12 ({old}) on the same official episodes: {n} pairs; outcome identical {same_o}, "
          f"steps identical {same_s}, path identical {same_p}")
    return 0


def val0(logs):
    vals = []
    for f in logs:
        ln = [x for x in open(f, errors="replace") if "VAL step 0 " in x]
        if not ln:
            print(f"{f}: no 'VAL step 0' line")
            return 1
        kv = dict(re.findall(r"(\w+)=([-+0-9.eEnaif]+)", ln[0].split("VAL step 0 ", 1)[1]))
        vals.append(kv)
        print(f"{os.path.basename(f)}: " + " ".join(f"{k}={v}" for k, v in sorted(kv.items())))
    ok = all(set(v) == set(vals[0]) for v in vals) and all(
        abs(float(v[k]) - float(vals[0][k])) <= 2e-4 * max(1.0, abs(float(vals[0][k]))) for v in vals for k in vals[0])
    print("VAL0", "IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


def data(dirs):
    ok = True
    for d in dirs:
        m = np.load(os.path.join(d, "meta.npz"))
        N = int(m["N"])
        eps = json.load(open(os.path.join(d, "episodes.json")))
        cond = json.load(open(os.path.join(d, "conditions.json")))
        nimg = os.path.getsize(os.path.join(d, "img.u8")) // (RES * RES * 3)
        nwr = os.path.getsize(os.path.join(d, "wrist.u8")) // (RES * RES * 3)
        cat, fw, init = m["cat"], m["fw"], m["init"]
        good = (N > 0 and nimg == N and nwr == N and len(cat) == N and len(fw) == N and bool((cat == 5).all())
                and bool((fw > 0).all()) and float(fw.max()) <= 1.1 + 1e-6 and sum(e["n"] for e in eps) == N
                and bool(m["act_mask"].any(1).all())
                and bool(np.all(((init >= 20) & (init <= 37)) | ((init >= 10000) & (init <= 10499)))))
        pc = collections.Counter(round(v["p"], 3) for v in cond.values())
        late = np.asarray([e["ts"] >= 52 for e in eps])
        ba = np.asarray([e.get("pair") == "B->A" for e in eps])
        fr = np.asarray([e["n"] for e in eps], float)
        w = np.asarray([e["w"] for e in eps], float)
        mass = fr * w
        print(f"{d}: {N} frames from {len(eps)} episodes; {len(cond)} conditions, p_c {dict(sorted(pc.items()))}; "
              f"mean fw {float(fw.mean()):.3f}; weight mass B->A {mass[ba].sum() / mass.sum():.2f}, "
              f"late {mass[late].sum() / mass.sum():.2f}, procedural {mass[np.asarray([e['init'] >= 10000 for e in eps])].sum() / mass.sum():.2f}"
              f" | {'ok' if good else 'NOT OK'}")
        ok &= good
    print("DATA", "OK" if ok else "FAILED")
    return 0 if ok else 1


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


def spsum(runs, names):
    for name in names:
        r = [x for x in rows(runs, name).values() if not x.get("error") and x.get("outcome") is not None]
        if not r:
            print(f"{name}: no results")
            continue
        succ = lambda xs: f"{sum(x['outcome'] == 'success' for x in xs)}/{len(xs)}"  # noqa: E731
        parts = []
        for p in ("A->B", "B->A"):
            for seg, f in (("committed", lambda x: x["ts"] < 52), ("late", lambda x: x["ts"] >= 52)):
                for lay, g in (("official", lambda x: x["init"] < 10000), ("procedural", lambda x: x["init"] >= 10000)):
                    xs = [x for x in r if x.get("pair") == p and f(x) and g(x)]
                    parts.append(f"{p} {seg} {lay} {succ(xs)}")
        grp = collections.defaultdict(list)
        for x in r:
            grp[x.get("grp")].append(x["outcome"] == "success")
        pc = collections.Counter(round(sum(v) / len(v), 2) for v in grp.values())
        print(f"{name}: success {succ(r)} | " + "; ".join(parts) + f" | conditions by p_c {dict(sorted(pc.items()))}")
    return 0


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
    if cmd == "smoke":
        return smoke(rest[0])
    if cmd == "equiv":
        return equiv(rest[0], rest[1], rest[2])
    if cmd == "val0":
        return val0(rest)
    if cmd == "data":
        return data(rest)
    if cmd == "runs":
        return runs_(rest[0], rest[1:])
    if cmd == "spsum":
        return spsum(rest[0], rest[1:])
    if cmd == "clean":
        return clean(rest[0], rest[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
