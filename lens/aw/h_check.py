"""Round h (handover 7.14, kinematics-aware attractor relabeling): resume specs of the new-instruction branch from the
round-g take-over snapshots, the pilot that selects the corridor height H, summaries and the data-volume rule.
Read-only except `specs` (writes one spec file).
usage:
  h_check.py specs RUNS SPECDIR NAME MODE "STUDENTS" [--only freeze] [--H 0.30] [--rec]
      one resume spec per round-g take-over (RUNS/pi05_g<student>_p*/results_*.jsonl: dg_reason k / freeze, stored
      snapshot) of the given students -> SPECDIR/pi05_NAME.jsonl.  The teacher (wo_off_h15: scripted transit to the
      hand-back point, then the released pi0.5) takes over at once from the stored state under the NEW instruction
      (ts 0, dg_k 0, at most 300 steps, as the counterfactual branches of round g).  Teachers:
        replay    the original transit (lens_lift 0.05, lens_clear 0.03; the round-g teacher)
        corridor  lens_lift 1.0 and lens_clear = max(0.03, z_new + H - max(z_ee, z_goal)): lift straight up to H above
                  the new bowl (at least 3 cm above the start), cross at that height, then go down to the hand-back point
                  (z_goal = z_new + 0.04 + 0.15, unchanged)
      MODE pilot = the three pilot teachers replay / corridor 0.30 / corridor 0.36 for every selected take-over.
  h_check.py pilot RUNS NAME OUT     pilot summary (successes per teacher x student, transit arrival) and the selection
                                     rule of 7.14: the H with more successes, a tie gives 0.30 -> OUT ("H=0.30")
  h_check.py hsum RUNS NAME [...]    round-h runs h<student>: outcomes by take-over reason / k of the round-g row, transit
  h_check.py enough RUNS NAME MIN    at least MIN successful episodes in run NAME (data-volume rule of 7.14)
Exit status 0 = passed."""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

BB = "pi05"
PAIR_BA = dict(suite="libero_spatial", scene=9, old=9, pair="B->A",
               new_lang="pick up the black bowl on the stove and place it on the plate",
               new_goal=[["on", "akita_black_bowl_2", "plate_1"]])
WO15 = dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.15, lens_handover="arrive",
            vlm_offset=[-0.017, 0.044], dg_k=0, max_steps=300)
ZTOP, HOVER = 0.04, 0.15          # oracle_ztop default of bh_client_lens15, lens_hover of wo_off_h15
PILOT = (("rep", None), ("c30", 0.30), ("c36", 0.36))


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


def snap_path(runs, p):
    """The stored path, or the same file under RUNS (local dry runs keep the server's absolute paths in the rows)."""
    if os.path.exists(p):
        return p
    q = os.path.join(runs, p.split("/runs/lens/", 1)[-1])
    return q if os.path.exists(q) else None


def takeovers(runs, student, only=None):
    out = []
    for rid, r in sorted(rows(runs, "g" + student).items()):
        if r.get("error") or r.get("dg_reason") not in ("k", "freeze") or not r.get("snap"):
            continue
        if only and r["dg_reason"] != only:
            continue
        p = snap_path(runs, r["snap"])
        if p is None:
            print(f"  snapshot missing: {r['snap']}")
            continue
        out.append((r, p))
    return out


def teacher(kind, H, z):
    if H is None:
        return dict(lens_lift=0.05, lens_clear=0.03)
    ee, nb = z["ee"].astype(np.float64), z["new_obj"].astype(np.float64)
    zg = float(nb[2]) + ZTOP + HOVER
    return dict(lens_lift=1.0, lens_clear=round(max(0.03, float(nb[2]) + H - max(float(ee[2]), zg)), 4))


def specs(a):
    out = []
    studs = a.students.split()
    arms = PILOT if a.mode == "pilot" else (("rep", None),) if a.mode == "replay" else ((f"c{int(round(a.H * 100))}", a.H),)
    for s in studs:
        tk = takeovers(a.runs, s, a.only)
        print(f"g{s}: {len(tk)} take-overs{' (' + a.only + ')' if a.only else ''}")
        for r, p in tk:
            z = np.load(p)
            tail = r["id"].split(f"_g{s}_", 1)[1]
            for arm, H in arms:
                sp = dict(PAIR_BA)
                sp.update(pair="B->A:h", id=f"{BB}_{a.name}_{s}_{arm}_{tail}" if a.mode == "pilot" else f"{BB}_{a.name}_{tail}",
                          init=int(r["init"]), seed=int(r.get("seed") or 7), ts=0, tag=a.name, variant="h_" + arm,
                          noise_tag=str(r.get("noise_tag") or "") + "h", rec_steps=bool(a.rec), grp=r.get("grp"),
                          resume=r["snap"], branch_of=r["id"])
                sp.update(WO15)
                sp.update(teacher(arm, H, z))
                out.append(sp)
    path = os.path.join(a.specdir, f"{BB}_{a.name}.jsonl")
    with open(path, "w") as f:
        for sp in out:
            f.write(json.dumps(sp) + "\n")
    cl = [sp["lens_clear"] for sp in out if sp["lens_lift"] == 1.0]
    print(path, len(out), f"(corridor lens_clear: median {np.median(cl):.3f}, min {min(cl):.3f}, max {max(cl):.3f})" if cl else "")
    return 0 if out else 1


def student_of(branch_of):
    return branch_of.split("_")[1][1:]          # pi05_g<student>_spBA_... -> <student>


def arrival(r):
    w = (r.get("worm") or [{}])[0]
    return w.get("steps"), w.get("arrive_err")


def pilot(runs, name, out):
    rs = [r for r in rows(runs, name).values() if not r.get("error") and r.get("outcome") is not None]
    print(f"{name}: {len(rs)} valid rows, errors {sum(bool(r.get('error')) for r in rows(runs, name).values())}")
    tab = collections.defaultdict(lambda: collections.Counter())
    arr = collections.defaultdict(list)
    for r in rs:
        v, s = r["variant"], student_of(r["branch_of"])
        tab[(v, s)][r["outcome"]] += 1
        st, er = arrival(r)
        if er is not None:
            arr[v].append((st, er))
    studs = sorted({s for _, s in tab})
    for v in ("h_rep", "h_c30", "h_c36"):
        tot = sum((tab[(v, s)] for s in studs), collections.Counter())
        n = sum(tot.values())
        a = arr[v]
        ok = sum(e < 0.02 for _, e in a)
        print(f"  {v}: success {tot['success']}/{n} | " + ", ".join(f"{s} {tab[(v, s)]['success']}/{sum(tab[(v, s)].values())}" for s in studs)
              + f" | outcomes {dict(tot)} | transit arrived (< 2 cm) {ok}/{len(a)}, steps median "
              f"{np.median([x for x, _ in a]) if a else float('nan'):.0f}, arrival error median {100 * np.median([e for _, e in a]) if a else float('nan'):.1f} cm")
    n30 = sum(tab[("h_c30", s)]["success"] for s in studs)
    n36 = sum(tab[("h_c36", s)]["success"] for s in studs)
    sel = "H=0.30" if n30 >= n36 else "H=0.36"
    print(f"selection (7.14: more successes, tie -> 0.30): corridor 0.30 {n30}, 0.36 {n36} -> {sel}")
    open(out, "w").write(sel + "\n")
    return 0


def hsum(runs, names):
    for name in names:
        s = name[1:]
        g = rows(runs, "g" + s)
        rs = [r for r in rows(runs, name).values() if not r.get("error")]
        by = collections.defaultdict(list)
        for r in rs:
            o = g.get(r["branch_of"], {})
            by[(o.get("dg_reason"), int(o.get("dg_k") or 0))].append(r)
        a = [arrival(r) for r in rs]
        a = [x for x in a if x[1] is not None]
        print(f"{name}: {len(rs)} episodes, errors {sum(bool(r.get('error')) for r in rows(runs, name).values())}; outcomes "
              f"{dict(collections.Counter(r.get('outcome') for r in rs))}; successful {sum(r['outcome'] == 'success' for r in rs)} | "
              + "; ".join(f"{rsn} k={k}: {sum(x['outcome'] == 'success' for x in xs)}/{len(xs)}" for (rsn, k), xs in sorted(by.items(), key=str))
              + f" | transit arrived {sum(e < 0.02 for _, e in a)}/{len(a)}, steps median {np.median([x for x, _ in a]) if a else float('nan'):.0f}")
    return 0


def enough(runs, name, mn):
    rs = [r for r in rows(runs, name).values() if not r.get("error")]
    n = sum(r.get("outcome") == "success" for r in rs)
    print(f"{name}: {n} successful new-instruction branches (needed {mn})")
    return 0 if n >= mn else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "specs":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("runs")
        ap.add_argument("specdir")
        ap.add_argument("name")
        ap.add_argument("mode", choices=("pilot", "replay", "corridor"))
        ap.add_argument("students")
        ap.add_argument("--only", default=None, choices=(None, "freeze", "k"))
        ap.add_argument("--H", type=float, default=None)
        ap.add_argument("--rec", action="store_true")
        a = ap.parse_args()
        if a.mode == "corridor" and a.H is None:
            print("--H is required for MODE corridor")
            return 2
        return specs(a)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "pilot":
        return pilot(rest[0], rest[1], rest[2])
    if cmd == "hsum":
        return hsum(rest[0], rest[1:])
    if cmd == "enough":
        return enough(rest[0], rest[1], int(rest[2]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
