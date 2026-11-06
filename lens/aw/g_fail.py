"""g_fail.py -- CPU only, read-only (11-01, round g follow-up).  Why does the teacher so rarely complete the B->A
take-overs from the students' states (round g: 10-18% of take-overs, 0 of the freeze-triggered ones), when it completes
92% of the switches it takes over at the switch itself?  For every take-over episode of the main branch (pi05_g<student>)
and its counterfactual branch (pi05_g<student>cf), from the step records: when the teacher script hands back to the
teacher policy (src >= 0), how close the end effector came to the hover point / the new bowl, and where the arm went
afterwards: toward the new bowl (and was the bowl lifted, placed?), back toward the old bowl, or nowhere.
usage: g_fail.py RUNS_DIR TAG [TAG ...]"""
import collections
import glob
import json
import os
import sys

import numpy as np


def rows_of(runs, tag):
    rows = {}
    for f in glob.glob(os.path.join(runs, f"pi05_{tag}_p*", "results_*.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            r["_dir"] = os.path.dirname(f)
            rows[r["id"]] = r
    return rows


def main():
    runs = sys.argv[1]
    first = True
    for tag in sys.argv[2:]:
        rows = [r for r in rows_of(runs, tag).values() if r.get("outcome") not in (None, "skip") and not r.get("error")]
        cnt = collections.Counter()
        tab = collections.defaultdict(list)
        for r in rows:
            f = os.path.join(r["_dir"], "steps", r["id"] + ".npz")
            if not os.path.exists(f):
                cnt["no step record"] += 1
                continue
            z = np.load(f, allow_pickle=True)
            if first:
                print("step-record keys:", sorted(z.files))
                first = False
            if "src" not in z.files:
                cnt["no src"] += 1
                continue
            src = np.asarray(z["src"]).astype(int)
            ee = np.asarray(z["ee"], np.float64)
            nb = np.asarray(z["new_obj"], np.float64)
            ob = np.asarray(z["old_obj"], np.float64)
            T = len(src)
            pol = np.where(src >= 0)[0]
            scr = np.where(src == -2)[0]
            stu = np.where(src <= -3)[0]
            tb = int(pol[0]) if len(pol) else None          # first teacher-policy step (hand-back)
            t0 = int(scr[0]) if len(scr) else (tb if tb is not None else None)
            d_nb = np.linalg.norm(ee[:, :2] - nb[:, :2], axis=1)
            d_ob = np.linalg.norm(ee[:, :2] - ob[:, :2], axis=1)
            lift_nb = float(nb[:, 2].max() - nb[0, 2])
            lift_ob = float(ob[:, 2].max() - ob[0, 2])
            key = r["outcome"] + ("/" + r.get("dg_reason", "") if r.get("dg_reason") else "")
            if tb is None:
                cnt[key + ": never handed back"] += 1
                tab[key + " script steps (no hand-back)"].append(len(scr))
                continue
            after = slice(tb, T)
            moved_new = float(d_nb[tb] - d_nb[after].min())
            moved_old = float(d_ob[tb] - d_ob[after].min())
            dest = "toward new bowl" if d_nb[-1] < d_ob[-1] else "toward old bowl"
            cnt[key + ": " + dest] += 1
            tab[key + " script steps"].append(len(scr))
            tab[key + " student steps"].append(len(stu))
            tab[key + " hand-back: dxy new bowl (cm)"].append(100 * float(d_nb[tb]))
            tab[key + " hand-back: z above new bowl (cm)"].append(100 * float(ee[tb, 2] - nb[tb, 2]))
            tab[key + " after: min dxy new bowl (cm)"].append(100 * float(d_nb[after].min()))
            tab[key + " after: min dxy old bowl (cm)"].append(100 * float(d_ob[after].min()))
            tab[key + " new bowl lifted (cm)"].append(100 * lift_nb)
            tab[key + " old bowl lifted (cm)"].append(100 * lift_ob)
            tab[key + " steps after hand-back"].append(T - tb)
        print(f"== {tag}: {len(rows)} take-over episodes")
        for k, v in sorted(cnt.items()):
            print(f"   {k}: {v}")
        for k in sorted(tab):
            v = np.asarray(tab[k], np.float64)
            print(f"   {k:55s} n={len(v):4d} median {np.median(v):7.1f}  q25 {np.percentile(v, 25):7.1f}  q75 {np.percentile(v, 75):7.1f}")


if __name__ == "__main__":
    main()
