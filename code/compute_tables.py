"""Regime-level tables (success / capture / escape time / discontinuity / jerk) from switch runs.

usage: compute_tables.py RUN_DIR OUT_CSV early_ts_csv committed_ts_csv [group_key]
"""
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from summarize import escape_time, load_results, switch_metrics, wilson  # noqa: E402


def main(run_dir, out_csv, early, committed, key="variant"):
    early = {int(x) for x in early.split(",")}
    committed = {int(x) for x in committed.split(",")}
    rows = load_results(run_dir)
    groups = collections.defaultdict(list)
    for r in rows:
        if "outcome" not in r or r["outcome"] == "done_before_switch" or not r.get("switched", True):
            continue
        reg = "early" if r["ts"] in early else ("committed" if r["ts"] in committed else None)
        if reg is None:
            continue
        groups[(r.get(key) or r.get("method"), reg)].append(r)
    out = []
    for (name, reg), rs in sorted(groups.items()):
        succ = [r["outcome"] == "success" for r in rs]
        capt = [r["outcome"] == "capture" for r in rs]
        tesc, disc, jerk = [], [], []
        for r in rs:
            f = os.path.join(run_dir, "traj", r["id"] + ".npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            d, j = switch_metrics(z, r["ts"])
            disc.append(d)
            jerk.append(j)
            if r["outcome"] == "success":
                te = escape_time(z, r["ts"])
                if te is not None:
                    tesc.append(te)
        p, lo, hi = wilson(sum(succ), len(succ))
        out.append(dict(method=name, regime=reg, n=len(rs), success=round(100 * p, 1), lo=round(100 * lo, 1),
                        hi=round(100 * hi, 1), capture=round(100 * np.mean(capt), 1),
                        t_esc=round(float(np.median(tesc)), 1) if tesc else "",
                        disc=round(float(np.nanmean(disc)), 3) if disc else "",
                        jerk=round(float(np.nanmean(jerk)), 3) if jerk else ""))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(o)


if __name__ == "__main__":
    main(*sys.argv[1:])
