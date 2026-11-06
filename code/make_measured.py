"""Build the data_measured/*.csv tables of the paper directly from simulator run directories.

usage: make_measured.py BACKBONE OUT_DIR E0_DIR SWITCH_DIR [SWITCH_DIR ...]
  E0_DIR      no-switch reference runs (tag e0/lib)          -> {bb}_noswitch_timing.csv, {bb}_noswitch_outcomes.json
  SWITCH_DIRs switching runs; rows are split by tag/suite:
     tag e1, libero_spatial            -> {bb}_spatial_sweep.csv          (group = A->B / B->A)
     tag e1, libero_goal/libero_object -> {bb}_goal_object_sweep.csv      (group = goal:i->j / object:i->j)
     tag e2/e3 (methods, ablations)    -> {bb}_spatial_methods.csv        (group = variant|pair)
Counts only; every number is an episode outcome read from results_*.jsonl."""
import collections
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import lift_time, load_results, wilson  # noqa: E402

FIELDS = ["backbone", "group", "ts", "n", "success", "capture", "timeout", "done_before", "p_esc", "lo", "hi"]


def close_time(z, width=0.06):
    w = z["grip"][:, 0] - z["grip"][:, 1]
    hit = np.where(w < width)[0]
    return int(hit[0]) if len(hit) else None


def sweep_rows(bb, rows, keyf):
    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        if "outcome" not in r or r.get("error"):  # episodes broken by infrastructure errors are not counted
            continue
        agg[(keyf(r), r["ts"])][r["outcome"]] += 1
    out = []
    for (g, ts), c in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        n = c["success"] + c["capture"] + c["timeout"]  # switched episodes only
        if n == 0:
            continue
        p, lo, hi = wilson(c["success"], n)
        out.append(dict(backbone=bb, group=g, ts=ts, n=n, success=c["success"], capture=c["capture"],
                        timeout=c["timeout"], done_before=c["done_before_switch"],
                        p_esc=round(p, 4), lo=round(lo, 4), hi=round(hi, 4)))
    return out


def write(path, rows, fields=FIELDS):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{path}: {len(rows)} rows")


def main(bb, out_dir, e0_dir, *switch_dirs):
    os.makedirs(out_dir, exist_ok=True)
    # no-switch reference
    e0 = load_results(e0_dir)
    outc = collections.defaultdict(collections.Counter)
    tim = []
    for r in e0:
        key = r.get("pair") or r.get("tag")
        outc[key][r["outcome"]] += 1
        f = os.path.join(e0_dir, "traj", r["id"] + ".npz")
        if os.path.exists(f) and r["outcome"] == "success":
            z = np.load(f)
            tim.append(dict(backbone=bb, id=r["id"], pair=key, close_t=close_time(z), lift_t=lift_time(z), steps=r["steps"]))
    with open(os.path.join(out_dir, f"{bb}_noswitch_outcomes.json"), "w") as f:
        json.dump({k: dict(v) for k, v in sorted(outc.items())}, f, indent=1)
    write(os.path.join(out_dir, f"{bb}_noswitch_timing.csv"), tim, ["backbone", "id", "pair", "close_t", "lift_t", "steps"])
    # switching runs
    rows = []
    for d in switch_dirs:
        rows += load_results(d)
    e1s = [r for r in rows if r.get("tag") == "e1" and r.get("suite") == "libero_spatial"]
    e1g = [r for r in rows if r.get("tag") == "e1" and r.get("suite") in ("libero_goal", "libero_object")]
    e23 = [r for r in rows if r.get("tag") in ("e2", "e3")]
    if e1s:
        write(os.path.join(out_dir, f"{bb}_spatial_sweep.csv"), sweep_rows(bb, e1s, lambda r: r["pair"]))
    if e1g:
        write(os.path.join(out_dir, f"{bb}_goal_object_sweep.csv"), sweep_rows(bb, e1g, lambda r: r["pair"]))
    if e23:
        write(os.path.join(out_dir, f"{bb}_spatial_methods.csv"),
              sweep_rows(bb, e23, lambda r: f"{r.get('variant') or r.get('method')}|{r['pair']}"))


if __name__ == "__main__":
    main(*sys.argv[1:])
