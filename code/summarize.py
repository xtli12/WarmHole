"""Local summaries of downloaded experiment results (results_*.jsonl + traj/*.npz)."""
import collections
import glob
import json
import math
import os
import sys

import numpy as np


def load_results(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "results_*.jsonl"))):
        for ln in open(p):
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def lift_time(traj, key="old_obj", dz=0.02):
    z = traj[key][:, 2]
    if np.all(np.isnan(z)):
        return None
    hit = np.where(z > z[0] + dz)[0]
    return int(hit[0]) if len(hit) else None


def switch_metrics(traj, ts, H=20):
    """Discontinuity at the switch and transition jerk over H steps (6 pose dims of executed actions)."""
    a = traj["act"][:, :6].astype(np.float64)
    if ts is None or ts + 1 >= len(a):
        return float("nan"), float("nan")
    disc = float(np.linalg.norm(a[ts] - a[ts - 1])) if ts >= 1 else float("nan")
    seg = a[ts:ts + H]
    if len(seg) < 3:
        return disc, float("nan")
    jerk = float(np.mean(np.linalg.norm(seg[2:] - 2 * seg[1:-1] + seg[:-2], axis=1)))
    return disc, jerk


def escape_time(traj, ts, window=5, gain=0.02):
    """Steps from ts until the end-effector heads to the new target: the distance to the new
    target has dropped by `gain` below its value at ts and keeps shrinking over `window` steps."""
    ee, tgt = traj["ee"], traj["new_obj"]
    if ts is None or ts >= len(ee) or np.all(np.isnan(tgt)):
        return None
    d = np.linalg.norm(ee - tgt, axis=1)
    d0 = d[ts]
    for t in range(ts, len(d) - window):
        if d[t] < d0 - gain and np.all(np.diff(d[t:t + window]) < 0.002):
            return t - ts
    return None


if __name__ == "__main__":
    for d in sys.argv[1:]:
        rows = load_results(d)
        c = collections.Counter(r.get("outcome", r.get("invalid")) for r in rows)
        print(d, len(rows), dict(c))
