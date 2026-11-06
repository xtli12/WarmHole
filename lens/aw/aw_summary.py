"""Success of the released and the self-trained policies on the instruction-switch split and on no-switch rollouts.
usage: aw_summary.py NAME [NAME ...]      (NAME = spec name, e.g. awd0, awdaw6k, awpaw6k; runs/lens/pi05_NAME_p*)
Per name and variant: switches at t_s = 2 (early), committed switches (t_s >= 12) with a Wilson 95% interval, by
direction and by t_s; capture / timeout; no-switch success by scene and instruction."""
import os
import collections
import glob
import json
import math
import sys

import numpy as np

BH = os.environ.get("BH", "/data/blackhole")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def load(name):
    """name = spec name (pi0.5) or BB:name (e.g. pi0:awdaw6k)."""
    bb, _, nm = name.rpartition(":")
    bb = bb or "pi05"
    rows = {}
    for f in glob.glob(f"{BH}/runs/lens/{bb}_{nm}_p*/results_*.jsonl"):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            rows[r["id"]] = r
    return list(rows.values())


def main():
    for name in sys.argv[1:]:
        rows = load(name)
        err = [r for r in rows if r.get("error")]
        rows = [r for r in rows if not r.get("error") and not r.get("invalid") and r.get("outcome") is not None]
        by_v = collections.defaultdict(list)
        for r in rows:
            by_v[r.get("variant")].append(r)
        print(f"== {name}: {len(rows)} episodes ({len(err)} aborted by errors, not counted)")
        for v, rs in sorted(by_v.items(), key=lambda x: str(x[0])):
            sw = [r for r in rs if r.get("ts") is not None]
            if sw:
                early = [r for r in sw if r["ts"] == 2]
                comm = [r for r in sw if r["ts"] >= 12]
                k = sum(r["outcome"] == "success" for r in comm)
                lo, hi = wilson(k, len(comm))
                oc = collections.Counter(r["outcome"] for r in comm)
                line = (f"  {str(v):10s} early {100 * np.mean([r['outcome'] == 'success' for r in early]) if early else float('nan'):5.1f} "
                        f"({len(early)}) | committed {100 * k / max(1, len(comm)):5.1f} [{100 * lo:4.1f}, {100 * hi:4.1f}] "
                        f"({len(comm)}) capture {100 * oc['capture'] / max(1, len(comm)):4.1f} timeout "
                        f"{100 * oc['timeout'] / max(1, len(comm)):4.1f} |")
                for d in ("A->B", "B->A"):
                    x = [r for r in comm if r.get("pair") == d]
                    line += f" {d} {100 * np.mean([r['outcome'] == 'success' for r in x]) if x else float('nan'):5.1f} ({len(x)})"
                tss = sorted({r["ts"] for r in sw})
                line += " | t_s " + " ".join(
                    f"{t}:{100 * np.mean([r['outcome'] == 'success' for r in sw if r['ts'] == t]):.0f}" for t in tss)
                print(line)
            ns = [r for r in rs if r.get("ts") is None]
            if ns:
                c = collections.Counter((r.get("pair"), r["outcome"]) for r in ns)
                k = sum(r["outcome"] == "success" for r in ns)
                print(f"  {str(v):10s} no-switch success {100 * k / len(ns):5.1f} ({len(ns)}) {dict(sorted(c.items()))}")


if __name__ == "__main__":
    main()
