"""Checks DAgger episodes (bh_client_lens13 with --teacher_port): per episode the outcome and the sequence of action
sources in steps/<id>.npz -- before the switch the student (src -4), after it exactly dg_k student steps (src -3), then
the teacher's scripted traversal (src -2) and the teacher policy (src >= 0).
usage: dag_check.py RUN_DIR [RUN_DIR ...] [--spec SPEC.jsonl]"""
import argparse
import collections
import glob
import json
import os

import numpy as np


def runs_of(src):
    """Run-length encoding of the source codes, with the policy queries (>= 0) merged into one code 'P'."""
    out = []
    for s in src:
        c = "P" if s >= 0 else int(s)
        if out and out[-1][0] == c:
            out[-1][1] += 1
        else:
            out.append([c, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--spec", default="")
    a = ap.parse_args()
    specs = {}
    if a.spec:
        for ln in open(a.spec):
            if ln.strip():
                s = json.loads(ln)
                specs[s["id"]] = s
    tot = collections.Counter()
    bad = 0
    for d in a.runs:
        res = {}
        for p in glob.glob(os.path.join(d, "results_*.jsonl")):
            for ln in open(p):
                r = json.loads(ln)
                res[r["id"]] = r
        for eid, r in sorted(res.items()):
            f = os.path.join(d, "steps", eid + ".npz")
            tot[(r.get("variant"), r.get("outcome"))] += 1
            if not os.path.exists(f):
                print(f"{eid}: {r.get('outcome')} error={r.get('error')} (no steps file)")
                continue
            z = np.load(f)
            src, sw = z["src"].astype(int), z["sw"].astype(bool)
            k = specs.get(eid, {}).get("dg_k", r.get("dg_k"))
            rl = runs_of(src)
            ok = True
            if sw.any():
                first = int(np.argmax(sw))
                ok &= bool((src[:first] == -4).all())
                n3 = int((src == -3).sum())
                after = src[first:]
                if (after == -2).any():             # the teacher took over: exactly dg_k student steps came first
                    ok &= bool((after[:n3] == -3).all()) and (k is None or n3 == int(k))
                    ok &= bool((after[n3:] != -3).all() and (after[n3:] != -4).all())
                else:                               # the episode ended while the student had control
                    ok &= k is None or n3 <= int(k)
            bad += not ok
            print(f"{eid}: {r.get('outcome')} steps {len(src)} ts {int(z['ts'])} dg_k {k} | sources {rl}"
                  f" | {'OK' if ok else 'UNEXPECTED'} | worm {[(w.get('t'), w.get('handover')) for w in r.get('worm', [])][:2]}")
    print("outcomes by variant:", dict(sorted(tot.items(), key=str)))
    print("episodes with an unexpected source sequence:", bad)


if __name__ == "__main__":
    main()
