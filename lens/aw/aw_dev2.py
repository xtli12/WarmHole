"""Harder development split "dev2" (fixed 10-30, before any dev2 result): development initial states 40-49, both
directions, t_s 12-62 as in the test split (n = 120 committed switches) = the development runs pi05_awd<TAG> (t_s 12-42)
plus the late-switch extension pi05_awe<TAG> (t_s 52, 62; aw_eval.sh ... dev2ext).
Committed success with Wilson 95% intervals, by direction and t_s, capture / timeout, and exact McNemar tests on the
same episodes (scene, direction, initial state, t_s, seed).
usage: aw_dev2.py [--runs DIR] TAG [TAG ...] [--pairs A:B,C:D]
c2 (11-01): a TAG may carry a backbone prefix, e.g. pi0/awx3p0 (default pi05); pairs use the same spelling."""
import argparse
import collections
import glob
import json
import math
import os


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(runs, name, bb="pi05"):
    rows = {}
    for f in glob.glob(os.path.join(runs, f"{bb}_{name}_p*", "results_*.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            rows[r["id"]] = r
    good = [r for r in rows.values() if not r.get("error") and not r.get("invalid") and r.get("outcome") is not None]
    return good, len(rows) - len(good)


def dev2(runs, tag):
    """Committed (t_s >= 12) flush episodes of awd<TAG> and awe<TAG>, keyed by (scene, pair, init, ts, seed)."""
    out, bad = {}, 0
    bb, _, tag = tag.rpartition("/")
    bb = bb or "pi05"
    for name in (f"awd{tag}", f"awe{tag}"):
        g, b = load(runs, name, bb)
        bad += b
        for r in g:
            if r.get("variant") == "flush" and r.get("ts") is not None and r["ts"] >= 12:
                out[(r.get("scene"), r.get("pair"), r.get("init"), r.get("ts"), r.get("seed"))] = r["outcome"]
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--runs", default=(os.environ.get("BH", "/data/blackhole") + "/runs/lens"))
    ap.add_argument("--pairs", default="", help="comma list A:B of tags to compare on the same episodes")
    a = ap.parse_args()
    res = {}
    for tag in a.tags:
        d, bad = dev2(a.runs, tag)
        res[tag] = d
        n = len(d)
        k = sum(v == "success" for v in d.values())
        lo, hi = wilson(k, n)
        oc = collections.Counter(d.values())
        by_dir = {p: [v for kk, v in d.items() if kk[1] == p] for p in ("A->B", "B->A")}
        by_ts = collections.defaultdict(list)
        for kk, v in d.items():
            by_ts[kk[3]].append(v)
        print(f"dev2 {tag:12s} committed {k}/{n} = {100 * k / max(n, 1):.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
              f"{'' if n == 120 else '  (INCOMPLETE: 120 expected)'} | capture {100 * oc['capture'] / max(n, 1):.1f}"
              f" timeout {100 * oc['timeout'] / max(n, 1):.1f} | "
              + " ".join(f"{p} {sum(v == 'success' for v in x)}/{len(x)}" for p, x in by_dir.items())
              + " | t_s " + " ".join(f"{t}:{100 * sum(v == 'success' for v in x) / len(x):.0f}"
                                      for t, x in sorted(by_ts.items()))
              + f" | excluded (errors) {bad}")
    for pr in [p for p in a.pairs.split(",") if p]:
        x, y = pr.split(":")
        if x not in res or y not in res:
            continue
        com = sorted(set(res[x]) & set(res[y]), key=str)
        sx = {kk: res[x][kk] == "success" for kk in com}
        sy = {kk: res[y][kk] == "success" for kk in com}
        b = sum(sx[kk] and not sy[kk] for kk in com)
        c = sum(sy[kk] and not sx[kk] for kk in com)
        line = (f"dev2 paired {x} vs {y}: {len(com)} pairs; success {sum(sx.values())} vs {sum(sy.values())}; "
                f"only {x} {b}, only {y} {c}; exact McNemar p = {mcnemar(b, c):.3g}")
        for p in ("A->B", "B->A"):
            cs = [kk for kk in com if kk[1] == p]
            b2 = sum(sx[kk] and not sy[kk] for kk in cs)
            c2 = sum(sy[kk] and not sx[kk] for kk in cs)
            line += f" | {p} {b2}/{c2} p = {mcnemar(b2, c2):.3g}"
        print(line)


if __name__ == "__main__":
    main()
