"""Paired comparison of two policies on the same switch episodes (same scene, direction, initial state, t_s and flow-noise
seed): committed switches (t_s >= 12), discordant pairs and the exact two-sided McNemar test.
usage: aw_paired.py NAME_A NAME_B [VARIANT]     (NAME = spec name, e.g. awtaw6k awtbc6k; variant default flush)"""
import math
import sys

import aw_summary as S


def key(r):
    return (r.get("scene"), r.get("pair"), r.get("init"), r.get("ts"), r.get("seed"))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    a, b = sys.argv[1], sys.argv[2]
    var = sys.argv[3] if len(sys.argv) > 3 else "flush"

    def rows(name):
        out = {}
        for r in S.load(name):
            if r.get("error") or r.get("invalid") or r.get("outcome") is None or r.get("variant") != var:
                continue
            if r.get("ts") is None or r["ts"] < 12:
                continue
            out[key(r)] = r["outcome"] == "success"
        return out

    ra, rb = rows(a), rows(b)
    common = sorted(set(ra) & set(rb), key=str)
    ab = sum(ra[k] and not rb[k] for k in common)
    ba = sum(rb[k] and not ra[k] for k in common)
    both = sum(ra[k] and rb[k] for k in common)
    print(f"{a} vs {b} ({var}, committed): {len(common)} paired episodes ({len(ra)} / {len(rb)} available); "
          f"success {sum(ra[k] for k in common)} vs {sum(rb[k] for k in common)}; both {both}; "
          f"only {a} {ab}, only {b} {ba}; exact McNemar p = {mcnemar(ab, ba):.3g}")
    for d in ("A->B", "B->A"):
        cs = [k for k in common if k[1] == d]
        x = sum(ra[k] and not rb[k] for k in cs)
        y = sum(rb[k] and not ra[k] for k in cs)
        print(f"  {d}: {len(cs)} pairs, only {a} {x}, only {b} {y}, p = {mcnemar(x, y):.3g}")


if __name__ == "__main__":
    main()
