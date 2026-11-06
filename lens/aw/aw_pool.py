"""Test split of round e (handover 7.11): pooled paired comparison over seeds (A1 with B1, A2 with B2 on the same
episodes: scene, direction, initial state, t_s, seed), all switches t_s >= 12 (n = 240 per tag) and the paper's
segments committed (t_s 12-42) / late (52, 62); successes, discordant pairs, exact McNemar, and the 95% interval of the
paired difference (Newcombe 1998, method 10).  Test runs: pi05_awt<TAG>_p*.
c2 (11-01, handover 7.12): any variant counts (flush, or the guided cig05 / cig10 of round f; one per directory).
usage: aw_pool.py A1+A2:B1+B2 [...] [--runs DIR] [--name awt]
c3 (11-02): the backbone prefix of the run directories comes from the environment variable BBP (pi05 | pi0, default pi05)."""
import argparse
import glob
import json
import math
import os


def wilson(k, n, z=1.959964):
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


def newcombe(a, b, c, d):
    """95% interval of p1 - p2 for paired proportions (a both, b only 1, c only 2, d neither)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    num = a * d - b * c
    if num > 0:                       # Newcombe's correction: AD - BC is replaced by max(AD - BC - n/2, 0)
        num = max(num - n / 2.0, 0.0)
    phi = num / math.sqrt(den) if den > 0 else 0.0
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return p1 - p2 - dl, p1 - p2 + du


def load(runs, name):
    rows = {}
    for f in glob.glob(os.path.join(runs, f"{os.environ.get('BBP', 'pi05')}_{name}_p*", "results_*.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            rows[r["id"]] = r
    out = {}
    for r in rows.values():
        if r.get("error") or r.get("invalid") or r.get("outcome") is None:      # c2: any variant (one per directory)
            continue
        if r.get("ts") is None or r["ts"] < 12:
            continue
        out[(r.get("scene"), r.get("pair"), r.get("init"), r.get("ts"), r.get("seed"))] = r["outcome"] == "success"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--runs", default=(os.environ.get("BH", "/data/blackhole") + "/runs/lens"))
    ap.add_argument("--name", default="awt")
    a = ap.parse_args()
    for spec in a.specs:
        xs, ys = spec.split(":")
        xs, ys = xs.split("+"), ys.split("+")
        seg = {"all t_s>=12": lambda k: True, "committed 12-42": lambda k: k[3] <= 42, "late 52/62": lambda k: k[3] >= 52,
               "A->B": lambda k: k[1] == "A->B", "B->A": lambda k: k[1] == "B->A"}
        print(f"== {'+'.join(xs)} vs {'+'.join(ys)} ({a.name}; pairs within seeds)")
        per = []
        for x, y in zip(xs, ys):
            rx, ry = load(a.runs, a.name + x), load(a.runs, a.name + y)
            com = sorted(set(rx) & set(ry), key=str)
            per.append((x, y, rx, ry, com))
            b = sum(rx[k] and not ry[k] for k in com)
            c = sum(ry[k] and not rx[k] for k in com)
            print(f"  {x} vs {y}: {len(com)} pairs ({len(rx)} / {len(ry)}); {sum(rx[k] for k in com)} vs "
                  f"{sum(ry[k] for k in com)}; only {x} {b}, only {y} {c}; p = {mcnemar(b, c):.3g}")
        for name, f in seg.items():
            A = B = C = D = 0
            for x, y, rx, ry, com in per:
                for k in com:
                    if not f(k):
                        continue
                    u, v = rx[k], ry[k]
                    A += u and v
                    B += u and not v
                    C += v and not u
                    D += (not u) and (not v)
            n = A + B + C + D
            if n == 0:
                continue
            lo, hi = newcombe(A, B, C, D)
            print(f"  pooled {name:16s} n {n}: {A + B} ({100 * (A + B) / n:.1f}%) vs {A + C} ({100 * (A + C) / n:.1f}%); "
                  f"only A {B}, only B {C}; McNemar p = {mcnemar(B, C):.3g}; diff {100 * (B - C) / n:+.1f} "
                  f"[{100 * lo:+.1f}, {100 * hi:+.1f}]")


if __name__ == "__main__":
    main()
