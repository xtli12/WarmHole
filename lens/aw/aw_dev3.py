"""Development splits of round e (handover 7.11, fixed 11-01 09:30 before any self-play result):
  dev2 = official initial states 40-49, both directions, t_s 12-62 (pi05_awd<TAG> + pi05_awe<TAG>), n = 120;
  devR = procedural layouts (seeds 501-520, bh_client_lens14), both directions, t_s 12-62 (pi05_awr<TAG>), n = 240;
  dev  = dev2 + devR, n = 360.
Committed success (Wilson 95%), by direction; paired comparisons on the same episodes (scene, direction, layout, t_s,
seed): exact McNemar per split.  --pool A1+A2:B1+B2 pools pairs within seeds (A1 with B1, A2 with B2).  --decide
A1+A2:B1+B2 applies the rule of 7.11: GO if the pooled dev (dev2 + devR) successes of A >= those of B on the same
episodes and every tag has its full dev (360); it writes GO / NOGO to --out.
c2 (11-01, handover 7.12): any variant of the switch episodes counts (flush, and the guided variants cig05 / cig10 of
round f; every result directory holds one variant); --select C1+C2,D1+D2,...:B1+B2 --out FILE picks the candidate group
with the most pooled dev (dev2 + devR) successes (ties: the first listed), and writes GO <group> if it has strictly more
than the baseline group B on the same episodes and every tag is complete, NOGO otherwise.
usage: aw_dev3.py TAG [TAG ...] [--pairs A:B,...] [--pool A1+A2:B1+B2,...] [--decide A1+A2:B1+B2 --out FILE]
       [--select C1+C2,D1+D2:B1+B2 --out FILE]"""
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


def split(runs, tag, which):
    """{key: outcome} of the committed (t_s >= 12) flush episodes of a split ('dev2' or 'devr')."""
    out, bad = {}, 0
    names = (f"awd{tag}", f"awe{tag}") if which == "dev2" else (f"awr{tag}",)
    for name in names:
        g, b = load(runs, name)
        bad += b
        for r in g:
            if r.get("ts") is not None and r["ts"] >= 12:                # c2: any variant (one per directory)
                out[(which, r.get("scene"), r.get("pair"), r.get("init"), r.get("ts"), r.get("seed"))] = r["outcome"]
    return out, bad


EXPECT = {"dev2": 120, "devr": 240}


def line(label, d):
    n = len(d)
    k = sum(v == "success" for v in d.values())
    lo, hi = wilson(k, n)
    oc = collections.Counter(d.values())
    by_dir = {p: [v for kk, v in d.items() if kk[2] == p] for p in ("A->B", "B->A")}
    late = [v for kk, v in d.items() if kk[4] >= 52]
    return (f"{label} {k}/{n} = {100 * k / max(n, 1):.1f} [{100 * lo:.1f}, {100 * hi:.1f}] | capture {oc['capture']} "
            f"timeout {oc['timeout']} | " + " ".join(f"{p} {sum(v == 'success' for v in x)}/{len(x)}" for p, x in by_dir.items())
            + f" | late {sum(v == 'success' for v in late)}/{len(late)}")


def compare(ra, rb):
    com = sorted(set(ra) & set(rb), key=str)
    sa = {k: ra[k] == "success" for k in com}
    sb = {k: rb[k] == "success" for k in com}
    b = sum(sa[k] and not sb[k] for k in com)
    c = sum(sb[k] and not sa[k] for k in com)
    return len(com), sum(sa.values()), sum(sb.values()), b, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--runs", default=(os.environ.get("BH", "/data/blackhole") + "/runs/lens"))
    ap.add_argument("--pairs", default="")
    ap.add_argument("--pool", default="")
    ap.add_argument("--decide", default="")
    ap.add_argument("--select", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res, complete = {}, {}
    for tag in a.tags:
        d2, b2 = split(a.runs, tag, "dev2")
        dr, br = split(a.runs, tag, "devr")
        res[tag] = {"dev2": d2, "devr": dr, "dev": {**d2, **dr}}
        complete[tag] = len(d2) == EXPECT["dev2"] and len(dr) == EXPECT["devr"]
        print(line(f"{tag:12s} dev2", d2) + ("" if len(d2) == 120 else "  (INCOMPLETE)") + f" | errors {b2}")
        print(line(f"{tag:12s} devR", dr) + ("" if len(dr) == 240 else "  (INCOMPLETE)") + f" | errors {br}")
        print(line(f"{tag:12s} dev ", res[tag]["dev"]))
    for pr in [p for p in a.pairs.split(",") if p]:
        x, y = pr.split(":")
        if x in res and y in res:
            for s in ("dev2", "devr", "dev"):
                n, kx, ky, b, c = compare(res[x][s], res[y][s])
                print(f"paired {s:4s} {x} vs {y}: {n} pairs; {kx} vs {ky}; only {x} {b}, only {y} {c}; p = {mcnemar(b, c):.3g}")
    for spec in [p for p in (a.pool + "," + a.decide).split(",") if p]:
        xs, ys = spec.split(":")
        xs, ys = xs.split("+"), ys.split("+")
        assert len(xs) == len(ys)
        for s in ("dev2", "devr", "dev"):
            tot = [0, 0, 0, 0, 0]
            for x, y in zip(xs, ys):
                if x in res and y in res:
                    for i, v in enumerate(compare(res[x][s], res[y][s])):
                        tot[i] += v
            n, kx, ky, b, c = tot
            print(f"pooled {s:4s} {'+'.join(xs)} vs {'+'.join(ys)}: {n} pairs; {kx} vs {ky}; only A {b}, only B {c}; "
                  f"p = {mcnemar(b, c):.3g}")
    if a.decide:
        xs, ys = a.decide.split(":")
        xs, ys = xs.split("+"), ys.split("+")
        ok = all(t in res and complete[t] for t in xs + ys)
        kx = sum(compare(res[x]["dev"], res[y]["dev"])[1] for x, y in zip(xs, ys)) if ok else -1
        ky = sum(compare(res[x]["dev"], res[y]["dev"])[2] for x, y in zip(xs, ys)) if ok else -1
        verdict = "GO" if ok and kx >= ky else "NOGO"
        msg = (f"{verdict} pooled dev (dev2 + devR) successes {'+'.join(xs)} {kx} vs {'+'.join(ys)} {ky}"
               + ("" if ok else " (a development split is incomplete)"))
        print("decision:", msg)
        if a.out:
            open(a.out, "w").write(msg + "\n")
    if a.select:
        cands, base = a.select.split(":")
        ys = base.split("+")
        best = None
        for grp in cands.split(","):
            xs = grp.split("+")
            assert len(xs) == len(ys)
            ok = all(t in res and complete[t] for t in xs + ys)
            kx = sum(compare(res[x]["dev"], res[y]["dev"])[1] for x, y in zip(xs, ys)) if ok else -1
            ky = sum(compare(res[x]["dev"], res[y]["dev"])[2] for x, y in zip(xs, ys)) if ok else -1
            print(f"select: {grp} {kx} vs baseline {base} {ky}" + ("" if ok else " (incomplete)"))
            if ok and (best is None or kx > best[1]):
                best = (grp, kx, ky)
        if best is not None and best[1] > best[2]:
            msg = f"GO {best[0]} pooled dev {best[1]} > baseline {base} {best[2]}"
        else:
            msg = "NOGO " + (f"best {best[0]} pooled dev {best[1]} <= baseline {base} {best[2]}" if best else "no complete candidate")
        print("selection:", msg)
        if a.out:
            open(a.out, "w").write(msg + "\n")


if __name__ == "__main__":
    main()
