"""Re-computes the self-trained round's tables from downloaded results (no server needed): committed success with Wilson 95%
intervals, by direction and t_s, capture / timeout, no-switch success, and the paired exact McNemar tests.
usage: aw_tables_local.py RUNS_DIR [SPEC ...]   (RUNS_DIR = the unpacked aw_results_*.tgz, holding pi05_<spec>_p*/results_*.jsonl)"""
import collections, glob, json, math, os, sys

R = sys.argv[1]

def wilson(k, n, z=1.96):
    if n == 0: return float('nan'), float('nan')
    p = k / n; den = 1 + z*z/n; c = (p + z*z/(2*n)) / den
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0, c-h), min(1, c+h)

def load(name):
    rows = {}
    for f in glob.glob(os.path.join(R, f"pi05_{name}_p*", "results_*.jsonl")):
        for ln in open(f):
            try: r = json.loads(ln)
            except Exception: continue
            rows[r["id"]] = r
    good = [r for r in rows.values() if not r.get("error") and not r.get("invalid") and r.get("outcome") is not None]
    return good, len(rows) - len(good)

def summ(name):
    rows, bad = load(name)
    out = []
    byv = collections.defaultdict(list)
    for r in rows: byv[r.get("variant")].append(r)
    for v, rs in sorted(byv.items(), key=lambda x: str(x[0])):
        comm = [r for r in rs if r.get("ts") is not None and r["ts"] >= 12]
        if not comm: 
            ns = [r for r in rs if r.get("ts") is None]
            out.append(f"{name:10s} {str(v):10s} no-switch {sum(r['outcome']=='success' for r in ns)}/{len(ns)}")
            continue
        k = sum(r["outcome"] == "success" for r in comm); lo, hi = wilson(k, len(comm))
        oc = collections.Counter(r["outcome"] for r in comm)
        d = {dd: [r for r in comm if r.get("pair") == dd] for dd in ("A->B", "B->A")}
        ts = sorted({r["ts"] for r in rs if r.get("ts") is not None})
        tsl = " ".join(f"{t}:{100*sum(r['outcome']=='success' for r in rs if r.get('ts')==t)/max(1,len([r for r in rs if r.get('ts')==t])):.0f}" for t in ts)
        out.append(f"{name:10s} {str(v):10s} committed {k}/{len(comm)} = {100*k/len(comm):.1f} [{100*lo:.1f}, {100*hi:.1f}] "
                   f"cap {100*oc['capture']/len(comm):.1f} tout {100*oc['timeout']/len(comm):.1f} | "
                   + " ".join(f"{dd} {sum(r['outcome']=='success' for r in x)}/{len(x)}" for dd, x in d.items())
                   + f" | t_s {tsl} | excluded {bad}")
    return out

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k+1)) / 2**n)

def paired(a, b, var="flush"):
    def rows(name):
        g, _ = load(name)
        return {(r.get("scene"), r.get("pair"), r.get("init"), r.get("ts"), r.get("seed")): r["outcome"] == "success"
                for r in g if r.get("variant") == var and r.get("ts") is not None and r["ts"] >= 12}
    ra, rb = rows(a), rows(b)
    com = sorted(set(ra) & set(rb), key=str)
    x = sum(ra[k] and not rb[k] for k in com); y = sum(rb[k] and not ra[k] for k in com)
    s = f"{a} vs {b}: n={len(com)} succ {sum(ra[k] for k in com)} vs {sum(rb[k] for k in com)}; only-a {x} only-b {y} p={mcnemar(x,y):.3g}"
    for d in ("A->B", "B->A"):
        cs = [k for k in com if k[1] == d]
        x = sum(ra[k] and not rb[k] for k in cs); y = sum(rb[k] and not ra[k] for k in cs)
        s += f" | {d}: {x}/{y} p={mcnemar(x,y):.3g}"
    return s

if __name__ == "__main__":
    for n in sys.argv[2:] or ["awd0", "awdaw6k", "awdbc6k", "awdawx2", "awdawa", "awpawx2", "awpawa",
                              "awtaw6k", "awtbc6k", "awtawx2", "awtawa", "awxaw6k", "awxawx2", "awxawa", "awyaw6k", "awybc6k"]:
        for line in summ(n): print(line)
    print()
    for a, b in [("awtawx2", "awtaw6k"), ("awtawx2", "awtbc6k"), ("awtawa", "awtaw6k"), ("awtawa", "awtbc6k"),
                 ("awtawx2", "awtawa"), ("awtaw6k", "awtbc6k"), ("awdawx2", "awdaw6k"), ("awdawx2", "awdbc6k"),
                 ("awdawa", "awdaw6k"), ("awdawa", "awdbc6k"), ("awdawx2", "awdawa")]:
        print(paired(a, b))
