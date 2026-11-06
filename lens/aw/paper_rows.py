"""Paper-table row (Early / Committed [Wilson] / Capture / Freeze / Completion / Late) for test runs <BBP>_awt<TAG>_p*,
pooled over the tags given with '+'.  usage: BBP=pi05 paper_rows.py RUNS TAG1[+TAG2...] [...]"""
import glob, json, math, os, statistics, sys
BB = os.environ.get("BBP", "pi05")
R = sys.argv[1]
def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z*z/n; c = (p + z*z/(2*n))/den; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return 100*max(0, c-h), 100*min(1, c+h)
def rows(tag):
    out = {}
    for f in glob.glob(os.path.join(R, f"{BB}_awt{tag}_p*", "results_*.jsonl")):
        for ln in open(f):
            r = json.loads(ln)
            if r.get("error") or r.get("outcome") is None: continue
            out[(tag, r["id"])] = r
    return list(out.values())
for spec in sys.argv[2:]:
    rs = [r for t in spec.split("+") for r in rows(t)]
    early = [r for r in rs if r["ts"] == 2]; comm = [r for r in rs if 12 <= r["ts"] <= 42]; late = [r for r in rs if r["ts"] >= 52]
    e = 100*sum(r["outcome"]=="success" for r in early)/len(early)
    kc = sum(r["outcome"]=="success" for r in comm); lo, hi = wilson(kc, len(comm))
    cap = 100*sum(r["outcome"]=="capture" for r in comm)/len(comm); frz = 100*sum(r["outcome"]=="timeout" for r in comm)/len(comm)
    comp = statistics.median(r["steps"]-r["ts"] for r in comm if r["outcome"]=="success")
    l = 100*sum(r["outcome"]=="success" for r in late)/len(late)
    print(f"{spec}: early {e:.1f} ({len(early)}) | committed {100*kc/len(comm):.1f} [{lo:.1f}, {hi:.1f}] ({kc}/{len(comm)}) | capture {cap:.1f} | freeze {frz:.1f} | completion {comp:.0f} | late {l:.1f} ({len(late)})")
