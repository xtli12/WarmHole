"""Summarize results_*.jsonl of an experiment directory."""
import collections, glob, json, sys
for d in sys.argv[1:]:
    rows = []
    for p in glob.glob(f"{d}/results_*.jsonl"):
        for ln in open(p):
            try: rows.append(json.loads(ln))
            except Exception: pass
    c = collections.Counter(r.get("outcome", r.get("invalid", "?")) for r in rows)
    wall = [r["wall_s"] for r in rows if "wall_s" in r]
    inf = [r["infer_ms"] for r in rows if r.get("infer_ms")]
    print(f"{d}: {len(rows)} episodes {dict(c)} mean wall {sum(wall)/max(1,len(wall)):.1f}s infer {sum(inf)/max(1,len(inf)):.0f}ms")
    errs = [r for r in rows if r.get("error")]
    if errs: print("  errors:", len(errs), errs[0].get("error"))
