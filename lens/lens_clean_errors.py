"""Move episodes aborted by infrastructure errors (result field "error" set) out of the results files, so that the
clients run them again; they are kept in errors_<worker>.jsonl.  usage: lens_clean_errors.py RUN_DIR [RUN_DIR ...]"""
import glob
import json
import os
import sys

tot = 0
for d in sys.argv[1:]:
    for p in glob.glob(os.path.join(d, "results_*.jsonl")):
        keep, bad = [], []
        for ln in open(p):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            (bad if r.get("error") else keep).append(ln)
        if bad:
            with open(p.replace("results_", "errors_"), "a") as f:
                f.writelines(bad)
            with open(p, "w") as f:
                f.writelines(keep)
            tot += len(bad)
print("moved", tot, "aborted episodes")
