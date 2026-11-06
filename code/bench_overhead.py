"""Wall-clock overhead of a Wormhole switch versus an ordinary policy query, on an otherwise idle server.

usage: bench_overhead.py PORT OBS_NPZ LIBRARY_NPZ OUT_JSON [REPEATS]
Measures (i) a plain policy query, (ii) the batched sampling of N=16 old-instruction continuations,
(iii) entry retrieval + OT throat + closed-form bridge on the CPU, and (iv) one learned-head traversal."""
import json
import sys
import time

import numpy as np
from openpi_client import websocket_client_policy as wcp

import bridge_np
import wormhole as W


def main(port, obs_npz, lib_npz, out_json, reps=30, head=None):
    reps = int(reps)
    z = np.load(obs_npz, allow_pickle=True)
    el = {"observation/image": z["img"][3], "observation/wrist_image": z["wrist"][3],
          "observation/state": z["state"][3], "prompt": str(z["prompt_old"])}
    c = wcp.WebsocketClientPolicy("127.0.0.1", int(port))
    res = {}
    for name, extra in (("plain", None), ("multi16", {"mode": "multi", "n": 16, "noise_seed": 1})):
        e = dict(el)
        if extra:
            e["bh"] = extra
        for _ in range(3):
            c.infer(e)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = c.infer(e)
            ts.append(1000 * (time.perf_counter() - t0))
        res[name] = dict(median_ms=float(np.median(ts)), p90_ms=float(np.percentile(ts, 90)))
        if name == "multi16":
            X = np.asarray(out["samples"], dtype=np.float64)[:, :10]
    lib = W.EntryLibrary(lib_npz, 10)
    st = z["state"][3]
    p, g = np.asarray(st[:3], np.float64), float(st[6] - st[7])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        Y, wY, _ = lib.retrieve(p, g, m=32)
        W.bridge_chunk(X, Y, wY, np.zeros(7), eta=0.1, ramp=3)
        ts.append(1000 * (time.perf_counter() - t0))
    res["ot_closed_form_cpu"] = dict(median_ms=float(np.median(ts)), p90_ms=float(np.percentile(ts, 90)))
    if head:
        h = bridge_np.BridgeNP(head)
        ts = []
        for _ in range(reps):
            Y, wY, _ = lib.retrieve(p, g, m=32)
            t0 = time.perf_counter()
            W.bridge_chunk(X, Y, wY, np.zeros(7), eta=0.1, ramp=3, head=h, state=np.r_[p, g])
            ts.append(1000 * (time.perf_counter() - t0))
        res["ot_learned_head_cpu"] = dict(median_ms=float(np.median(ts)), p90_ms=float(np.percentile(ts, 90)))
    res["library_size"] = int(len(lib.states))
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3], a[4] if len(a) > 4 else 30, a[5] if len(a) > 5 else None)
