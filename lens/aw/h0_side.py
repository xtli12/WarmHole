"""h0_side.py -- read-only, LIBERO venv (11-01, round h phase 1 diagnostic).  Which limit of joint 4 (the elbow) do
the round-g freeze snapshots touch (h0_trap_c1: smallest margin 0.00 rad at joint 4 in 20 / 20)?  Upper limit
(-0.0698 rad) = elbow stretched out, lower limit (-3.0718 rad) = elbow folded.  For the freeze-triggered take-overs and
the successful k = 40 take-overs (contrast) of each tag: the 7 arm joint positions, the joint-4 range and side, the
distance of the end effector from the robot base (horizontal) and its height above the base.  Lines start with SIDE.
usage: h0_side.py RUNS_DIR TAG [TAG ...] [--n 10]"""
import argparse
import collections

import numpy as np

import h0_trap_c1 as H      # restore(), load_rows(), joints() of the trap diagnostic (same directory)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    cache = H.C.SceneCache()
    for tag in a.tags:
        rows = H.load_rows(a.runs, tag)
        groups = (("freeze", [r for r in rows if r.get("dg_reason") == "freeze"][:a.n]),
                  ("k40 ok", [r for r in rows if r.get("dg_reason") == "k" and r.get("dg_k") == 40
                              and r.get("outcome") == "success"][:a.n]))
        for label, sel in groups:
            side = collections.Counter()
            q4, dist, hgt, qs = [], [], [], []
            for r in sel:
                env, obs, zs = H.restore(cache, r)
                js = H.joints(env)
                j4 = js[3]
                q4.append(j4[1])
                side["upper (stretched)" if (j4[3] - j4[1]) < (j4[1] - j4[2]) else "lower (folded)"] += 1
                qs.append([j[1] for j in js])
                sim = env.env.sim
                try:
                    bid = sim.model.body_name2id("robot0_base")
                except Exception:  # noqa: BLE001
                    bid = sim.model.body_name2id("robot0_link0")
                base = np.asarray(sim.data.body_xpos[bid], np.float64)
                ee = np.asarray(obs["robot0_eef_pos"], np.float64)
                dist.append(100 * float(np.linalg.norm(ee[:2] - base[:2])))
                hgt.append(100 * float(ee[2] - base[2]))
            if not sel:
                print(f"SIDE {tag} {label}: no snapshots", flush=True)
                continue
            lo, hi = js[3][2], js[3][3]
            print(f"SIDE {tag} {label}: n {len(sel)}; joint 4 range [{lo:.4f}, {hi:.4f}]; q4 median {np.median(q4):.4f} "
                  f"(min {min(q4):.4f}, max {max(q4):.4f}); side {dict(side)}", flush=True)
            print(f"SIDE {tag} {label}: q1-q7 median {np.round(np.median(np.asarray(qs), 0), 3).tolist()}; end effector "
                  f"from the base: horizontal {np.median(dist):.1f} cm, height {np.median(hgt):.1f} cm", flush=True)


if __name__ == "__main__":
    main()
