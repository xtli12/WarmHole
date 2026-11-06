"""h0_trap.py -- read-only, LIBERO venv, one EGL device (11-01, after round g).  Is the B->A freeze point a physical
trap?  In round g the students, the scripted teacher (90 steps, never arrived) and the released pi0.5 after the hand-back
all stayed ~9-10 cm (horizontally) from the stove bowl at ~23 cm above it, while the counterfactual branches from the very
same states (back to the bowl on the wooden cabinet) succeeded in 87-96%.
For snapshots stored at round-g take-overs (pi05_g<student>_p*/snap/<id>.npz) the simulator state is restored in its
layout (as bh_client_lens15 does for the counterfactual branches), then:
  (1) joint positions of the arm against their limits (smallest margin and its joint);
  (2) active contacts of robot / gripper bodies with anything else (body names);
  (3) open-loop probes, each from the freshly restored state: 15 control steps of a constant OSC delta action
      (0.5 x unit direction, orientation delta 0, gripper open) toward the new bowl (horizontal), toward the teacher's
      hand-back point (new bowl + vlm offset, 20 cm above), straight up, straight down, toward the old bowl
      (horizontal), and zero; end-effector displacement along the commanded direction (cm) and in total (cm).
Groups: freeze-triggered take-overs (all failed) and k = 40 take-overs that succeeded (contrast).
usage: h0_trap.py RUNS_DIR TAG [TAG ...] [--n_freeze 12] [--n_ok 6]"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, (os.environ.get("BH", "/data/blackhole") + "/lens"))
import bh_client_lens15 as C  # noqa: E402

OFFSET = np.array([-0.017, 0.044], np.float64)     # vlm_offset of the round-g specs (wo_off_h15)


def load_rows(runs, tag):
    rows = {}
    for f in glob.glob(os.path.join(runs, "pi05_%s_p*" % tag, "results_*.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if r.get("snap"):
                rows[r["id"]] = r
    return sorted(rows.values(), key=lambda r: r["id"])


def restore(cache, row):
    env = cache.env(row["suite"], int(row["scene"]))
    init_id, seed = int(row["init"]), int(row.get("seed", 7))
    if init_id >= C.RLAYOUT:
        np.random.seed(init_id - C.RLAYOUT)
        env.seed(init_id - C.RLAYOUT)
        env.reset()
        np.random.seed(seed)
    else:
        np.random.seed(seed)
        env.seed(seed)
        env.reset()
        env.set_init_state(cache.init_states(row["suite"], int(row["scene"]))[init_id])
    zs = np.load(row["snap"])
    obs = env.set_init_state(zs["state"])
    return env, obs, zs


def joints(env):
    sim = env.env.sim
    out = []
    for j in range(1, 8):
        jid = sim.model.joint_name2id("robot0_joint%d" % j)
        q = float(sim.data.qpos[sim.model.jnt_qposadr[jid]])
        lo, hi = (float(x) for x in sim.model.jnt_range[jid])
        out.append((j, q, lo, hi, min(q - lo, hi - q)))
    return out


def contacts(env):
    sim = env.env.sim
    pairs = set()
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        b1 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom1]) or "?"
        b2 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom2]) or "?"
        r1, r2 = b1.startswith(("robot0", "gripper0")), b2.startswith(("robot0", "gripper0"))
        if (r1 or r2) and not (r1 and r2):
            pairs.add("%s|%s" % (b1, b2) if r1 else "%s|%s" % (b2, b1))
    return sorted(pairs)


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--n_freeze", type=int, default=12)
    ap.add_argument("--n_ok", type=int, default=6)
    ap.add_argument("--steps", type=int, default=15)
    a = ap.parse_args()
    cache = C.SceneCache()
    for tag in a.tags:
        rows = load_rows(a.runs, tag)
        groups = (("freeze-triggered (failed)", [r for r in rows if r.get("dg_reason") == "freeze"][:a.n_freeze]),
                  ("k = 40, succeeded", [r for r in rows if r.get("dg_reason") == "k" and r.get("dg_k") == 40
                                         and r.get("outcome") == "success"][:a.n_ok]))
        for label, sel in groups:
            st = collections.defaultdict(list)
            cnt = collections.Counter()
            for r in sel:
                env, obs, zs = restore(cache, r)
                ee = np.asarray(obs["robot0_eef_pos"], np.float64)
                nb, ob = zs["new_obj"].astype(np.float64), zs["old_obj"].astype(np.float64)
                st["restore error (cm)"].append(100 * float(np.linalg.norm(ee - zs["ee"].astype(np.float64))))
                st["dxy new bowl (cm)"].append(100 * float(np.linalg.norm(ee[:2] - nb[:2])))
                st["z above new bowl (cm)"].append(100 * float(ee[2] - nb[2]))
                js = joints(env)
                jm = min(js, key=lambda x: x[4])
                st["smallest joint margin (rad)"].append(jm[4])
                cnt["closest joint %d" % jm[0]] += 1
                for p in contacts(env):
                    cnt["contact " + p] += 1
                hb = np.array([nb[0] + OFFSET[0], nb[1] + OFFSET[1], nb[2] + 0.20])
                dirs = (("toward new bowl (horiz.)", unit(np.r_[nb[:2] - ee[:2], 0.0])),
                        ("toward hand-back point", unit(hb - ee)),
                        ("up", np.array([0.0, 0.0, 1.0])),
                        ("down", np.array([0.0, 0.0, -1.0])),
                        ("toward old bowl (horiz.)", unit(np.r_[ob[:2] - ee[:2], 0.0])),
                        ("zero action", np.zeros(3)))
                for name, d in dirs:
                    env, obs, _ = restore(cache, r)
                    p0 = np.asarray(obs["robot0_eef_pos"], np.float64)
                    act = [0.0] * 6 + [-1.0]
                    act[0], act[1], act[2] = (float(x) for x in 0.5 * d)
                    for _ in range(a.steps):
                        obs, _, _, _ = env.step(act)
                    dp = np.asarray(obs["robot0_eef_pos"], np.float64) - p0
                    st["probe %-26s along (cm)" % name].append(100 * float(dp @ d) if np.any(d) else float("nan"))
                    st["probe %-26s total (cm)" % name].append(100 * float(np.linalg.norm(dp)))
                    if name == "toward new bowl (horiz.)":
                        js2 = joints(env)
                        jm2 = min(js2, key=lambda x: x[4])
                        cnt["after the new-bowl probe: closest joint %d" % jm2[0]] += 1
                        st["after the new-bowl probe: smallest joint margin (rad)"].append(jm2[4])
                        for p in contacts(env):
                            cnt["after the new-bowl probe: contact " + p] += 1
            print("== %s, %s: %d snapshots" % (tag, label, len(sel)), flush=True)
            for k in sorted(st):
                v = np.asarray([x for x in st[k] if x == x], np.float64)
                if len(v):
                    print("   %-62s median %7.2f  q25 %7.2f  q75 %7.2f" % (k, np.median(v), np.percentile(v, 25), np.percentile(v, 75)))
            for k, v in sorted(cnt.items()):
                print("   %-62s %d / %d" % (k, v, len(sel)))


if __name__ == "__main__":
    main()
