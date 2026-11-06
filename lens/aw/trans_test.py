"""trans_test.py -- read-only, LIBERO venv, one EGL device (11-01 22:30, handover 7.15 round i).  Why is the teacher's
B->A transit so slow from the students' far-out switch states (round-i smoke: reach 73-77 cm at the switch, about
0.2 cm per step back toward the base, 90 steps without arriving), and is there a scripted transit that is fast?

For recorded round-i episodes (pi05_<run>_p*/traj/<id>.npz, B->A) the switch state is rebuilt exactly: the same reset
as bh_client_lens16 (layout, seed, settle steps) and then the executed actions up to t_s (replay error printed; MuJoCo
is deterministic).  From the rebuilt state (restored for every variant) the scripted transits below run for at most
150 steps (no policy):
  V0 round-i teacher: Tunnel to the hand-back point, corridor H = 0.36 m, orientation driven to the start orientation
  V1 as V0 but the orientation is held (no rotation command)
  V2 as V0 but without the corridor (lift 5 cm, clear 3 cm: the original teacher wo_off_h15)
  V3 retract first: straight back toward the start position at the current height (orientation held), then V1
  V4 as V0 with the action rate limiter off
Reported per group (far: reach >= 72 cm at the switch; near: < 66 cm) and variant: arrived within 90 / 150 steps
(< 2 cm), median steps to arrive, median reach after 30 / 60 steps, the arm's smallest joint margin at the switch and its
joint, and the angle between the end-effector orientation at the switch and the start orientation.
usage: trans_test.py RUNS NAME [NAME ...] [--n_far 10] [--n_near 4]"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, (os.environ.get("BH", "/data/blackhole") + "/lens"))
import bh_client_lens16 as C  # noqa: E402
from robosuite.utils import transform_utils as T  # noqa: E402

BASE = np.array([-0.66, 0.0])


def rows_of(runs, names):
    out = []
    for n in names:
        for d in sorted(glob.glob(os.path.join(runs, "pi05_%s_p*" % n))):
            for p in sorted(glob.glob(os.path.join(d, "results_*.jsonl"))):
                for ln in open(p):
                    try:
                        r = json.loads(ln)
                    except Exception:  # noqa: BLE001
                        continue
                    if r.get("error") or r.get("pair") != "B->A" or r.get("outcome") is None:
                        continue
                    f = os.path.join(d, "traj", r["id"] + ".npz")
                    if os.path.exists(f):
                        r["_traj"] = f
                        out.append(r)
    return out


def reset(cache, r):
    env = cache.env(r["suite"], int(r["scene"]))
    init_id, seed = int(r["init"]), int(r.get("seed", 7))
    if init_id >= C.RLAYOUT:
        np.random.seed(init_id - C.RLAYOUT)
        env.seed(init_id - C.RLAYOUT)
        obs = env.reset()
        np.random.seed(seed)
    else:
        np.random.seed(seed)
        env.seed(seed)
        env.reset()
        obs = env.set_init_state(cache.init_states(r["suite"], int(r["scene"]))[init_id])
    for _ in range(C.SETTLE):
        obs, _, _, _ = env.step(C.DUMMY)
    return env, obs


def joints(env):
    sim = env.env.sim
    out = []
    for j in range(1, 8):
        jid = sim.model.joint_name2id("robot0_joint%d" % j)
        q = float(sim.data.qpos[sim.model.jnt_qposadr[jid]])
        lo, hi = (float(x) for x in sim.model.jnt_range[jid])
        out.append((j, q, min(q - lo, hi - q)))
    return out


def quat_angle(q1, q2):
    d = T.quat_multiply(np.asarray(q1, np.float64), T.quat_inverse(np.asarray(q2, np.float64)))
    return float(np.degrees(np.linalg.norm(T.quat2axisangle(d))))


class Tun(C.Tunnel):
    """C.Tunnel with optional orientation hold and an optional retract-first waypoint."""

    def __init__(self, *a, hold=False, retract=None, **k):
        super().__init__(*a, **k)
        self.hold, self.retract, self.rphase = hold, retract, retract is not None

    def act(self, p, q):
        if self.rphase:
            p = np.asarray(p, np.float64)
            w = np.array([self.retract[0], self.retract[1], p[2]])
            if np.linalg.norm(p[:2] - w[:2]) < 0.03:
                self.rphase = False
            else:
                dp = np.clip((w - p) / 0.05, -1.0, 1.0) * self.vmax
                step = np.concatenate([dp, np.zeros(3)]) - self.f
                nrm = float(np.linalg.norm(step))
                self.f = self.f + (step if nrm <= self.rate else step * (self.rate / nrm))
                self.k += 1
                return np.concatenate([self.f, [-1.0]]).astype(np.float32), self.k >= self.max_steps, float(np.linalg.norm(p - self.goal))
        if self.hold:
            self.qg = np.asarray(q, np.float64)      # no rotation command (and none of the rate budget spent on it)
        return super().act(p, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--n_far", type=int, default=10)
    ap.add_argument("--n_near", type=int, default=4)
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()
    cache = C.SceneCache()
    rows = rows_of(a.runs, a.names)
    far, near = [], []
    for r in rows:
        z = np.load(r["_traj"])
        ts = int(r["ts"])
        reach = float(np.linalg.norm(z["ee"][ts, :2] - BASE))
        (far if reach >= 0.72 else near if reach < 0.66 else []).append((r, reach))
    far, near = far[:a.n_far], near[:a.n_near]
    print("%d B->A episodes; far %d, near %d selected" % (len(rows), len(far), len(near)), flush=True)
    for label, sel in (("far (reach >= 72 cm at the switch)", far), ("near (reach < 66 cm)", near)):
        st = collections.defaultdict(list)
        cnt = collections.Counter()
        for r, reach in sel:
            z = np.load(r["_traj"])
            ts = int(r["ts"])
            env, obs = reset(cache, r)
            home_pos = np.asarray(obs["robot0_eef_pos"], np.float64)
            home_quat = np.asarray(obs["robot0_eef_quat"], np.float64)
            for t in range(ts):
                obs, _, _, _ = env.step(z["act"][t].astype(np.float64).tolist())
            p = np.asarray(obs["robot0_eef_pos"], np.float64)
            st["replay error (mm)"].append(1000 * float(np.linalg.norm(p - z["ee"][ts].astype(np.float64))))
            state = np.asarray(env.env.sim.get_state().flatten())
            js = joints(env)
            jm = min(js, key=lambda x: x[2])
            cnt["closest joint %d" % jm[0]] += 1
            st["smallest joint margin (rad)"].append(jm[2])
            st["q4 (rad)"].append(js[3][1])
            st["orientation vs start (deg)"].append(quat_angle(obs["robot0_eef_quat"], home_quat))
            new_obj = C.key_object(r.get("new_goal") or [["on", "akita_black_bowl_2", "plate_1"]])
            pn = np.asarray(C.obj_pos(env, new_obj), np.float64)
            goal = np.array([pn[0] - 0.017, pn[1] + 0.044, pn[2] + 0.04 + 0.15])
            a_prev = z["act"][ts - 1].astype(np.float64) if ts > 0 else np.zeros(7)
            clear_c = max(0.03, pn[2] + 0.36 - max(p[2], goal[2]))
            variants = (("V0 round-i teacher", dict(clear=clear_c, lift=1.0)),
                        ("V1 orientation held", dict(clear=clear_c, lift=1.0, hold=True)),
                        ("V2 original teacher", dict(clear=0.03, lift=0.05)),
                        ("V3 retract first", dict(clear=clear_c, lift=1.0, hold=True, retract=home_pos[:2])),
                        ("V4 no rate limit", dict(clear=clear_c, lift=1.0, rate=10.0)))
            for name, kw in variants:
                obs = env.set_init_state(state)
                q0 = np.asarray(obs["robot0_eef_pos"], np.float64)
                tun = Tun(goal, home_quat, q0, a_prev, max_steps=a.steps, **kw)
                arrived = None
                r30 = r60 = float("nan")
                for k in range(a.steps):
                    act, done, err = tun.act(obs["robot0_eef_pos"], obs["robot0_eef_quat"])
                    obs, _, _, _ = env.step(act.tolist())
                    pk = np.asarray(obs["robot0_eef_pos"], np.float64)
                    if k == 29:
                        r30 = 100 * float(np.linalg.norm(pk[:2] - BASE))
                    if k == 59:
                        r60 = 100 * float(np.linalg.norm(pk[:2] - BASE))
                    if arrived is None and float(np.linalg.norm(pk - goal)) < 0.02:
                        arrived = k + 1
                        break
                st[name + ": arrived within 90"].append(float(arrived is not None and arrived <= 90))
                st[name + ": arrived within %d" % a.steps].append(float(arrived is not None))
                if arrived is not None:
                    st[name + ": steps to arrive"].append(arrived)
                st[name + ": reach after 30 steps (cm)"].append(r30)
                st[name + ": reach after 60 steps (cm)"].append(r60)
            st["reach at the switch (cm)"].append(100 * reach)
        print("== %s: %d episodes" % (label, len(sel)), flush=True)
        for k in sorted(st):
            v = np.asarray([x for x in st[k] if x == x], np.float64)
            if len(v):
                if "arrived within" in k:
                    print("   %-52s %d / %d" % (k, int(v.sum()), len(v)))
                else:
                    print("   %-52s median %7.2f  q25 %7.2f  q75 %7.2f  (n %d)" % (k, np.median(v), np.percentile(v, 25),
                                                                                    np.percentile(v, 75), len(v)))
        for k, v in sorted(cnt.items()):
            print("   %-52s %d / %d" % (k, v, len(sel)))


if __name__ == "__main__":
    main()
