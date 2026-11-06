"""posture.py -- read-only, LIBERO venv, one EGL device (11-01 22:25, handover 7.15).  Does the student put the arm
into the stretched-elbow configuration (joint 4 at its upper limit) BEFORE the switch, while it still executes the old
task, more often than the teacher (the released pi0.5)?  (223: at the far-out B->A switch states of the round-i smoke
the elbow was already at its limit, q4 = -0.07, and no scripted transit got back in time.)

For B->A switch episodes with t_s >= 32 of the given result sets (traj/<id>.npz with the executed actions), the
pre-switch part is replayed exactly (reset as bh_client_lens16, settle, the executed actions up to t_s; replay error
printed) and joint 4 is read at every step.  Per set x class (S success, T1 never within 6 cm of the new bowl, T2, T3, C;
the class of the recorded episode): the elbow margin to its upper limit at the switch, the share of episodes whose
elbow is within 0.1 rad of the limit at the switch, the share of pre-switch steps within 0.1 rad, the step at which the
limit was first reached, and the reach at the switch.  Sets: run names (pi05_<name>_p*) or directories.
c2 (11-01 22:30): --handback SET ...: for A->B successes of these (teacher) sets, the elbow margin, reach and height at the
hand-back of the scripted transit (is the teacher's own hand-back point above the far cabinet bowl a stretched pose?).
usage: posture.py RUNS SET [SET ...] [--max 30] [--handback SET ...]"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, (os.environ.get("BH", "/data/blackhole") + "/lens"))
import bh_client_lens16 as C  # noqa: E402

BASE = np.array([-0.66, 0.0])


def rows_of(runs, name):
    dirs = [name] if os.path.isdir(name) else sorted(glob.glob(os.path.join(runs, "pi05_%s_p*" % name)))
    out = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "results_*.jsonl"))):
            for ln in open(p):
                try:
                    r = json.loads(ln)
                except Exception:  # noqa: BLE001
                    continue
                pair = r.get("pair") or ("B->A" if "_spBA_" in r["id"] else "?")
                if r.get("error") or r.get("outcome") is None or pair != "B->A" or int(r.get("ts") or 0) < 32:
                    continue
                if r.get("variant") not in ("flush", None) and "teach" not in d:
                    continue
                f = os.path.join(d, "traj", r["id"] + ".npz")
                if os.path.exists(f):
                    r["_traj"] = f
                    out.append(r)
    return out


def klass(r, z):
    sw = np.where(z["prompt_new"] > 0)[0]
    s = int(sw[0]) if len(sw) else int(r["ts"])
    ee, new = z["ee"].astype(float), z["new_obj"].astype(float)
    dxy = np.linalg.norm(ee[s:, :2] - new[s:, :2], axis=1)
    lifted = bool((new[s:, 2] - new[s, 2]).max() > 0.03)
    if r["outcome"] == "success":
        return "S"
    if r["outcome"] == "capture":
        return "C"
    if not (dxy < 0.06).any():
        return "T1"
    return "T3" if lifted else "T2"


def reset(cache, r):
    env = cache.env(r.get("suite") or "libero_spatial", int(r.get("scene") or 9))
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
        obs = env.set_init_state(cache.init_states(r.get("suite") or "libero_spatial", int(r.get("scene") or 9))[init_id])
    for _ in range(C.SETTLE):
        obs, _, _, _ = env.step(C.DUMMY)
    return env, obs


def q4(env):
    sim = env.env.sim
    jid = sim.model.joint_name2id("robot0_joint4")
    q = float(sim.data.qpos[sim.model.jnt_qposadr[jid]])
    hi = float(sim.model.jnt_range[jid][1])
    return q, hi - q


def handback(cache, runs, name, mx):
    """c2: A->B switch episodes of a teacher set: the elbow margin at the hand-back (the last scripted-transit step,
    traj 'bridge' flag) and the reach / height there, replayed exactly; successes only."""
    dirs = [name] if os.path.isdir(name) else sorted(glob.glob(os.path.join(runs, "pi05_%s_p*" % name)))
    sel = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "results_*.jsonl"))):
            for ln in open(p):
                try:
                    r = json.loads(ln)
                except Exception:  # noqa: BLE001
                    continue
                pair = r.get("pair") or ("A->B" if "_spAB_" in r["id"] else "?")
                f = os.path.join(d, "traj", r["id"] + ".npz")
                if pair == "A->B" and r.get("outcome") == "success" and int(r.get("ts") or 0) >= 12 and os.path.exists(f):
                    r["_traj"] = f
                    sel.append(r)
    rs = np.random.default_rng(0)
    if len(sel) > mx:
        sel = [sel[i] for i in sorted(rs.choice(len(sel), size=mx, replace=False))]
    st = collections.defaultdict(list)
    for r in sel:
        z = np.load(r["_traj"])
        if "bridge" not in z.files or not z["bridge"].any():
            continue
        e = int(np.where(z["bridge"] > 0)[0][-1])
        env, obs = reset(cache, dict(r, scene=r.get("scene") or 7))
        for t in range(e + 1):
            obs, _, _, _ = env.step(z["act"][t].astype(np.float64).tolist())
        q, m = q4(env)
        p = np.asarray(obs["robot0_eef_pos"], np.float64)
        st["elbow margin at the hand-back (rad)"].append(m)
        st["elbow at the limit at the hand-back (< 0.1 rad)"].append(float(m < 0.1))
        st["reach at the hand-back (cm)"].append(100 * float(np.linalg.norm(p[:2] - BASE)))
        st["height at the hand-back (m)"].append(float(p[2]))
        st["replay error (mm)"].append(1000 * float(np.linalg.norm(p - z["ee"][e + 1].astype(np.float64))) if e + 1 < len(z["ee"]) else 0.0)
    lab = os.path.basename(name.rstrip("/"))
    print("  %s A->B hand-back n=%3d | " % (lab, len(st["reach at the hand-back (cm)"])) + " | ".join(
        "%s %s" % (k, ("%.2f" % np.mean(v)) if "(< 0.1" in k else ("%.2f" % np.median(v))) for k, v in sorted(st.items())), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--handback", nargs="*", default=[], help="c2: teacher sets whose A->B hand-back posture is measured")
    a = ap.parse_args()
    cache = C.SceneCache()
    for name in a.handback:
        handback(cache, a.runs, name, a.max)
    for name in a.sets:
        rows = rows_of(a.runs, name)
        by = collections.defaultdict(list)
        for r in rows:
            z = np.load(r["_traj"])
            by[klass(r, z)].append(r)
        lab = os.path.basename(name.rstrip("/"))
        print("== %s: %d B->A switch episodes (t_s >= 32): %s" % (lab, len(rows), {k: len(v) for k, v in sorted(by.items())}), flush=True)
        for c in ("S", "T1", "T3", "C", "T2"):
            sel = by.get(c, [])
            rs = np.random.default_rng(0)
            if len(sel) > a.max:
                sel = [sel[i] for i in sorted(rs.choice(len(sel), size=a.max, replace=False))]
            if not sel:
                continue
            st = collections.defaultdict(list)
            for r in sel:
                z = np.load(r["_traj"])
                ts = int(r["ts"])
                env, obs = reset(cache, r)
                margins = []
                for t in range(ts):
                    margins.append(q4(env)[1])
                    obs, _, _, _ = env.step(z["act"][t].astype(np.float64).tolist())
                q, m = q4(env)
                p = np.asarray(obs["robot0_eef_pos"], np.float64)
                st["replay error (mm)"].append(1000 * float(np.linalg.norm(p - z["ee"][ts].astype(np.float64))))
                st["elbow margin at the switch (rad)"].append(m)
                st["elbow at the limit at the switch (< 0.1 rad)"].append(float(m < 0.1))
                mm = np.asarray(margins + [m])
                st["share of pre-switch steps at the limit"].append(float((mm < 0.1).mean()))
                hit = np.where(mm < 0.1)[0]
                if len(hit):
                    st["first step at the limit"].append(int(hit[0]))
                st["reach at the switch (cm)"].append(100 * float(np.linalg.norm(p[:2] - BASE)))
                st["t_s"].append(ts)
            print("  %s %-3s n=%3d | " % (lab, c, len(sel)) + " | ".join(
                "%s %s" % (k, ("%.2f" % np.mean(v)) if ("share" in k or "(< 0.1" in k) else ("%.2f" % np.median(v)))
                for k, v in sorted(st.items())), flush=True)


if __name__ == "__main__":
    main()
