"""Instruction-switch harness for LIBERO (runs in the LIBERO client venv, Python 3.8).

Each line of the spec file (JSONL) describes one episode:
  id        unique string
  suite     libero_goal | libero_object | libero_spatial | libero_10
  scene     task id whose BDDL file and initial states define the scene
  init      index of the LIBERO initial state
  old       task id of the first instruction
  new       task id of the second instruction (null = no switch)
  ts        switch step, counted in policy control steps after the settle phase (null = no switch)
  method    vanilla | flush | retreat | cig | bias
  w, beta   strength of contrastive guidance / attention bias (cig / bias only)
  seed      seed of the simulator reset and of the policy noise
  record    store the policy inputs at every query (for offline analysis)
  video     store a 448x448 agent-view video
  stop_on_old  stop as soon as the old goal is reached after the switch (default true)
Results: one JSON line per episode in <out>/results_<worker>.jsonl and a compressed
trajectory file <out>/traj/<id>.npz.
"""
import argparse
import collections
import json
import math
import os
import pathlib
import sys
import time
import traceback
import zlib

import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from openpi_client import image_tools
from openpi_client import websocket_client_policy as wcp
import robosuite.utils.transform_utils as T

DUMMY = [0.0] * 6 + [-1.0]
SETTLE = 10
RES = 256
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class SceneCache:
    def __init__(self):
        self.bd = benchmark.get_benchmark_dict()
        self.suites = {}
        self.envs = {}
        self.goals = {}
        self.inits = {}

    def suite(self, name):
        if name not in self.suites:
            self.suites[name] = self.bd[name]()
        return self.suites[name]

    def bddl(self, suite, tid):
        t = self.suite(suite).get_task(tid)
        return os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file), t.language

    def goal(self, suite, tid):
        key = (suite, tid)
        if key not in self.goals:
            f, lang = self.bddl(suite, tid)
            p = robosuite_parse_problem(f)
            self.goals[key] = (p["goal_state"], lang, p.get("obj_of_interest", []))
        return self.goals[key]

    def env(self, suite, scene):
        key = (suite, scene)
        if key not in self.envs:
            # one env at a time per worker keeps the memory footprint small
            for k in list(self.envs):
                try:
                    self.envs[k].close()
                except Exception:
                    pass
                del self.envs[k]
            f, _ = self.bddl(suite, scene)
            self.envs[key] = OffScreenRenderEnv(bddl_file_name=f, camera_heights=RES, camera_widths=RES)
        return self.envs[key]

    def init_states(self, suite, scene):
        key = (suite, scene)
        if key not in self.inits:
            self.inits[key] = self.suite(suite).get_task_init_states(scene)
        return self.inits[key]


def eval_goal(env, goal):
    try:
        return all(bool(env.env._eval_predicate(g)) for g in goal)
    except KeyError:
        return False


def obj_pos(env, name):
    try:
        st = env.env.object_states_dict[name]
        return np.array(st.get_geom_state()["pos"], dtype=np.float32)
    except Exception:
        return np.full(3, np.nan, dtype=np.float32)


def key_object(goal):
    # first argument of the first goal predicate, e.g. ("on", "akita_black_bowl_1", "plate_1")
    for g in goal:
        if len(g) >= 2:
            return g[1]
    return None


def make_element(obs, prompt, resize=224, raw=False):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    raw_img = img.copy() if raw else None
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize))
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"])).astype(np.float32)
    el = {"observation/image": img, "observation/wrist_image": wrist,
          "observation/state": state, "prompt": str(prompt)}
    if raw_img is not None:
        el["observation/image_raw"] = raw_img
    return el


def retreat_action(obs, home_pos, home_quat, k):
    """Scripted OSC_POSE controller: open the gripper and lift for 6 steps, then return to the home pose."""
    p = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    q = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    if k < 6:
        return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0], float(np.linalg.norm(home_pos - p))
    dp = np.clip((home_pos - p) / 0.05, -1.0, 1.0)
    rot_err = T.quat2axisangle(T.quat_multiply(home_quat, T.quat_inverse(q)))
    dr = np.clip(rot_err / 0.5, -0.5, 0.5)
    return np.concatenate([dp, dr, [-1.0]]).tolist(), float(np.linalg.norm(home_pos - p))


class Tunnel:
    """Wormhole v2 traversal: carries the end effector to the selected mouth of the new task
    (lift to clear the old object, transit above the workspace, settle on the mouth), tracked in
    closed loop through the same OSC_POSE action interface as the policy. The commanded pose action
    changes by at most `rate` per step, starting from the last executed action (tidal ramp); the
    gripper is kept open."""

    def __init__(self, goal_pos, goal_quat, p0, a_prev, clear=0.03, lift=0.05, vmax=1.0, rate=0.3,
                 arrive=0.02, max_steps=90):
        self.goal = np.asarray(goal_pos, dtype=np.float64)
        self.qg = np.asarray(goal_quat, dtype=np.float64)
        self.ztop = max(float(p0[2]), float(self.goal[2])) + clear
        self.zlift = min(self.ztop, float(p0[2]) + lift)
        self.vmax, self.rate, self.arrive, self.max_steps = vmax, rate, arrive, max_steps
        self.phase, self.k = 0, 0
        self.f = np.asarray(a_prev[:6], dtype=np.float64) if a_prev is not None else np.zeros(6)

    def act(self, p, q):
        p = np.asarray(p, dtype=np.float64)
        if self.phase == 0:
            w = np.array([p[0], p[1], self.zlift + 0.01])
            if p[2] >= self.zlift:
                self.phase = 1
        if self.phase == 1:
            w = np.array([self.goal[0], self.goal[1], self.ztop])
            if np.linalg.norm(p[:2] - self.goal[:2]) < 0.02:
                self.phase = 2
        if self.phase == 2:
            w = self.goal
        dp = np.clip((w - p) / 0.05, -1.0, 1.0) * self.vmax
        rot = T.quat2axisangle(T.quat_multiply(self.qg, T.quat_inverse(np.asarray(q, dtype=np.float64))))
        dr = np.clip(rot / 0.5, -0.5, 0.5)
        step = np.concatenate([dp, dr]) - self.f  # rate limiter: the pose action changes by at most `rate`
        nrm = float(np.linalg.norm(step))
        self.f = self.f + (step if nrm <= self.rate else step * (self.rate / nrm))
        self.k += 1
        err = float(np.linalg.norm(p - self.goal))
        done = (self.phase == 2 and err < self.arrive) or self.k >= self.max_steps
        return np.concatenate([self.f, [-1.0]]).astype(np.float32), done, err


class RobustClient:
    """Reconnects and retries a query if the websocket drops (e.g. keepalive timeout)."""

    def __init__(self, host, port):
        self.host, self.port = host, port
        self.c = wcp.WebsocketClientPolicy(host, port)

    def infer(self, el):
        for k in range(8):
            try:
                return self.c.infer(el)
            except Exception as e:  # noqa: BLE001
                print(f"infer failed ({e!r}); reconnecting [{k}]", flush=True)
                time.sleep(5 + 5 * k)
                try:
                    self.c = wcp.WebsocketClientPolicy(self.host, self.port)
                except Exception as e2:  # noqa: BLE001
                    print(f"reconnect failed ({e2!r})", flush=True)
        raise RuntimeError("policy server unreachable")


def run_episode(spec, client, cache, out_dir):
    suite = spec["suite"]
    scene = int(spec["scene"])
    ts = spec.get("ts")
    method = spec.get("method", "vanilla")
    seed = int(spec.get("seed", 7))
    record = bool(spec.get("record", False))
    video = bool(spec.get("video", False))
    stop_on_old = bool(spec.get("stop_on_old", True))
    replan = int(spec.get("replan", 5))
    max_steps = int(spec.get("max_steps", MAX_STEPS[suite]))

    def resolve(which):
        tid = spec.get(which)
        lang, goal = spec.get(which + "_lang"), spec.get(which + "_goal")
        if tid is not None:
            g, l, _ = cache.goal(suite, int(tid))
            lang = lang or l
            goal = goal or g
        return goal, lang

    old_goal, old_lang = resolve("old")
    new_goal, new_lang = resolve("new")
    old_obj, new_obj = key_object(old_goal), (key_object(new_goal) if new_goal else None)

    env = cache.env(suite, scene)
    np.random.seed(seed)
    env.seed(seed)
    env.reset()
    init_states = cache.init_states(suite, scene)
    obs = env.set_init_state(init_states[int(spec["init"])])
    for _ in range(SETTLE):
        obs, _, _, _ = env.step(DUMMY)

    if new_goal is not None and eval_goal(env, new_goal):
        return {"invalid": "new goal already satisfied at start"}
    if new_goal is not None and any(len(g) >= 2 and g[1] not in env.env.object_states_dict for g in new_goal):
        return {"invalid": "new goal refers to objects absent from the scene"}

    home_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64)
    home_quat = np.array(obs["robot0_eef_quat"], dtype=np.float64)

    prompt = old_lang
    plan = collections.deque()
    q_idx = 0
    switched = False
    retreating = False
    retreat_steps = 0
    worm_left = 0
    worm_log = []
    lib = None
    head = None
    tun = None
    tun_pending = False
    lib2 = None
    if method == "wormhole2":
        import wormhole as W
        lib2 = W.EntryLibrary2(spec["library"], int(spec.get("worm_h", 10)))
    if method == "wormhole":
        import wormhole as W
        lib = W.EntryLibrary(spec["library"], int(spec.get("worm_h", 10)))
        if spec.get("head"):
            import bridge_np
            head = bridge_np.BridgeNP(spec["head"])
    t_old = t_new = None
    traj = collections.defaultdict(list)
    rec = collections.defaultdict(list)
    frames = []
    infer_ms = []
    t = 0
    err = None
    chunk_q = 0
    while t < max_steps:
        try:
            if new_goal is not None and ts is not None and not switched and t >= int(ts):
                switched = True
                prompt = new_lang
                if method in ("flush", "retreat", "cig", "bias", "wormhole", "wormhole2"):
                    plan.clear()
                if method == "wormhole2":
                    tun_pending = True
                if method == "retreat":
                    retreating = True
                if method == "wormhole":
                    worm_left = int(spec.get("worm_chunks", 2))
            if tun_pending:
                tun_pending = False
                p = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                g = float(obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1])
                d_b = lib2.manifold_dist(p)
                if d_b >= float(spec.get("worm2_gate", 0.02)):
                    t0 = time.time()
                    X = None
                    if spec.get("worm2_ot", True):
                        el = make_element(obs, old_lang)
                        el["bh"] = {"mode": "multi", "n": int(spec.get("worm_n", 16)),
                                    "noise_seed": int(zlib.crc32(f"{seed}-{spec['init']}-w2-{t}".encode()) % 100000)}
                        res = client.infer(el)
                        X = np.asarray(res["samples"], dtype=np.float64)[:, :int(spec.get("worm_h", 10))]
                    mo = W.select_mouth(lib2, p, X, margin=int(spec.get("worm2_margin", 10)),
                                        v=float(spec.get("worm2_v", 0.015)), lam=float(spec.get("worm2_lam", 1.0)),
                                        kappa=float(spec.get("worm2_kappa", 5.0)), clear=float(spec.get("worm2_clear", 0.03)),
                                        m=int(spec.get("worm_m", 64)), eta=float(spec.get("worm_eta", 0.1)),
                                        use_ot=X is not None)
                    infer_ms.append((time.time() - t0) * 1000)
                    a_prev = traj["act"][-1].astype(np.float64) if traj["act"] else np.zeros(7)
                    tun = Tunnel(mo["pos"], mo["quat"], p, a_prev, clear=float(spec.get("worm2_clear", 0.03)),
                                 lift=float(spec.get("worm2_lift", 0.05)), vmax=float(spec.get("worm2_vmax", 1.0)),
                                 rate=float(spec.get("worm2_rate", 0.3)), arrive=float(spec.get("worm2_arrive", 0.02)),
                                 max_steps=int(spec.get("worm2_max", 90)))
                    worm_log.append({"t": t, "d_new": d_b, "grip": g, "mouth": [float(v) for v in mo["pos"]],
                                     "mouth_t": mo["t"], "mouth_rem": mo["rem"], "travel": mo["travel"],
                                     "score": mo["score"], "ms": infer_ms[-1]})
                else:
                    worm_log.append({"t": t, "d_new": d_b, "grip": g, "skipped": True})
            if retreating:
                action, dist = retreat_action(obs, home_pos, home_quat, retreat_steps)
                retreat_steps += 1
                if (retreat_steps > 6 and dist < 0.02) or retreat_steps >= 60:
                    retreating = False
                    plan.clear()
                act = np.asarray(action, dtype=np.float32)
                chunk_q = -1
            elif tun is not None:
                act, done, terr = tun.act(obs["robot0_eef_pos"], obs["robot0_eef_quat"])
                chunk_q = -2
                if done:
                    worm_log[-1].update({"steps": tun.k, "arrive_err": terr})
                    tun = None
                    plan.clear()
            elif worm_left > 0 and not plan:
                # open the wormhole from the current state: exits under the old instruction,
                # entries retrieved from executions of the new instruction, OT throat, bridge chunk
                el = make_element(obs, old_lang)
                el["bh"] = {"mode": "multi", "n": int(spec.get("worm_n", 16)),
                            "noise_seed": int(zlib.crc32(f"{seed}-{spec['init']}-w{worm_left}-{t}".encode()) % 100000)}
                t0 = time.time()
                res = client.infer(el)
                hw = int(spec.get("worm_h", 10))
                X = np.asarray(res["samples"], dtype=np.float64)[:, :hw]
                p = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                g = float(obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1])
                if spec.get("worm_naive"):  # ablation: entries ignore state compatibility
                    rs = np.random.default_rng(t)
                    idx = rs.choice(len(lib.states), size=int(spec.get("worm_m", 32)), replace=False)
                    Y, wY = lib.segs[idx], np.full(len(idx), 1.0 / len(idx))
                else:
                    Y, wY, idx = lib.retrieve(p, g, m=int(spec.get("worm_m", 32)))
                a_prev = traj["act"][-1].astype(np.float64) if traj["act"] else np.zeros(7)
                br, d_ot, _ = W.bridge_chunk(X, Y, wY, a_prev, eta=float(spec.get("worm_eta", 0.1)),
                                             ramp=int(spec.get("worm_ramp", 3)),
                                             anchor=not spec.get("worm_noanchor", False),
                                             use_ot=not spec.get("worm_noot", False),
                                             head=head, state=np.r_[p, g])
                infer_ms.append((time.time() - t0) * 1000)
                worm_log.append({"t": t, "dot_norm": float(np.linalg.norm(d_ot[:, :3])),
                                 "entry_dist": float(np.linalg.norm(lib.states[idx[0], :3] - p))})
                plan.extend(br[:int(spec.get("worm_exec", hw))])
                worm_left -= 1
                act = np.asarray(plan.popleft(), dtype=np.float32)
                chunk_q = -2
            elif plan and chunk_q == -2:
                act = np.asarray(plan.popleft(), dtype=np.float32)
            else:
                if not plan:
                    el = make_element(obs, prompt, raw=bool(spec.get("raw_image", False)))
                    ctl = {"noise_seed": int(zlib.crc32(f"{seed}-{spec['init']}-{q_idx}-{spec.get('noise_tag', '')}".encode()))}
                    if switched and method == "cig":
                        ctl.update({"mode": "cig", "prompt_old": old_lang, "w": float(spec.get("w", 1.0))})
                    if switched and method == "bias":
                        ctl.update({"mode": "bias", "beta": float(spec.get("beta", 1.0))})
                    el["bh"] = ctl
                    t0 = time.time()
                    res = client.infer(el)
                    infer_ms.append((time.time() - t0) * 1000)
                    chunk = np.asarray(res["actions"], dtype=np.float32)
                    if record:
                        rec["img"].append(el["observation/image"])
                        rec["wrist"].append(el["observation/wrist_image"])
                        rec["state"].append(el["observation/state"])
                        rec["t"].append(t)
                        rec["prompt_new"].append(1 if switched else 0)
                        rec["chunk"].append(chunk)
                    plan.extend(chunk[:replan])
                    q_idx += 1
                act = np.asarray(plan.popleft(), dtype=np.float32)
                chunk_q = q_idx - 1
            traj["ee"].append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
            traj["aa"].append(quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32))
            traj["grip"].append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))
            traj["old_obj"].append(obj_pos(env, old_obj) if old_obj else np.full(3, np.nan, np.float32))
            traj["new_obj"].append(obj_pos(env, new_obj) if new_obj else np.full(3, np.nan, np.float32))
            traj["act"].append(act)
            traj["prompt_new"].append(1 if switched else 0)
            traj["retreat"].append(1 if (switched and method == "retreat" and chunk_q == -1) else 0)
            traj["bridge"].append(1 if chunk_q == -2 else 0)
            if video and t % 2 == 0:
                frames.append(env.env.sim.render(camera_name="agentview", width=448, height=448)[::-1])
            obs, _, _, _ = env.step(act.tolist())
            t += 1
            ok_old = eval_goal(env, old_goal)
            ok_new = eval_goal(env, new_goal) if new_goal is not None else False
            if ok_old and t_old is None:
                t_old = t
            if ok_new and t_new is None:
                t_new = t
            if new_goal is None:
                if ok_old:
                    break
            else:
                if ok_new:
                    break
                if ok_old and stop_on_old:
                    break
        except Exception as e:  # noqa: BLE001
            err = repr(e)
            traceback.print_exc()
            break

    if video and frames:
        import imageio
        vdir = pathlib.Path(out_dir) / "videos"
        vdir.mkdir(parents=True, exist_ok=True)
        frames.append(env.env.sim.render(camera_name="agentview", width=448, height=448)[::-1])
        imageio.mimwrite(vdir / f"{spec['id']}.mp4", frames, fps=10, quality=8)
        sim = env.env.sim
        cid = sim.model.camera_name2id("agentview")
        np.savez_compressed(vdir / f"{spec['id']}_frames.npz", frames=np.stack(frames), every=2,
                            cam_pos=np.array(sim.data.cam_xpos[cid]), cam_rot=np.array(sim.data.cam_xmat[cid]).reshape(3, 3),
                            cam_fovy=float(sim.model.cam_fovy[cid]), size=448)
    tdir = pathlib.Path(out_dir) / "traj"
    tdir.mkdir(parents=True, exist_ok=True)
    arrs = {k: np.stack(v) for k, v in traj.items() if len(v)}
    np.savez_compressed(tdir / f"{spec['id']}.npz", **arrs)
    if record and rec:
        rdir = pathlib.Path(out_dir) / "obs"
        rdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(rdir / f"{spec['id']}.npz", **{k: np.stack(v) for k, v in rec.items()},
                            prompt_old=old_lang, prompt_newtxt=(new_lang or ""))
    if new_goal is None:
        outcome = "success" if t_old is not None else "timeout"
    else:
        if t_new is not None and (t_old is None or t_new <= t_old):
            outcome = "success"
        elif t_old is not None:
            outcome = "capture"
        else:
            outcome = "timeout"
    if new_goal is not None and not switched:
        outcome = "done_before_switch" if t_old is not None else "timeout"
    return {"outcome": outcome, "t_old": t_old, "t_new": t_new, "steps": t, "error": err, "switched": switched,
            "old_lang": old_lang, "new_lang": new_lang, "retreat_steps": retreat_steps,
            "infer_ms": float(np.mean(infer_ms)) if infer_ms else None, "n_queries": q_idx, "worm": worm_log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8810)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / f"results_{args.worker}.jsonl"
    done = set()
    for p in out.glob("results_*.jsonl"):
        for ln in open(p):
            try:
                done.add(json.loads(ln)["id"])
            except Exception:
                pass
    specs = [json.loads(ln) for ln in open(args.spec) if ln.strip()]
    mine = [s for i, s in enumerate(specs) if i % args.nworkers == args.worker and s["id"] not in done]
    print(f"worker {args.worker}: {len(mine)} episodes to run", flush=True)
    client = RobustClient(args.host, args.port)
    cache = SceneCache()
    t0 = time.time()
    for k, spec in enumerate(mine):
        t1 = time.time()
        r = run_episode(spec, client, cache, out)
        r.update({kk: spec.get(kk) for kk in ("id", "suite", "scene", "init", "old", "new", "ts", "method", "w", "beta", "seed", "tag", "pair", "variant")})
        r["wall_s"] = round(time.time() - t1, 2)
        with open(res_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        if k % 10 == 0:
            el = time.time() - t0
            print(f"[w{args.worker}] {k+1}/{len(mine)} {spec['id']} -> {r.get('outcome', r.get('invalid'))} "
                  f"({r['wall_s']}s, eta {el/(k+1)*(len(mine)-k-1)/60:.1f} min)", flush=True)
    print("worker done", flush=True)
    # the websocket client keeps a non-daemon thread alive, which made finished workers hang at
    # interpreter shutdown (holding RAM and an EGL context); results are already on disk, so exit hard.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
