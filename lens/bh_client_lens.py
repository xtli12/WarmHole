"""Instruction-switch harness for LIBERO (runs in the LIBERO client venv, Python 3.8).

Each line of the spec file (JSONL) describes one episode:
  id        unique string
  suite     libero_goal | libero_object | libero_spatial | libero_10
  scene     task id whose BDDL file and initial states define the scene
  init      index of the LIBERO initial state; lens14: init >= 10000 = a procedurally sampled layout, seed init - 10000
            (np.random.seed + env.seed, then env.reset(): LIBERO's own placement sampler, as for the official states)
  old       task id of the first instruction
  new       task id of the second instruction (null = no switch)
  ts        switch step, counted in policy control steps after the settle phase (null = no switch)
  method    vanilla | flush | retreat | cig | bias | wormhole | wormhole2 | lens | lens_cut | attn_worm | attn_flow
  lens_variant, lens_beta, lens_queries, lens_nsamp
            attention lens (method lens): at the switch the queued actions are dropped; every later query
            re-generates the chunk with an additive bias beta * map on the action queries' attention to the image
            patches.  variant cur: map = attention difference A(l_new) - A(l_old) on the current observation;
            early: the difference on the first observation of the episode; oracle: +/- around the new / old object
            (projected from the simulator, an upper bound); learned: predicted by the focus network.  lens_queries
            limits the number of steered queries after the switch.
  lens_off, lens_aecut, lens_queries, lens_gate, lens_theta
            attention truncation (method lens_cut): for at most lens_queries queries after the switch the chunk is
            generated with the cameras in lens_off masked out of the model and the key groups in lens_aecut
            (wrist | base | lang | state) hidden from the action queries.  With lens_gate (dA_l1 | dchunk) every
            truncated query also measures how much the instruction steers the untruncated policy there, and the
            truncation is released once that engagement reaches lens_theta (vigilance gate).
  lens_layers, lens_kind, lens_frac, lens_hover, lens_recheck, lens_handover, lens_tau, lens_delta, lens_ema
            attention-guided wormhole (method attn_worm, needs BH_DEPTH=1): at the switch the backbone attention
            of the new and the old instruction over the base-camera patches is compared (layers lens_layers);
            the region where the new instruction draws more attention is lifted to 3-D with the depth image,
            and the end effector is carried above it (lift, transit, settle; gripper open) at lens_hover above
            the region's top.  The attention is re-read every lens_recheck steps (the target follows it); with
            lens_handover=engage the policy takes over as soon as its own chunk under the new instruction heads
            to the attended region and away from the region of the old one; with lens_handover=vigil as soon as
            the vigilance probe (every lens_vigil_every steps, after lens_vigil_min tunnel steps) shows that the
            instruction steers the policy again (lens_vigil_metric >= lens_theta_open); otherwise on arrival.
  lens_target_src, vlm_url, vlm_res, vlm_old, vlm_est, vlm_strategy, vlm_offset
            attn_worm with lens_target_src=vlm: the target is the object the new instruction refers to according to a
            grounding foundation model (ground_server.py, Qwen2.5-VL), queried with the upright agent-view image
            (candidate enumeration vlm_strategy, set-of-mark selection); 3-D from the depth image with the box
            estimator vlm_est (lens_geom.bbox_target: inner | rim | rimc | seg); vlm_offset (dx, dy in m) is added to
            the object centre (the policy's own grasp offset).
  save_switch
            store the agent view at the switch (512 px RGB, metric depth, camera, true objects) in <out>/switch/<id>.npz
  lens_target_src=oracle, oracle_ztop
            diagnostic upper bound of the wormhole design: the target is the true position of the new object
            (simulator), object top = position + oracle_ztop; vlm_offset applies as well.
  retreat_vigil (method retreat)
            vigilance-gated retreat: every lens_vigil_every retreat steps (after lens_vigil_min) the probe measures
            whether the instruction steers the policy again (lens_vigil_metric >= lens_theta_open); then it stops.
  lens_hover_home, lens_gate_open, lens_theta_open
            attn_worm options: exit at the start height above the attended region (an attention-directed partial
            retreat), and open the wormhole only if the policy is captured (vigilance probe below lens_theta_open).
  lens_lam, lens_gsteps, lens_arrive (+ lens_off, lens_layers, lens_frac, lens_hover, lens_recheck, lens_gate)
            attention-guided flow (method attn_flow, needs BH_DEPTH=1): the target is read from the attention as in
            attn_worm, but the path is generated by the policy itself: every query until release the chunk is sampled
            with the cameras in lens_off masked out and the flow guided so that its first lens_gsteps actions move by
            lens_lam toward the hover point; release on arrival (lens_arrive), by the vigilance gate, or after
            lens_queries queries.
  w, beta   strength of contrastive guidance / attention bias (cig / bias only)
  seed      seed of the simulator reset and of the policy noise
  record    store the policy inputs at every query (for offline analysis)
  aw_ctl, aw_keep_prompt
            diagnostics of a self-trained wormhole policy (aw_server.py): aw_ctl (dict) is added to every request
            after the switch ("__OLD__" / "__NEW__" in its string values are replaced by the old / new instruction),
            e.g. {"worm_scale": 0} or {"donor_prompt": "__NEW__"}; with aw_keep_prompt the instruction tokens stay
            the old ones after the switch (only the transplanted wormhole carries the new instruction)
  dg_k, dg_ctl (method attn_worm, needs --teacher_port)
            DAgger episodes of the self-trained wormhole policy: the policy on --port (the student) acts before the
            switch and for dg_k control steps after it (re-queried under the new instruction; dg_ctl, e.g.
            {"worm_scale": 0, "hide_loc": true}, is added to those student queries only); then the teacher takes over
            from the state the student reached: the scripted traversal of this attn_worm variant (e.g. the oracle
            wo_off_h15), and after the hand-back the policy on --teacher_port (the released policy).  Student actions
            are recorded with src -4 (before the switch) / -3 (after), so that only teacher actions become labels.
  rec_steps store, at every control step, the policy inputs under the current instruction (224 px agent and wrist
            views, state), the executed action, who produced it (policy query / scripted teacher), both camera poses
            and the true object positions in <out>/steps/<id>.npz (teacher data of the self-trained wormhole policy)
  video     store a 448x448 agent-view video
  stop_on_old  stop as soon as the old goal is reached after the switch (default true)
  dg_cond   lens15 (handover 7.13, round g), with dg_k: conditional take-over.  The student keeps control after the switch
            for dg_k steps, or less once it is frozen (at least freeze_min steps after the switch and the end effector
            moved < freeze_path m over the last freeze_win steps).  Then the teacher takes over only if the student is
            still in transit (horizontal distance end effector - new object > dxy m and the new object not lifted by more
            than 3 cm); otherwise the episode ends at once (outcome "skip", no teacher label).  At a take-over the
            simulator state is stored in <out>/snap/<id>.npz (the counterfactual branch starts from it).  Results:
            dg_reason (k / freeze / none), dg_since (steps after the switch), dg_dxy, dg_z, snap.
  lens_corridor  lens16 (handover 7.15, round i): H (m).  At the first traversal of an attn_worm episode, if the new
            object rests lower than the old one (initial heights differ by > 5 cm), the transit lifts straight up to H
            above the new object (at least lens_clear above the start), crosses at that height and then goes down to the
            hand-back point (lens_clear = max(lens_clear, z_new + H - max(z_ee, z_goal)), lens_lift = 1); logged as
            corridor_clear in the first worm entry.  Otherwise the transit is unchanged.
  lens_margin  lens17 (handover 7.16, round j): M (rad).  During the scripted transit of an attn_worm episode the elbow
            margin (joint 4 to its upper limit, privileged simulator state) is read at every step; as soon as it is
            below M and more than 0.05 rad below its value at the start of the transit (the transit itself is
            straightening the arm), the transit hands back at once (handover "margin").  The first worm entry gets
            margin_start, margin_min and margin_handback.  Without lens_margin the behaviour is that of lens16.
  resume    lens15: a stored simulator state (<out>/snap/<id>.npz): after the usual reset of the layout the state is
            restored from it, without the settle steps (resume_err = |end effector - stored end effector| right after
            the restore); used with ts 0 and dg_k 0 for the counterfactual branch (the teacher at once, other instruction).
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
RLAYOUT = 10000      # lens14: init >= RLAYOUT = procedurally sampled layout with seed init - RLAYOUT (not an official state)
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
            self.envs[key] = OffScreenRenderEnv(bddl_file_name=f, camera_heights=RES, camera_widths=RES,
                                                camera_depths=bool(os.environ.get("BH_DEPTH")))
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


def elbow_margin(env):
    """lens17: distance (rad) of the arm's joint 4 (elbow) from its upper limit (the fully stretched arm)."""
    sim = env.env.sim
    jid = sim.model.joint_name2id("robot0_joint4")
    return float(sim.model.jnt_range[jid][1]) - float(sim.data.qpos[sim.model.jnt_qposadr[jid]])


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


def camera(env, name="agentview"):
    sim = env.env.sim
    cid = sim.model.camera_name2id(name)
    return np.array(sim.data.cam_xpos[cid]), np.array(sim.data.cam_xmat[cid]).reshape(3, 3), float(sim.model.cam_fovy[cid])


def attention_map(G_new, G_old, layers, kind="diff"):
    """Base-camera maps (new, old) from the backbone grounding (L, 512) of the new and the old instruction."""
    lo, hi = (int(x) for x in str(layers).split("-"))
    gn = np.asarray(G_new, np.float64)[lo:hi + 1, :256].mean(0)
    go = np.asarray(G_old, np.float64)[lo:hi + 1, :256].mean(0)
    if kind == "diff":
        return gn - go, go - gn
    if kind == "ratio":
        return np.log((gn + 1e-6) / (go + 1e-6)), np.log((go + 1e-6) / (gn + 1e-6))
    raise ValueError(kind)


def attention_target(env, obs, m, frac=0.5):
    """3-D target of the positive part of a base-camera map, lifted with the agent-view depth image."""
    import lens_geom as LG
    sim = env.env.sim
    ext = float(sim.model.stat.extent)
    zn, zf = float(sim.model.vis.map.znear) * ext, float(sim.model.vis.map.zfar) * ext
    D = LG.real_depth(LG.policy_depth(obs["agentview_depth"]), zn, zf)
    cam_pos, cam_rot, fovy = camera(env)
    return LG.map_target(np.clip(m, 0.0, None), D, cam_pos, cam_rot, fovy, frac=frac)


def render_view(env, obs, res):
    """Upright agent-view RGB (res, res, 3) and metric depth (res, res); falls back to the observation (256 px)."""
    sim = env.env.sim
    ext = float(sim.model.stat.extent)
    zn, zf = float(sim.model.vis.map.znear) * ext, float(sim.model.vis.map.zfar) * ext
    try:
        rgb, dep = sim.render(camera_name="agentview", width=res, height=res, depth=True)
    except Exception:  # noqa: BLE001
        rgb, dep = obs["agentview_image"], obs["agentview_depth"]
    rgb = np.ascontiguousarray(np.asarray(rgb)[::-1])
    dep = np.asarray(dep, np.float64).squeeze()[::-1]
    return rgb, zn / (1.0 - dep * (1.0 - zn / zf))


def vlm_ground(url, rgb, instr, timeout=900, strategy=None, instrs=()):
    """Ask the grounding server which object the instruction refers to: box in pixels of rgb.  strategy: how the
    server enumerates the candidates (ground_server.py); instrs: the instructions known for the scene."""
    import base64
    import io
    import urllib.request
    buf = io.BytesIO()
    try:
        from PIL import Image
        Image.fromarray(rgb).save(buf, format="PNG")
    except ImportError:
        import imageio
        imageio.imwrite(buf, rgb, format="png")
    payload = {"image": base64.b64encode(buf.getvalue()).decode(), "instruction": instr}
    if strategy:
        payload["strategy"] = strategy
    if instrs:
        payload["instructions"] = [str(x) for x in instrs if x]
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"grounding request failed ({e!r}) [{k}]", flush=True)
            time.sleep(3 + 3 * k)
    return {"bbox": None, "error": "grounding server unreachable"}


def vlm_targets(env, obs, new_lang, old_lang, spec):
    """Targets of the new and the old instruction from the grounding model (3-D via the depth image)."""
    import lens_geom as LG
    res = int(spec.get("vlm_res", 512))
    urls = str(spec.get("vlm_url", "http://127.0.0.1:8500/ground")).split(",")     # replicas: one per episode
    url = urls[zlib.crc32(str(spec.get("id", "")).encode()) % len(urls)]
    rgb, dep = render_view(env, obs, res)
    cam_pos, cam_rot, fovy = camera(env)
    info = {}
    out = []
    for name, instr in (("new", new_lang), ("old", old_lang)):
        if name == "old" and not spec.get("vlm_old", False):
            out.append({"xy": np.full(2, np.nan), "ztop": np.nan})
            continue
        r = vlm_ground(url, rgb, instr, strategy=spec.get("vlm_strategy"), instrs=(old_lang, new_lang))
        if r.get("error") == "grounding server unreachable":   # infrastructure, not the method: abort the episode
            raise RuntimeError(f"grounding server unreachable: {url}")
        info[f"vlm_{name}_bbox"] = r.get("bbox")
        info[f"vlm_{name}_label"] = r.get("label")
        info[f"vlm_{name}_ms"] = r.get("ms")
        info[f"vlm_{name}_how"] = r.get("how")
        info[f"vlm_{name}_ncand"] = len(r.get("candidates") or [])
        if r.get("bbox") is None:
            info["vlm_fail"] = r.get("raw", r.get("error"))
            out.append({"xy": np.full(2, np.nan), "ztop": np.nan})
            continue
        ex = (np.asarray(obs["robot0_eef_pos"], np.float64), float(spec.get("vlm_exclude", 0.10)))   # own gripper
        tg = LG.bbox_target(r["bbox"], dep, cam_pos, cam_rot, fovy, method=spec.get("vlm_est", "inner"),
                            exclude=ex if spec.get("vlm_exclude", 0.10) else None)
        info[f"vlm_{name}_center"] = [float(v) for v in tg["xy"]]
        if spec.get("vlm_offset") is not None:      # where the policy itself grasps, relative to the object centre
            tg["xy"] = np.asarray(tg["xy"], np.float64) + np.asarray(spec["vlm_offset"], np.float64)[:2]
        out.append(tg)
    if np.isnan(out[0]["xy"]).any():   # no answer: fall back to the start height above the scene centre
        out[0] = {"xy": np.array([0.0, 0.0]), "ztop": 1.0}
        info["vlm_fallback"] = True
    return out[0], out[1], info


def read_target(client, env, obs, new_lang, old_lang, spec, tag, new_obj):
    """Attention target: where the new instruction draws more backbone attention than the old one, lifted to 3-D;
    returns the hover point above it, the old instruction's region (xy), the server reply and a log entry."""
    el = make_element(obs, new_lang)
    el["bh"] = {"mode": "lens_target", "prompt_old": old_lang, "noise_seed": int(zlib.crc32(tag.encode()) % 100000)}
    t0 = time.time()
    res = client.infer(el)
    if spec.get("lens_learned") and res.get("M_new") is not None:     # learned head readout of the attention
        m_new = np.asarray(res["M_new"], np.float64)
        m_old = -m_new
    else:
        m_new, m_old = attention_map(res["G_new"], res["G_old"], spec.get("lens_layers", "0-17"),
                                     spec.get("lens_kind", "diff"))
    frac = float(spec.get("lens_frac", 0.5))
    tg = attention_target(env, obs, m_new, frac)
    tg_old = attention_target(env, obs, m_old, frac)
    goal = np.array([tg["xy"][0], tg["xy"][1], tg["ztop"] + float(spec.get("lens_hover", 0.08))])
    true_new = obj_pos(env, new_obj) if new_obj else np.full(3, np.nan)
    entry = {"target": [float(v) for v in goal], "target_old": [float(v) for v in tg_old["xy"]],
             "n_patches": tg["n_patches"], "err_xy_true": float(np.linalg.norm(goal[:2] - true_new[:2])),
             "z_top": tg["ztop"], "ms": (time.time() - t0) * 1000}
    return goal, np.asarray(tg_old["xy"]), res, entry


def oracle_map(env, old_obj, new_obj):
    """Attention-lens oracle: +1 around the new object, -1 around the old one, on the base-camera patches."""
    import lens_geom as LG
    sim = env.env.sim
    cid = sim.model.camera_name2id("agentview")
    cam_pos = np.array(sim.data.cam_xpos[cid])
    cam_rot = np.array(sim.data.cam_xmat[cid]).reshape(3, 3)
    uv = LG.project_to_policy_image(np.stack([obj_pos(env, new_obj), obj_pos(env, old_obj)]), cam_pos, cam_rot,
                                    float(sim.model.cam_fovy[cid]))
    return LG.object_map(uv[0], uv[1])


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

    def retarget(self, goal_pos):
        """Move the goal (the attention-guided wormhole re-reads its target during the traversal)."""
        self.goal = np.asarray(goal_pos, dtype=np.float64)
        self.ztop = max(self.ztop, float(self.goal[2]))

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


def run_episode(spec, client, cache, out_dir, teacher=None):
    suite = spec["suite"]
    scene = int(spec["scene"])
    ts = spec.get("ts")
    method = spec.get("method", "vanilla")
    seed = int(spec.get("seed", 7))
    record = bool(spec.get("record", False))
    rec_steps = bool(spec.get("rec_steps", False))
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
    init_id = int(spec["init"])
    if init_id >= RLAYOUT:
        # lens14: a procedurally sampled layout -- LIBERO's own placement sampler (env.reset() re-samples every object in
        # its BDDL region), seeded with init - RLAYOUT; the official initial states were drawn by the same sampler
        np.random.seed(init_id - RLAYOUT)
        env.seed(init_id - RLAYOUT)
        obs = env.reset()
        np.random.seed(seed)
    else:
        np.random.seed(seed)
        env.seed(seed)
        env.reset()
        init_states = cache.init_states(suite, scene)
        obs = env.set_init_state(init_states[init_id])
    resume_err = None
    if spec.get("resume"):
        # lens15: the counterfactual branch starts from the simulator state stored at a take-over (no settle steps)
        zs = np.load(spec["resume"])
        obs = env.set_init_state(zs["state"])
        resume_err = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], np.float64) - zs["ee"].astype(np.float64)))
    for _ in range(0 if spec.get("resume") else SETTLE):
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
    tun_m0 = tun_mmin = None         # lens17: elbow margin at the start of the transit / its minimum
    tun_pending = False
    lib2 = None
    lens_q = 0
    lens_map = None
    lens_log = []
    el0 = None
    aw_pending = False
    aw_next = None
    aw = {}
    lens_released = False
    af = None
    aw_gate = {}
    aw_vig_next = 0
    dg_k = spec.get("dg_k")          # DAgger: student steps after the switch before the teacher takes over
    dg_mode = dg_k is not None
    if dg_mode and (teacher is None or method != "attn_worm"):
        raise RuntimeError("dg_k needs method attn_worm and --teacher_port (the teacher policy server)")
    dg_left = None
    dg_started = False
    dg_cond = spec.get("dg_cond")    # lens15: conditional take-over (handover 7.13)
    dg_info = {}
    dg_skip = False
    t_switch = None
    new_z0 = None
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
    stp = collections.defaultdict(list)
    frames = []
    infer_ms = []
    t = 0
    err = None
    chunk_q = 0
    while t < max_steps:
        try:
            if new_goal is not None and ts is not None and not switched and t >= int(ts):
                switched = True
                t_switch = t
                new_z0 = float(obj_pos(env, new_obj)[2]) if new_obj else None
                prompt = old_lang if spec.get("aw_keep_prompt") else new_lang
                if spec.get("save_switch"):
                    # the agent view at the switch (512 px RGB + metric depth, camera, true objects): lets the grounding
                    # strategies be compared offline on in-episode frames of the development split
                    try:
                        rgb_s, dep_s = render_view(env, obs, int(spec.get("vlm_res", 512)))
                        cp, cr, fv = camera(env)
                        sdir = pathlib.Path(out_dir) / "switch"
                        sdir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            sdir / f"{spec['id']}.npz", rgb=rgb_s, depth=dep_s.astype(np.float32), cam_pos=cp, cam_rot=cr,
                            fovy=fv, new_obj=obj_pos(env, new_obj) if new_obj else np.full(3, np.nan),
                            old_obj=obj_pos(env, old_obj) if old_obj else np.full(3, np.nan),
                            ee=np.asarray(obs["robot0_eef_pos"]), new_lang=str(new_lang), old_lang=str(old_lang), t=t)
                    except Exception as e:  # noqa: BLE001
                        print("switch frame not saved", repr(e), flush=True)
                if method in ("flush", "retreat", "cig", "bias", "wormhole", "wormhole2", "lens", "lens_cut", "attn_worm",
                              "attn_flow"):
                    plan.clear()
                if method == "attn_worm":
                    if dg_mode:
                        dg_left = int(dg_k)      # the student keeps control for dg_k steps under the new instruction
                    else:
                        aw_pending = True
                if method == "attn_flow":
                    af = {"active": True, "goal": None, "next": t, "q": 0}
                if method == "lens" and spec.get("lens_variant") == "early" and el0 is not None:
                    e = dict(el0)
                    e["prompt"] = new_lang
                    e["bh"] = {"mode": "lens_map", "prompt_old": old_lang, "nsamp_attn": int(spec.get("lens_nsamp", 2)),
                               "noise_seed": int(zlib.crc32(f"{seed}-{spec['init']}-lens0".encode()) % 100000)}
                    r0 = client.infer(e)
                    lens_map = np.asarray(r0["map"], dtype=np.float32)
                    lens_log.append({"t": t, "early_dA_l1": float(r0["dA_l1"])})
                if method == "wormhole2":
                    tun_pending = True
                if method == "retreat":
                    retreating = True
                if method == "wormhole":
                    worm_left = int(spec.get("worm_chunks", 2))
            if dg_mode and dg_cond and switched and not dg_started and dg_left is not None and dg_left > 0:
                # lens15: the student is frozen before dg_k steps -> take over now (if still in transit, see below)
                win = int(dg_cond.get("freeze_win", 12))
                if t - t_switch >= int(dg_cond.get("freeze_min", 20)) and len(traj["ee"]) >= win:
                    pts = np.vstack([np.asarray(traj["ee"][-win:], np.float64),
                                     np.asarray(obs["robot0_eef_pos"], np.float64)[None]])
                    if float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) < float(dg_cond.get("freeze_path", 0.012)):
                        dg_left = 0
                        dg_info["dg_trigger"] = "freeze"
            if dg_mode and switched and not dg_started and dg_left is not None and dg_left <= 0:
                if dg_cond:
                    p_now = np.asarray(obs["robot0_eef_pos"], np.float64)
                    pn = np.asarray(obj_pos(env, new_obj), np.float64)
                    dxy = float(np.linalg.norm(p_now[:2] - pn[:2]))
                    lifted = new_z0 is not None and float(pn[2]) - new_z0 > 0.03
                    dg_info.update(dg_since=int(t - t_switch), dg_dxy=dxy, dg_z=float(p_now[2]))
                    if dxy <= float(dg_cond.get("dxy", 0.07)) or lifted:
                        dg_info["dg_reason"] = "none"     # not in transit: no take-over, the episode ends here
                        dg_info.pop("dg_trigger", None)
                        dg_skip = True
                        break
                    dg_info["dg_reason"] = dg_info.pop("dg_trigger", "k")
                    sdir = pathlib.Path(out_dir) / "snap"
                    sdir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(sdir / f"{spec['id']}.npz", state=np.asarray(env.env.sim.get_state().flatten()),
                                        ee=p_now, t=t, since=int(t - t_switch), new_obj=pn,
                                        old_obj=np.asarray(obj_pos(env, old_obj), np.float64), dxy=dxy)
                    dg_info["snap"] = str(sdir / f"{spec['id']}.npz")
                dg_started = True
                aw_pending = True                # the teacher takes over from the state the student reached
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
            if aw_pending and spec.get("lens_gate_open"):
                # vigilance: does switching the instruction still steer the full policy here?  If so, no wormhole.
                el = make_element(obs, new_lang)
                el["bh"] = {"mode": "lens2", "probe": True, "prompt_old": old_lang,
                            "noise_seed": int(zlib.crc32(f"{seed}-{spec['init']}-gate-{t}".encode()) % 100000)}
                res = client.infer(el)
                pr = {k: float(v) for k, v in res["probe"].items()}
                if pr.get(spec.get("lens_gate_open"), 0.0) >= float(spec.get("lens_theta_open", 2.0)):
                    aw_pending = False
                    worm_log.append({"t": t, "skipped": True, **pr})
                else:
                    aw_gate = {"gate_" + k: v for k, v in pr.items()}
            if aw_pending or (method == "attn_worm" and tun is not None and aw_next is not None and t >= aw_next):
                first = aw_pending
                aw_pending = False
                t0 = time.time()
                vlm_info = None
                if spec.get("lens_target_src") == "vlm":
                    # attention from a grounding foundation model: the object the new instruction refers to
                    tg, tg_old, vlm_info = vlm_targets(env, obs, new_lang, old_lang, spec)
                    res = {}
                    m_new = np.zeros(1)
                elif spec.get("lens_target_src") == "oracle":
                    # diagnostic upper bound of the wormhole design: the true object position from the simulator
                    pn, po = obj_pos(env, new_obj), obj_pos(env, old_obj)
                    dz = float(spec.get("oracle_ztop", 0.04))
                    tg = {"xy": np.asarray(pn[:2], np.float64), "ztop": float(pn[2]) + dz}
                    tg_old = {"xy": np.asarray(po[:2], np.float64), "ztop": float(po[2]) + dz}
                    if spec.get("vlm_offset") is not None:
                        tg["xy"] = tg["xy"] + np.asarray(spec["vlm_offset"], np.float64)[:2]
                    vlm_info = {"oracle": True}
                    res = {}
                    m_new = np.zeros(1)
                else:
                    el = make_element(obs, new_lang)
                    el["bh"] = {"mode": "lens_target", "prompt_old": old_lang,
                                "noise_seed": int(zlib.crc32(f"{seed}-{spec['init']}-aw-{t}".encode()) % 100000)}
                    res = client.infer(el)
                    if spec.get("lens_learned") and res.get("M_new") is not None:
                        m_new = np.asarray(res["M_new"], np.float64)
                        m_old = -m_new
                    else:
                        m_new, m_old = attention_map(res["G_new"], res["G_old"], spec.get("lens_layers", "0-17"),
                                                     spec.get("lens_kind", "diff"))
                    frac = float(spec.get("lens_frac", 0.5))
                    tg = attention_target(env, obs, m_new, frac)
                    tg_old = attention_target(env, obs, m_old, frac)
                p = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                hover = float(spec.get("lens_hover", 0.08))
                goal = np.array([tg["xy"][0], tg["xy"][1], tg["ztop"] + hover])
                if spec.get("lens_hover_home"):      # exit at the height the robot starts from, above the attended region
                    goal[2] = max(goal[2], float(home_pos[2]) + float(spec.get("lens_hover_home_dz", 0.0)))
                true_new = obj_pos(env, new_obj) if new_obj else np.full(3, np.nan)
                entry = {"t": t, "target": [float(v) for v in goal], "target_old": [float(v) for v in tg_old["xy"]],
                         "n_patches": tg.get("n_patches", -1), "err_xy_true": float(np.linalg.norm(goal[:2] - true_new[:2])),
                         "z_top": tg["ztop"], "true_new": [float(v) for v in true_new],
                         "dmap_pos": float(np.clip(m_new, 0, None).sum()), "ms": (time.time() - t0) * 1000}
                if vlm_info is not None:
                    entry.update(vlm_info)
                infer_ms.append(entry["ms"])
                if first:
                    a_prev = traj["act"][-1].astype(np.float64) if traj["act"] else np.zeros(7)
                    clear_, lift_ = float(spec.get("lens_clear", 0.03)), float(spec.get("lens_lift", 0.05))
                    if spec.get("lens_corridor") is not None and new_obj and old_obj and traj["new_obj"]:
                        # lens16 (handover 7.15, round i): kinematics-aware corridor.  When the new object rests lower
                        # than the old one (B->A: from the far, elevated cabinet to the stove), lift straight up to
                        # lens_corridor m above the new object (at least lens_clear above the start), cross at that
                        # height, then go down to the hand-back point -- round h's corridor teacher (H = 0.36)
                        zn0, zo0 = float(traj["new_obj"][0][2]), float(traj["old_obj"][0][2])
                        if zn0 < zo0 - 0.05:
                            zn = float(obj_pos(env, new_obj)[2])
                            clear_ = max(clear_, zn + float(spec["lens_corridor"]) - max(float(p[2]), float(goal[2])))
                            lift_ = 1.0
                            entry["corridor_clear"] = clear_
                    tun = Tunnel(goal, home_quat, p, a_prev, clear=clear_,
                                 lift=lift_, vmax=float(spec.get("lens_vmax", 1.0)),
                                 rate=float(spec.get("lens_rate", 0.3)), arrive=float(spec.get("lens_arrive", 0.02)),
                                 max_steps=int(spec.get("lens_max", 90)))
                    aw = {"goal": goal, "old_xy": np.asarray(tg_old["xy"])}
                    entry["open"] = True
                    entry.update(aw_gate)
                else:
                    # attention consistency: a re-read moves the target only if it agrees with the current estimate
                    # (the arm can occlude the target during the traversal and the read then jumps elsewhere)
                    a = float(spec.get("lens_ema", 0.5))
                    if np.linalg.norm(goal[:2] - aw["goal"][:2]) < float(spec.get("lens_consist", 0.06)):
                        aw["goal"] = a * goal + (1 - a) * aw["goal"]
                        tun.retarget(aw["goal"])
                        entry["accepted"] = True
                    else:
                        entry["accepted"] = False
                    aw["old_xy"] = a * np.asarray(tg_old["xy"]) + (1 - a) * aw["old_xy"]
                    # hand over once the policy's own chunk follows the new attention and leaves the old one
                    ch = np.asarray(res["actions"], dtype=np.float64)
                    d = 1.21 * ch[:replan, :2].sum(0)
                    u_new = aw["goal"][:2] - p[:2]
                    u_old = aw["old_xy"] - p[:2]
                    nd = float(np.linalg.norm(d))
                    cn = float(d @ u_new / (nd * np.linalg.norm(u_new) + 1e-9))
                    co = float(d @ u_old / (nd * np.linalg.norm(u_old) + 1e-9))
                    entry.update({"cos_new": cn, "cos_old": co, "chunk_cm": nd, "phase": tun.phase})
                    if (spec.get("lens_handover", "arrive") == "engage" and nd >= 1.0
                            and cn >= float(spec.get("lens_tau", 0.8)) and cn - co >= float(spec.get("lens_delta", 0.3))):
                        entry["handover"] = "engage"
                        worm_log.append(entry)
                        worm_log[0].update({"steps": tun.k, "arrive_err": float(np.linalg.norm(p - tun.goal)),
                                            "handover": "engage", "t_handover": t})
                        tun = None
                        plan.clear()
                        aw_next = None
                        entry = None
                if entry is not None:
                    worm_log.append(entry)
                if tun is not None:
                    aw_next = t + int(spec.get("lens_recheck", 10))
            if (method == "attn_worm" and tun is not None and spec.get("lens_handover") == "vigil"
                    and t >= aw_vig_next):
                # vigilance hand-over: probe whether the instruction steers the full policy again on the way
                aw_vig_next = t + int(spec.get("lens_vigil_every", 5))
                el = make_element(obs, new_lang)
                el["bh"] = {"mode": "lens2", "probe": True, "prompt_old": old_lang,
                            "noise_seed": int(zlib.crc32(f"{seed}-{spec['init']}-vig-{t}".encode()) % 100000)}
                res_v = client.infer(el)
                pr = {k: float(v) for k, v in res_v["probe"].items()}
                worm_log.append({"t": t, "vigil": True, "phase": tun.phase, **pr})
                if (tun.k >= int(spec.get("lens_vigil_min", 10))
                        and pr.get(spec.get("lens_vigil_metric", "dchunk"), 0.0) >= float(spec.get("lens_theta_open", 2.0))):
                    p_now = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                    worm_log[0].update({"steps": tun.k, "arrive_err": float(np.linalg.norm(p_now - tun.goal)),
                                        "handover": "vigil", "t_handover": t})
                    tun = None
                    plan.clear()
                    aw_next = None
            if retreating and spec.get("retreat_vigil") and retreat_steps >= int(spec.get("lens_vigil_min", 6)) \
                    and retreat_steps % int(spec.get("lens_vigil_every", 5)) == 0:
                # vigilance-gated retreat: stop backing off as soon as the instruction steers the policy again
                el = make_element(obs, new_lang)
                el["bh"] = {"mode": "lens2", "probe": True, "prompt_old": old_lang,
                            "noise_seed": int(zlib.crc32(f"{seed}-{spec['init']}-rv-{t}".encode()) % 100000)}
                res_v = client.infer(el)
                pr = {k: float(v) for k, v in res_v["probe"].items()}
                worm_log.append({"t": t, "vigil": True, "retreat_steps": retreat_steps, **pr})
                if pr.get(spec.get("lens_vigil_metric", "dchunk"), 0.0) >= float(spec.get("lens_theta_open", 2.0)):
                    retreating = False
                    plan.clear()
                    worm_log.append({"t": t, "handover": "vigil", "retreat_steps": retreat_steps})
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
                hov = "arrive"
                if spec.get("lens_margin") is not None and method == "attn_worm":
                    # lens17 (handover 7.16, round j): kinematically safe wormhole exit -- hand back as soon as the
                    # scripted transit itself has straightened the elbow below lens_margin rad (privileged joint state)
                    em = elbow_margin(env)
                    if tun_m0 is None:
                        tun_m0 = em
                    tun_mmin = em if tun_mmin is None else min(tun_mmin, em)
                    if not done and em < float(spec["lens_margin"]) and em < tun_m0 - 0.05:
                        done, hov = True, "margin"
                if done:
                    worm_log[0 if method == "attn_worm" else -1].update(
                        {"steps": tun.k, "arrive_err": terr, "handover": hov, "t_handover": t})
                    if spec.get("lens_margin") is not None and method == "attn_worm":
                        worm_log[0].update({"margin_start": tun_m0, "margin_min": tun_mmin, "margin_handback": elbow_margin(env)})
                    tun = None
                    plan.clear()
                    aw_next = None
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
                    if switched and method == "lens" and lens_q < int(spec.get("lens_queries", 10 ** 6)):
                        v = spec.get("lens_variant", "cur")
                        lctl = {"mode": "lens", "prompt_old": old_lang, "beta": float(spec.get("lens_beta", 1.0)),
                                "nsamp_attn": int(spec.get("lens_nsamp", 2))}
                        if v == "cur":
                            lctl["variant"] = "cur"
                        elif v == "early":
                            lctl.update(variant="map", map=lens_map)
                        elif v == "oracle":
                            lctl.update(variant="map", map=oracle_map(env, old_obj, new_obj))
                        elif v == "learned":
                            lctl["variant"] = "learned"
                        else:
                            raise ValueError(v)
                        ctl.update(lctl)
                        lens_q += 1
                    if (switched and method == "lens_cut" and not lens_released
                            and lens_q < int(spec.get("lens_queries", 10 ** 6))):
                        ctl.update({"mode": "lens2", "off": list(spec.get("lens_off", [])),
                                    "ae_cut": list(spec.get("lens_aecut", []))})
                        if spec.get("lens_gate"):
                            ctl.update({"probe": True, "prompt_old": old_lang})
                        lens_q += 1
                    if switched and method == "attn_flow" and af is not None and af["active"]:
                        p = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                        if t >= af["next"]:
                            goal, _, _, entry = read_target(client, env, obs, new_lang, old_lang, spec,
                                                            f"{seed}-{spec['init']}-af-{t}", new_obj)
                            a = float(spec.get("lens_ema", 0.5))
                            if af["goal"] is None:
                                af["goal"] = goal
                            elif np.linalg.norm(goal[:2] - af["goal"][:2]) < float(spec.get("lens_consist", 0.06)):
                                af["goal"] = a * goal + (1 - a) * af["goal"]
                                entry["accepted"] = True
                            else:
                                entry["accepted"] = False
                            af["next"] = t + int(spec.get("lens_recheck", 10))
                            infer_ms.append(entry["ms"])
                            worm_log.append({"t": t, **entry})
                        dist = float(np.linalg.norm(af["goal"] - p))
                        if dist < float(spec.get("lens_arrive", 0.03)) or af["q"] >= int(spec.get("lens_queries", 40)):
                            af["active"] = False
                            worm_log.append({"t": t, "release": "arrive" if dist < float(spec.get("lens_arrive", 0.03))
                                             else "max", "dist": dist, "q": af["q"]})
                        else:
                            u = (af["goal"] - p) / max(dist, 1e-9)
                            ctl.update({"mode": "lens2", "off": list(spec.get("lens_off", [])),
                                        "ae_cut": list(spec.get("lens_aecut", [])),
                                        "guide": {"dir": [float(x) for x in u], "lam": float(spec.get("lens_lam", 0.5)),
                                                  "steps": int(spec.get("lens_gsteps", replan))}})
                            if spec.get("lens_gate"):
                                ctl.update({"probe": True, "prompt_old": old_lang})
                            af["q"] += 1
                            lens_q += 1
                    if switched and dg_mode and not dg_started and spec.get("dg_ctl"):
                        ctl.update(spec["dg_ctl"])           # student queries after the switch only
                    if switched and spec.get("aw_ctl"):
                        for kk, vv in spec["aw_ctl"].items():
                            if isinstance(vv, str):
                                vv = vv.replace("__OLD__", str(old_lang)).replace("__NEW__", str(new_lang))
                            ctl[kk] = vv
                    if el0 is None:
                        el0 = dict(el)
                    el["bh"] = ctl
                    t0 = time.time()
                    res = (teacher if (dg_mode and dg_started) else client).infer(el)
                    infer_ms.append((time.time() - t0) * 1000)
                    if isinstance(res, dict) and res.get("lens") is not None:
                        lens_log.append({"t": t, **{k: v for k, v in res["lens"].items()}})
                    if isinstance(res, dict) and res.get("probe") is not None:
                        pr = {k: float(v) for k, v in res["probe"].items()}
                        # vigilance gate: release the truncation once the instruction steers the full policy again
                        if pr.get(spec.get("lens_gate", "dchunk"), 0.0) >= float(spec.get("lens_theta", 2.0)):
                            lens_released = True
                            if af is not None and af["active"]:
                                af["active"] = False
                                worm_log.append({"t": t, "release": "gate", "q": af["q"], **pr})
                        lens_log.append({"t": t, "q": lens_q, **pr, "released": lens_released})
                    elif switched and method in ("lens_cut", "attn_flow") and ctl.get("mode") == "lens2":
                        lens_log.append({"t": t, "q": lens_q})
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
                if dg_mode and not dg_started:
                    chunk_q = -3 if switched else -4      # produced by the student: no teacher label
                    if switched:
                        dg_left -= 1
            if rec_steps:
                es = make_element(obs, prompt)
                stp["img"].append(es["observation/image"])
                stp["wrist"].append(es["observation/wrist_image"])
                stp["state"].append(es["observation/state"])
                stp["act"].append(np.asarray(act, dtype=np.float32))
                stp["src"].append(int(chunk_q))
                stp["sw"].append(1 if switched else 0)
                wp, wr, wf = camera(env, "robot0_eye_in_hand")
                stp["wcam_pos"].append(wp.astype(np.float32))
                stp["wcam_rot"].append(wr.astype(np.float32))
                stp["wfovy"].append(np.float32(wf))
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
    if rec_steps and stp and not dg_skip:     # lens15: a skipped episode has no label, its step record is not kept
        sdir = pathlib.Path(out_dir) / "steps"
        sdir.mkdir(parents=True, exist_ok=True)
        bp, br, bf = camera(env)
        np.savez_compressed(sdir / f"{spec['id']}.npz", **{k: np.stack(v) for k, v in stp.items()},
                            ee=arrs["ee"], old_obj=arrs["old_obj"], new_obj=arrs["new_obj"],
                            bcam_pos=bp, bcam_rot=br, bfovy=np.float32(bf), old_lang=str(old_lang),
                            new_lang=str(new_lang or ""), ts=-1 if ts is None else int(ts))
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
    if dg_skip:
        outcome = "skip"                 # lens15: no take-over (the student was not in transit), no label
    return {"outcome": outcome, "t_old": t_old, "t_new": t_new, "steps": t, "error": err, "switched": switched,
            "old_lang": old_lang, "new_lang": new_lang, "retreat_steps": retreat_steps,
            "infer_ms": float(np.mean(infer_ms)) if infer_ms else None, "n_queries": q_idx, "worm": worm_log,
            "lens": lens_log, "resume_err": resume_err, **dg_info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8810)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--teacher_port", type=int, default=0, help="teacher policy server of DAgger episodes (dg_k)")
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
    teacher = RobustClient(args.host, args.teacher_port) if args.teacher_port else None
    cache = SceneCache()
    t0 = time.time()
    for k, spec in enumerate(mine):
        t1 = time.time()
        r = run_episode(spec, client, cache, out, teacher)
        r.update({kk: spec.get(kk) for kk in ("id", "suite", "scene", "init", "old", "new", "ts", "method", "w", "beta", "seed", "tag", "pair", "variant")})
        for kk in ("dg_k", "resume", "branch_of"):          # lens15
            if spec.get(kk) is not None:
                r[kk] = spec.get(kk)
        if spec.get("grp") is not None:          # lens14: the self-play condition and its noise tag (make_specs_lens_r21)
            r.update(grp=spec.get("grp"), noise_tag=spec.get("noise_tag"))
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
