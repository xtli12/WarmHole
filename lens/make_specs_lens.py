"""Episode specs for the attention-lens experiments (spatial pair, both directions).

usage: make_specs_lens.py OUT_DIR [--test v1,v2] [--goal v1,v2]
Writes
  {bb}_lsmoke.jsonl  two episodes per variant (development initial states, committed switch) to check the code paths
  {bb}_ld.jsonl      development split: initial states 40-49, t_s in {2, 12, 22, 32, 42} (--ts to change)
  {bb}_lt.jsonl      test split: initial states 0-19, every t_s of Table 1 (only with --test, once choices are fixed)
  {bb}_lg.jsonl      LIBERO-Goal switches (only with --goal)
  {bb}_<name>.jsonl  with --dagger v1,v2: DAgger episodes of a self-trained wormhole policy on the TRAINING initial
                     states 20-39 only (never the development or test states), t_s from --dag_ts, every control step
                     recorded; the student policy acts first, the oracle wormhole teacher takes over (bh_client_lens13)
Initial states 40-49 were never used for the paper's tables; the test split is run once.
"""
import argparse
import json
import os

PAIRS = {
    "AB": dict(suite="libero_spatial", scene=7, old=7, pair="A->B",
               new_lang="pick up the black bowl on the wooden cabinet and place it on the plate",
               new_goal=[["on", "akita_black_bowl_2", "plate_1"]]),
    "BA": dict(suite="libero_spatial", scene=9, old=9, pair="B->A",
               new_lang="pick up the black bowl on the stove and place it on the plate",
               new_goal=[["on", "akita_black_bowl_2", "plate_1"]]),
}
WRIST = "left_wrist_0_rgb"


VLM_EST = "rimt"                    # box -> 3-D estimator of the grounding target (set with --vlm_est)
VLM_URL = "http://127.0.0.1:8500/ground"   # grounding server(s), comma-separated replicas (set with --vlm_url)
VLM_STRATEGY = "ref"                # grounding strategy of the server (set with --vlm_strategy)
VLM_OFFSET = {}                     # backbone -> [dx, dy] own grasp offset from the object centre (--vlm_offset);
                                    # or backbone -> {"A->B": [dx, dy], "B->A": [dx, dy]} (per direction)
VLM_OFFSET_SCENE = {}               # the same per direction (--vlm_offset_scene), used by the *_s variants
SAVE_SWITCH = False                 # store the agent view at the switch of every attn_worm episode (--save_switch)
REC_STEPS = False                   # record every control step of every episode (--rec_steps)
THETA = {"pi05": 2.0, "pi0": 6.0}   # engagement (dchunk) of an attentive policy: pi0.5 ~2.7 vs ~1 when captured;
                                   # pi0 (50-step chunks) ~6-10 vs ~4.5 when captured (stage 0b)


def variants(bb="pi05"):
    cut = lambda q, **kw: dict(method="lens_cut", lens_off=[WRIST], lens_queries=q, **kw)  # noqa: E731
    _wo15 = dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.15,
                 lens_handover="arrive", vlm_offset=VLM_OFFSET.get(bb))          # = wo_off_h15 (the teacher)
    aw = dict(method="attn_worm", lens_layers="8-12", lens_kind="diff", lens_frac=0.5, lens_hover=0.08,
              lens_recheck=10, lens_handover="arrive")
    return {
        "vanilla": dict(method="vanilla"),
        "flush": dict(method="flush"),
        # r22: contrastive instruction guidance on a trained policy (aw_server_c9; the client re-queries at the switch
        # like flush and sends mode cig with the old instruction on every query after it)
        "cig05": dict(method="cig", w=0.5),
        "cig10": dict(method="cig", w=1.0),
        "cut_w2": cut(2),
        "cut_w4": cut(4),
        "cut_w8": cut(8),
        "cut_wall": cut(1000),
        "cut_wg": cut(12, lens_gate="dchunk", lens_theta=THETA[bb]),  # truncation released by the vigilance gate
        "cut_ws4": dict(method="lens_cut", lens_off=[WRIST], lens_aecut=["state"], lens_queries=4),  # pi0 only
        "aw": aw,
        "aw7": dict(aw, lens_frac=0.7),
        "aw_eng": dict(aw, lens_frac=0.7, lens_handover="engage"),
        # round 2: re-reads must agree with the current target (client default lens_consist = 6 cm)
        "awc": dict(aw, lens_frac=0.7, lens_consist=0.06),
        "awc_eng": dict(aw, lens_frac=0.7, lens_consist=0.06, lens_handover="engage"),
        "awc_once": dict(aw, lens_frac=0.7, lens_recheck=1000),            # target read once at the switch
        # attention-guided flow: the policy generates the path, egocentric camera truncated, pulled to the target
        "af": dict(method="attn_flow", lens_off=[WRIST], lens_layers="8-12", lens_kind="diff", lens_frac=0.7,
                   lens_hover=0.08, lens_recheck=10, lens_consist=0.06, lens_lam=0.5, lens_gsteps=5,
                   lens_arrive=0.03, lens_queries=40),
        "af_g": dict(method="attn_flow", lens_off=[WRIST], lens_layers="8-12", lens_kind="diff", lens_frac=0.7,
                     lens_hover=0.08, lens_recheck=10, lens_consist=0.06, lens_lam=0.5, lens_gsteps=5,
                     lens_arrive=0.03, lens_queries=40, lens_gate="dchunk", lens_theta=THETA[bb]),
        "af_nocut": dict(method="attn_flow", lens_off=[], lens_layers="8-12", lens_kind="diff", lens_frac=0.7,
                         lens_hover=0.08, lens_recheck=10, lens_consist=0.06, lens_lam=0.5, lens_gsteps=5,
                         lens_arrive=0.03, lens_queries=40),
        # round 3: the learned attention lens (head readout trained on the robot's own grasps) gives the target
        "awL": dict(aw, lens_frac=0.7, lens_consist=0.06, lens_learned=True),
        "afL": dict(method="attn_flow", lens_off=[WRIST], lens_frac=0.7, lens_hover=0.08, lens_recheck=10,
                    lens_consist=0.06, lens_lam=0.5, lens_gsteps=5, lens_arrive=0.03, lens_queries=40, lens_learned=True),
        # round 4: attention-directed exit at the start height (the policy re-plans from above), vigilance gate
        "awh": dict(aw, lens_frac=0.7, lens_consist=0.06, lens_hover_home=True),
        "awh_g": dict(aw, lens_frac=0.7, lens_consist=0.06, lens_hover_home=True, lens_gate_open="dchunk",
                      lens_theta_open=THETA[bb]),
        "awh_eng": dict(aw, lens_frac=0.7, lens_consist=0.06, lens_hover_home=True, lens_handover="engage"),
        "awc_g": dict(aw, lens_frac=0.7, lens_consist=0.06, lens_gate_open="dchunk", lens_theta_open=THETA[bb]),
        # round 5: the attended target comes from a grounding foundation model (Qwen2.5-VL, ground_server.py)
        "vw": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.08,
                   lens_handover="arrive", vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL, vlm_strategy=VLM_STRATEGY),
        "vw_g": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.08,
                     lens_handover="arrive", vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL, vlm_strategy=VLM_STRATEGY, lens_gate_open="dchunk",
                     lens_theta_open=THETA[bb]),
        "vw_h12": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.12,
                       lens_handover="arrive", vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL, vlm_strategy=VLM_STRATEGY),
        # the hover point is where the policy itself grasps such objects (offset from its own successful rollouts)
        "vw_off": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.08,
                       lens_handover="arrive", vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL, vlm_strategy=VLM_STRATEGY,
                       vlm_offset=VLM_OFFSET.get(bb)),
        # handover-design study with the true target (diagnostic upper bound, development split only)
        "wo": dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.08,
                   lens_handover="arrive"),
        "wo_off": dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.08,
                       lens_handover="arrive", vlm_offset=VLM_OFFSET.get(bb)),
        "wo_off_h15": dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.15,
                           lens_handover="arrive", vlm_offset=VLM_OFFSET.get(bb)),
        "wo_v": dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.08,
                     lens_handover="vigil", lens_vigil_every=5, lens_vigil_min=10, lens_theta_open=THETA[bb]),
        "wo_off_v": dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.08,
                         lens_handover="vigil", lens_vigil_every=5, lens_vigil_min=10, lens_theta_open=THETA[bb],
                         vlm_offset=VLM_OFFSET.get(bb)),
        "rt": dict(method="retreat"),
        "rv": dict(method="retreat", retreat_vigil=True, lens_vigil_every=5, lens_vigil_min=6,
                   lens_theta_open=THETA[bb]),
        # round 5b: grounded target, exit above the policy's own grasp point (offset from its own rollouts), 15 cm
        "vwo": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.15,
                    lens_handover="arrive", vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL, vlm_strategy=VLM_STRATEGY,
                    vlm_offset=VLM_OFFSET.get(bb)),
        "vwo_g": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.15,
                      lens_handover="arrive", vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL,
                      vlm_strategy=VLM_STRATEGY, vlm_offset=VLM_OFFSET.get(bb), lens_gate_open="dchunk",
                      lens_theta_open=THETA[bb]),
        "vwo_v": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.15,
                      lens_handover="vigil", lens_vigil_every=5, lens_vigil_min=10, lens_theta_open=THETA[bb],
                      vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL, vlm_strategy=VLM_STRATEGY,
                      vlm_offset=VLM_OFFSET.get(bb)),
        "vwo_gv": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.15,
                       lens_handover="vigil", lens_vigil_every=5, lens_vigil_min=10, lens_gate_open="dchunk",
                       lens_theta_open=THETA[bb], vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL,
                       vlm_strategy=VLM_STRATEGY, vlm_offset=VLM_OFFSET.get(bb)),
        # round 6 (dev only): contrastive grounding -- the new and the old instruction are grounded separately; if
        # both point at the same object, the new one is grounded again (ground_server.py, strategies contrast*)
        "vwc": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.15,
                    lens_handover="arrive", vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL, vlm_strategy="contrast",
                    vlm_offset=VLM_OFFSET.get(bb)),
        "vwcv": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.15,
                     lens_handover="arrive", vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL, vlm_strategy="contrast_v",
                     vlm_offset=VLM_OFFSET.get(bb)),
        "vwcp": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.15,
                     lens_handover="arrive", vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL, vlm_strategy="contrast_p",
                     vlm_offset=VLM_OFFSET.get(bb)),
        # round 6 (dev only): hand-over study with the true target, own grasp offset per direction (where the policy
        # grasps that object in that scene in its own no-switch rollouts), different exit heights
        "wo_s15": dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.15,
                       lens_handover="arrive", vlm_offset=VLM_OFFSET_SCENE.get(bb)),
        "wo_s25": dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.25,
                       lens_handover="arrive", vlm_offset=VLM_OFFSET_SCENE.get(bb)),
        "wo_shome": dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.15,
                         lens_handover="arrive", lens_hover_home=True, vlm_offset=VLM_OFFSET_SCENE.get(bb)),
        "wo_off_h25": dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.25,
                           lens_handover="arrive", vlm_offset=VLM_OFFSET.get(bb)),
        "wo_off_home": dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.15,
                            lens_handover="arrive", lens_hover_home=True, vlm_offset=VLM_OFFSET.get(bb)),
        # learned attention lens (head readout trained on the policy's own grasp points): the policy's own attention
        # difference gives the exit point, read once at the switch, 15 cm above it (no external grounding)
        "awL15": dict(method="attn_worm", lens_layers="8-12", lens_kind="diff", lens_frac=0.7, lens_hover=0.15,
                      lens_recheck=1000, lens_handover="arrive", lens_learned=True),
        "wo_off_h15_g": dict(method="attn_worm", lens_target_src="oracle", lens_recheck=1000, lens_hover=0.15,
                             lens_handover="arrive", vlm_offset=VLM_OFFSET.get(bb), lens_gate_open="dchunk",
                             lens_theta_open=THETA[bb]),
        # vigilance closes the wormhole as well: hand over once the instruction steers the policy again
        "vw_v": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.08,
                     lens_handover="vigil", lens_vigil_every=5, lens_vigil_min=10, lens_theta_open=THETA[bb],
                     vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL, vlm_strategy=VLM_STRATEGY),
        "vw_gv": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.08,
                      lens_handover="vigil", lens_vigil_every=5, lens_vigil_min=10, lens_gate_open="dchunk",
                      lens_theta_open=THETA[bb], vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL,
                      vlm_strategy=VLM_STRATEGY),
        "vw_off_g": dict(method="attn_worm", lens_target_src="vlm", lens_recheck=1000, lens_hover=0.08,
                         lens_handover="arrive", vlm_res=512, vlm_est=VLM_EST, vlm_url=VLM_URL,
                         vlm_strategy=VLM_STRATEGY, vlm_offset=VLM_OFFSET.get(bb), lens_gate_open="dchunk",
                         lens_theta_open=THETA[bb]),
        # self-trained wormhole policy (aw_server.py): plain re-query on the switch, and inference-time diagnostics
        # DAgger (bh_client_lens13 with --teacher_port): the student acts before the switch and dg_k control steps after
        # it, then the teacher (the oracle wormhole wo_off_h15 + the released policy) takes over from where the student
        # got to.  dgb*: the student acts blind (its wormhole closed), i.e. drifts toward its no-instruction default.
        "dg10": dict(_wo15, dg_k=10),
        "dg30": dict(_wo15, dg_k=30),
        "dgb20": dict(_wo15, dg_k=20, dg_ctl={"worm_scale": 0.0, "hide_loc": True}),
        # r23 (handover 7.13, round g): conditional take-over at the student's own attractor states (bh_client_lens15
        # dg_cond; dg_k is set per episode), and the counterfactual branch from the stored state (the other instruction,
        # the teacher at once)
        "cfdg": dict(_wo15, dg_cond=dict(dxy=0.07, freeze_min=20, freeze_win=12, freeze_path=0.012), max_steps=400),
        "cfbr": dict(_wo15, dg_k=0, max_steps=300),
        # r24 (handover 7.15, round i): switch-state relabeling -- the student acts until the switch, the teacher takes
        # over AT the switch (dg_k 0) from the student's own state; kinematics-aware corridor in B->A (bh_client_lens16)
        "dg0c": dict(_wo15, dg_k=0, lens_corridor=0.36, max_steps=300),
        "aw_off": dict(method="flush", aw_ctl={"worm_scale": 0.0, "hide_loc": True}),     # wormhole closed
        "aw_bias0": dict(method="flush", aw_ctl={"worm_scale": 0.0}),                     # attention bias removed
        "aw_hl": dict(method="flush", aw_ctl={"hide_loc": True}),                         # [LOC] key/value hidden
        "aw_tn": dict(method="flush", aw_keep_prompt=True, aw_ctl={"donor_prompt": "__NEW__"}),  # words old, wormhole new
        "aw_to": dict(method="flush", aw_ctl={"donor_prompt": "__OLD__"}),                # words new, wormhole old
        # language-pathway probes after the switch: instruction words hidden from the action expert (ht), image
        # tokens not reading the instruction in the backbone (int), both (only [LOC] can carry the instruction)
        "aw_ht": dict(method="flush", aw_ctl={"hide_text": True}),
        "aw_ht_off": dict(method="flush", aw_ctl={"hide_text": True, "hide_loc": True, "worm_scale": 0.0}),
        "aw_int": dict(method="flush", aw_ctl={"img_no_text": True}),
        "aw_int_ht": dict(method="flush", aw_ctl={"img_no_text": True, "hide_text": True}),
        "af_l1": dict(method="attn_flow", lens_off=[WRIST], lens_layers="8-12", lens_kind="diff", lens_frac=0.7,
                      lens_hover=0.08, lens_recheck=10, lens_consist=0.06, lens_lam=1.0, lens_gsteps=5,
                      lens_arrive=0.03, lens_queries=40),
    }


def episodes(bb, tag, variant, inits, tss, video_inits=()):
    out = []
    v = variants(bb)[variant]
    for d, P in PAIRS.items():
        for ts in tss:
            for i in inits:
                s = dict(P)
                s.update(id=f"{bb}_{tag}_sp{d}_{variant}_ts{ts}_i{i}", init=i, seed=7, ts=ts, tag=tag, variant=variant)
                s.update(v)
                if isinstance(s.get("vlm_offset"), dict):          # own grasp offset given per direction
                    s["vlm_offset"] = s["vlm_offset"][P["pair"]]
                if SAVE_SWITCH:            # (before r18 only attn_worm episodes stored the switch view)
                    s["save_switch"] = True
                if REC_STEPS:
                    s["rec_steps"] = True
                if i in video_inits:
                    s["video"] = True
                out.append(s)
    return out


GOAL_PAIRS = [(0, 7), (1, 8), (2, 6), (6, 2), (7, 0), (8, 1), (8, 9), (9, 8)]


def goal_episodes(bb, tag, variant, inits, tss):
    out = []
    v = variants(bb)[variant]
    for old, new in GOAL_PAIRS:
        for ts in tss:
            for i in inits:
                s = dict(suite="libero_goal", scene=old, old=old, new=new, pair=f"goal:{old}->{new}",
                         id=f"{bb}_{tag}_goal_{old}to{new}_{variant}_ts{ts}_i{i}", init=i, seed=7, ts=ts, tag=tag,
                         variant=variant)
                s.update(v)
                out.append(s)
    return out


def recordings(bb, inits):
    """No-switch rollouts under each instruction of each scene with the policy inputs recorded (training data of the
    learned attention lens; initial states outside the development and test splits)."""
    out = []
    for d, P in PAIRS.items():
        for which in ("old", "new"):
            for i in inits:
                s = dict(suite=P["suite"], scene=P["scene"], pair=P["pair"] + ":" + which, id=f"{bb}_lrec_sp{d}_{which}_i{i}",
                         init=i, seed=7, method="vanilla", record=True, tag="lrec", variant="rec_" + which)
                if which == "old":
                    s["old"] = P["old"]
                else:
                    s.update(old_lang=P["new_lang"], old_goal=P["new_goal"])
                out.append(s)
    return out


TEACH_TS = list(range(2, 63, 5))     # switch steps of the teacher data (2, 7, ..., 62)


def teacher_data(bb, inits=range(20, 40), seeds=(7, 8)):
    """Teacher data of the self-trained wormhole policy (initial states 20-39 only, never the development or the test
    split): switch episodes in both directions at every t_s of TEACH_TS, in which the existing oracle wormhole
    (wo_off_h15: the true position of the new object, the policy's own grasp offset, exit 15 cm above, hand-back on
    arrival) produces the post-switch behaviour, plus no-switch rollouts of the policy under each instruction of each
    scene.  Every control step is recorded (rec_steps).  Only successful episodes are used for training."""
    out = []
    v = variants(bb)["wo_off_h15"]
    for d, P in PAIRS.items():
        for ts in TEACH_TS:
            for i in inits:
                s = dict(P)
                s.update(id=f"{bb}_teach_sp{d}_wo_off_h15_ts{ts}_i{i}", init=i, seed=7, ts=ts, tag="teach",
                         variant="wo_off_h15", rec_steps=True)
                s.update(v)
                out.append(s)
        for which in ("old", "new"):
            for i in inits:
                for sd in (seeds if which == "old" else seeds[:1]):
                    s = dict(suite=P["suite"], scene=P["scene"], pair=P["pair"] + ":" + which, init=i, seed=sd,
                             id=f"{bb}_teach_sp{d}_{which}_s{sd}_i{i}", method="vanilla", rec_steps=True, tag="teach",
                             variant="plain_" + which)
                    if which == "old":
                        s["old"] = P["old"]
                    else:
                        s.update(old_lang=P["new_lang"], old_goal=P["new_goal"])
                    out.append(s)
    return out


def dagger_data(bb, name, names, tss, inits=range(20, 40)):
    """DAgger episodes (training initial states only), every control step recorded."""
    out = []
    for v in names:
        assert "dg_k" in variants(bb)[v], v
        for e in episodes(bb, name, v, inits, tss):
            e["rec_steps"] = True
            out.append(e)
    return out


def plain_dev(bb, tag, inits=range(40, 50)):
    """No-switch rollouts on the development states: each scene under its own instruction and under the other
    instruction of the pair (checks that a fine-tuned policy keeps its ordinary behaviour)."""
    out = []
    for d, P in PAIRS.items():
        for which in ("old", "new"):
            for i in inits:
                s = dict(suite=P["suite"], scene=P["scene"], pair=P["pair"] + ":" + which, init=i, seed=7,
                         id=f"{bb}_{tag}_sp{d}_{which}_i{i}", method="vanilla", tag=tag, variant="plain_" + which)
                if which == "old":
                    s["old"] = P["old"]
                else:
                    s.update(old_lang=P["new_lang"], old_goal=P["new_goal"])
                out.append(s)
    return out


RLAYOUT = 10000      # r21 (r22: + variants cig05 / cig10, devR with --devr_variant; r23: + cfdg / cfbr, --cfdag, --cfbranch; r24: + dg0c, --swdag; r25: + --teach_m): init >= RLAYOUT = procedurally sampled layout (bh_client_lens.py), seed init - RLAYOUT
SP_MAX_SEED = 499    # r21: self-play layout seeds are < 500, devR layout seeds >= 500 (never the same layout)


def layouts_arg(s):
    """'20-37,r101-124' -> [20, ..., 37, 10101, ..., 10124] (r = procedural layout seeds)."""
    out = []
    for part in [x.strip() for x in s.split(",") if x.strip()]:
        r = part.startswith("r")
        part = part[1:] if r else part
        a, _, b = part.partition("-")
        xs = range(int(a), int(b or a) + 1)
        out += [RLAYOUT + x if r else x for x in xs]
    return out


def selfplay(bb, name, layouts, tss, k):
    """r21: switch self-play of a trained policy (method flush = plain inference, as in the evaluation): training
    layouts only (official initial states 20-37 -- 38/39 are the validation states -- and procedural layouts with seeds
    below 500), both directions, every t_s of tss, k rollouts per condition that differ only in the policy noise
    (noise_tag sp1..spk); every control step recorded (steps/<id>.npz).  grp = the condition (direction, t_s, layout)."""
    for i in layouts:
        assert (20 <= i <= 37) or (RLAYOUT <= i <= RLAYOUT + SP_MAX_SEED), f"layout {i} not allowed for self-play"
    out = []
    v = variants(bb)["flush"]
    for d, P in PAIRS.items():
        for ts in tss:
            for i in layouts:
                for n in range(1, k + 1):
                    s = dict(P)
                    s.update(id=f"{bb}_{name}_sp{d}_flush_ts{ts}_i{i}_n{n}", init=i, seed=7, ts=ts, tag=name,
                             variant="flush", noise_tag=f"sp{n}", rec_steps=True, grp=f"{d}_ts{ts}_i{i}")
                    s.update(v)
                    out.append(s)
    return out


def devr(bb, name, layouts, tss, variant="flush"):
    """r21: development split on procedural layouts (seeds >= 500, disjoint from the self-play layouts and from every
    official initial state): both directions, every t_s of tss, one rollout per condition.  r22: any variant
    (default flush, as in r21)."""
    for i in layouts:
        assert i >= RLAYOUT + SP_MAX_SEED + 1, f"layout {i} not allowed for devR"
    return episodes(bb, name, variant, layouts, tss)


def cfdag(bb, name, layouts, tss, ks, nnoise, dirs=("BA",)):
    """r23 (handover 7.13): conditional DAgger at the student's own attractor states -- training layouts only (as
    self-play: official 20-37, procedural seeds < 500), the given directions (B->A), every t_s of tss, every take-over
    step k of ks (dg_k) and nnoise policy noises (noise_tag dg1..); every control step recorded.  The student drives
    after the switch; the teacher (wo_off_h15) takes over at k steps after it or earlier when the student is frozen, but
    only while the student is still in transit (bh_client_lens15 dg_cond); the simulator state of a take-over is stored
    for the counterfactual branch."""
    for i in layouts:
        assert (20 <= i <= 37) or (RLAYOUT <= i <= RLAYOUT + SP_MAX_SEED), f"layout {i} not allowed for round g"
    out = []
    v = variants(bb)["cfdg"]
    assert v.get("vlm_offset") is not None, "--vlm_offset is required (the teacher's own grasp offset)"
    for d in dirs:
        P = PAIRS[d]
        for ts in tss:
            for k in ks:
                for i in layouts:
                    for n in range(1, nnoise + 1):
                        s = dict(P)
                        s.update(id=f"{bb}_{name}_sp{d}_cfdg_k{k}_ts{ts}_i{i}_n{n}", init=i, seed=7, ts=ts, tag=name,
                                 variant="cfdg", noise_tag=f"dg{n}", rec_steps=True, grp=f"{d}_ts{ts}_i{i}")
                        s.update(v)
                        s["dg_k"] = int(k)
                        out.append(s)
    return out


def swdag(bb, name, layouts, tss, noise):
    """r24 (handover 7.15, round i): switch-state relabeling on the training layouts only (as self-play: official
    20-37, procedural seeds < 500).  noise = {direction: number of policy noises} (noise_tag sw1..); every t_s of tss;
    every control step recorded.  The student executes the OLD instruction until t_s; at t_s the teacher (wo_off_h15
    with the B->A corridor, variant dg0c) takes over from the student's own state."""
    for i in layouts:
        assert (20 <= i <= 37) or (RLAYOUT <= i <= RLAYOUT + SP_MAX_SEED), f"layout {i} not allowed for round i"
    out = []
    v = variants(bb)["dg0c"]
    assert v.get("vlm_offset") is not None, "--vlm_offset is required (the teacher's own grasp offset)"
    for d, nn in noise.items():
        P = PAIRS[d]
        for ts in tss:
            for i in layouts:
                for n in range(1, nn + 1):
                    s = dict(P)
                    s.update(id=f"{bb}_{name}_sp{d}_dg0c_ts{ts}_i{i}_n{n}", init=i, seed=7, ts=ts, tag=name,
                             variant="dg0c", noise_tag=f"sw{n}", rec_steps=True, grp=f"{d}_ts{ts}_i{i}")
                    s.update(v)
                    out.append(s)
    return out


def teach_margin(bb, name, margin, inits, tss, dirs=("AB",), with_orig=False):
    """r25 (handover 7.16, round j): teacher episodes with the kinematically safe wormhole exit -- wo_off_h15 plus
    lens_margin (bh_client_lens17: the scripted transit hands back as soon as it has straightened the elbow below
    `margin` rad).  Same conditions as the teacher data (seed 7, every control step recorded); ids
    {bb}_{name}_sp{d}_wo_off_h15m{10*margin}_ts{ts}_i{i}.  with_orig: also the original wo_off_h15 episodes (for the
    reproduction check of the smoke)."""
    for i in inits:
        assert (20 <= i <= 39) or (RLAYOUT <= i <= RLAYOUT + SP_MAX_SEED), f"layout {i} not allowed for teacher data"
    out = []
    base = variants(bb)["wo_off_h15"]
    assert base.get("vlm_offset") is not None, "--vlm_offset is required (the teacher's own grasp offset)"
    vname = f"wo_off_h15m{int(round(10 * margin))}"
    arms = [(vname, dict(base, lens_margin=float(margin)))] + ([("wo_off_h15", dict(base))] if with_orig else [])
    for d in dirs:
        P = PAIRS[d]
        for ts in tss:
            for i in inits:
                for vn, v in arms:
                    s = dict(P)
                    s.update(id=f"{bb}_{name}_sp{d}_{vn}_ts{ts}_i{i}", init=i, seed=7, ts=ts, tag=name, variant=vn,
                             rec_steps=True)
                    s.update(v)
                    out.append(s)
    return out


def cfbranch(bb, name, runs, main):
    """r23 (handover 7.13): the counterfactual branch of every take-over of the conditional DAgger run <main>: the same
    scene, the stored simulator state (resume), the instruction switched back to the OLD one at once (ts 0, dg_k 0:
    the teacher takes over at the first step and carries out the old task).  old / new are swapped: new = the old task
    of the pair, old = the pair's new instruction (whose goal, if reached first, counts as capture)."""
    import glob
    out = []
    rows = {}
    for d in sorted(glob.glob(os.path.join(runs, f"{bb}_{main}_p*"))):
        for p in sorted(glob.glob(os.path.join(d, "results_*.jsonl"))):
            for ln in open(p):
                try:
                    r = json.loads(ln)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("error") or r.get("dg_reason") not in ("k", "freeze") or not r.get("snap"):
                    continue
                if os.path.exists(r["snap"]):
                    rows[r["id"]] = r
    v = variants(bb)["cfbr"]
    assert v.get("vlm_offset") is not None, "--vlm_offset is required (the teacher's own grasp offset)"
    for rid, r in sorted(rows.items()):
        d = [k for k, P in PAIRS.items() if P["pair"] == r["pair"]][0]
        P = PAIRS[d]
        s = dict(suite=P["suite"], scene=P["scene"], new=P["old"], old_lang=P["new_lang"], old_goal=P["new_goal"],
                 pair=P["pair"] + ":cf")
        tail = rid.split(f"_{main}_", 1)[1]
        s.update(id=f"{bb}_{name}_{tail}_cf", init=int(r["init"]), seed=int(r.get("seed") or 7), ts=0, tag=name,
                 variant="cfbr", noise_tag=str(r.get("noise_tag") or "") + "cf", rec_steps=True, grp=r.get("grp"),
                 resume=r["snap"], branch_of=rid)
        s.update(v)
        out.append(s)
    return out


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(path, len(rows))


def main():
    global VLM_EST, VLM_URL, VLM_STRATEGY, VLM_OFFSET, VLM_OFFSET_SCENE, SAVE_SWITCH, REC_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dev", default="", help="comma list of variants for the development split (default: preset)")
    ap.add_argument("--name", default="ld", help="name of the development spec file (ld, ld2, ...)")
    ap.add_argument("--bbs", default="pi05,pi0")
    ap.add_argument("--rec", action="store_true", help="only write the recording specs {bb}_lrec.jsonl (inits 20-39)")
    ap.add_argument("--test", default="")
    ap.add_argument("--goal", default="")
    ap.add_argument("--vlm_est", default=VLM_EST, help="lens_geom.bbox_target method for the grounding variants")
    ap.add_argument("--vlm_url", default=VLM_URL, help="grounding server URL(s), comma-separated")
    ap.add_argument("--vlm_strategy", default=VLM_STRATEGY)
    ap.add_argument("--vlm_offset", default="", help='JSON {"pi05": [dx, dy], "pi0": [dx, dy]} (metres)')
    ap.add_argument("--vlm_offset_scene", default="",
                    help='JSON {"pi05": {"A->B": [dx, dy], "B->A": [dx, dy]}, ...} (metres), for the *_s variants')
    ap.add_argument("--ts", default="2,12,22,32,42", help="switch steps of the development split")
    ap.add_argument("--save_switch", action="store_true")
    ap.add_argument("--rec_steps", action="store_true", help="record every control step (steps/<id>.npz)")
    ap.add_argument("--teach", action="store_true", help="only write the teacher-data specs {bb}_teach.jsonl")
    ap.add_argument("--plain", default="", help="only write no-switch development specs {bb}_<name>.jsonl")
    ap.add_argument("--dagger", default="", help="only write DAgger specs {bb}_<name>.jsonl (variants dg*, inits 20-39)")
    ap.add_argument("--dag_ts", default="12,22,32,42,52,62", help="switch steps of the DAgger episodes")
    ap.add_argument("--selfplay", default="", help="r21: only write switch self-play specs {bb}_<name>.jsonl")
    ap.add_argument("--sp_layouts", default="20-37,r101-124", help="r21: self-play layouts (r = procedural seeds < 500)")
    ap.add_argument("--sp_k", type=int, default=3, help="r21: rollouts per self-play condition (noise tags)")
    ap.add_argument("--sp_ts", default="12,22,32,42,52,62", help="r21: switch steps of the self-play episodes")
    ap.add_argument("--devr", default="", help="r21: only write devR specs {bb}_<name>.jsonl (procedural layouts)")
    ap.add_argument("--devr_layouts", default="r501-520", help="r21: devR layouts (procedural seeds >= 500)")
    ap.add_argument("--devr_ts", default="12,22,32,42,52,62", help="r21: switch steps of devR")
    ap.add_argument("--devr_variant", default="flush", help="r22: variant of the devR episodes")
    ap.add_argument("--cfdag", default="", help="r23: only write round-g conditional DAgger specs {bb}_<name>.jsonl")
    ap.add_argument("--cf_layouts", default="20-37,r101-124", help="r23: layouts of the conditional DAgger episodes")
    ap.add_argument("--cf_ts", default="12,22,32,42,52,62", help="r23: switch steps")
    ap.add_argument("--cf_ks", default="40,70,100", help="r23: take-over steps after the switch")
    ap.add_argument("--cf_noise", type=int, default=2, help="r23: policy noises per condition")
    ap.add_argument("--cf_dirs", default="BA", help="r23: directions (keys of PAIRS)")
    ap.add_argument("--cfbranch", default="", help="r23: only write the counterfactual branch {bb}_<name>.jsonl")
    ap.add_argument("--cf_from", default="", help="r23: the conditional DAgger run whose take-overs are branched")
    ap.add_argument("--cf_runs", default=(os.environ.get("BH", "/data/blackhole") + "/runs/lens"), help="r23: result root of --cf_from")
    ap.add_argument("--swdag", default="", help="r24: only write round-i switch-state relabeling specs {bb}_<name>.jsonl")
    ap.add_argument("--sw_layouts", default="20-37,r101-124", help="r24: layouts (training only)")
    ap.add_argument("--sw_ts", default="12,17,22,27,32,37,42,47,52,57,62", help="r24: switch steps")
    ap.add_argument("--sw_noise", default="AB:1,BA:2", help="r24: policy noises per direction (keys of PAIRS)")
    ap.add_argument("--teach_m", default="", help="r25: only write round-j teacher specs {bb}_<name>.jsonl (lens_margin)")
    ap.add_argument("--m_margin", type=float, default=1.0, help="r25: elbow margin (rad) of the safe exit")
    ap.add_argument("--m_inits", default="20-39", help="r25: layouts (official 20-39, or r<seed> procedural)")
    ap.add_argument("--m_ts", default=",".join(str(x) for x in TEACH_TS), help="r25: switch steps")
    ap.add_argument("--m_dirs", default="AB", help="r25: directions (keys of PAIRS)")
    ap.add_argument("--m_orig", action="store_true", help="r25: also the original wo_off_h15 episodes")
    a = ap.parse_args()
    VLM_EST, VLM_URL, VLM_STRATEGY, SAVE_SWITCH = a.vlm_est, a.vlm_url, a.vlm_strategy, a.save_switch
    REC_STEPS = a.rec_steps
    if a.vlm_offset:
        VLM_OFFSET = {k: [float(x) for x in v] for k, v in json.loads(a.vlm_offset).items()}
    if a.vlm_offset_scene:
        VLM_OFFSET_SCENE = {k: {d: [float(x) for x in xy] for d, xy in v.items()}
                            for k, v in json.loads(a.vlm_offset_scene).items()}
    dev_ts = [int(x) for x in a.ts.split(",")]
    os.makedirs(a.out, exist_ok=True)
    if a.cfdag:
        for bb in a.bbs.split(","):
            write(os.path.join(a.out, f"{bb}_{a.cfdag}.jsonl"),
                  cfdag(bb, a.cfdag, layouts_arg(a.cf_layouts), [int(x) for x in a.cf_ts.split(",")],
                        [int(x) for x in a.cf_ks.split(",")], a.cf_noise, a.cf_dirs.split(",")))
        return
    if a.teach_m:
        for bb in a.bbs.split(","):
            write(os.path.join(a.out, f"{bb}_{a.teach_m}.jsonl"),
                  teach_margin(bb, a.teach_m, a.m_margin, layouts_arg(a.m_inits), [int(x) for x in a.m_ts.split(",")],
                               a.m_dirs.split(","), a.m_orig))
        return
    if a.swdag:
        noise = {x.split(":")[0]: int(x.split(":")[1]) for x in a.sw_noise.split(",") if x}
        for bb in a.bbs.split(","):
            write(os.path.join(a.out, f"{bb}_{a.swdag}.jsonl"),
                  swdag(bb, a.swdag, layouts_arg(a.sw_layouts), [int(x) for x in a.sw_ts.split(",")], noise))
        return
    if a.cfbranch:
        for bb in a.bbs.split(","):
            write(os.path.join(a.out, f"{bb}_{a.cfbranch}.jsonl"), cfbranch(bb, a.cfbranch, a.cf_runs, a.cf_from))
        return
    if a.selfplay:
        for bb in a.bbs.split(","):
            write(os.path.join(a.out, f"{bb}_{a.selfplay}.jsonl"),
                  selfplay(bb, a.selfplay, layouts_arg(a.sp_layouts), [int(x) for x in a.sp_ts.split(",")], a.sp_k))
        return
    if a.devr:
        for bb in a.bbs.split(","):
            write(os.path.join(a.out, f"{bb}_{a.devr}.jsonl"),
                  devr(bb, a.devr, layouts_arg(a.devr_layouts), [int(x) for x in a.devr_ts.split(",")], a.devr_variant))
        return
    if a.dagger:
        for bb in a.bbs.split(","):
            write(os.path.join(a.out, f"{bb}_{a.name}.jsonl"),
                  dagger_data(bb, a.name, a.dagger.split(","), [int(x) for x in a.dag_ts.split(",")]))
        return
    if a.plain:
        for bb in a.bbs.split(","):
            write(os.path.join(a.out, f"{bb}_{a.plain}.jsonl"), plain_dev(bb, a.plain))
        return
    if a.teach:
        for bb in a.bbs.split(","):
            write(os.path.join(a.out, f"{bb}_teach.jsonl"), teacher_data(bb))
        return
    if a.rec:
        for bb in a.bbs.split(","):
            write(os.path.join(a.out, f"{bb}_lrec.jsonl"), recordings(bb, range(20, 40)))
        return
    for bb in a.bbs.split(","):
        names = (["cut_w2", "cut_w4", "cut_w8", "cut_wg", "aw", "aw7", "aw_eng"] if bb == "pi05"
                 else ["cut_w2", "cut_w4", "cut_w8", "cut_wg", "cut_ws4"])
        if a.dev:
            names = a.dev.split(",")
        smoke = []
        for v in names + ["vanilla"]:
            smoke += [e for e in episodes(bb, "ls", v, [40], [32]) if True]
        write(os.path.join(a.out, f"{bb}_{'lsmoke' if a.name == 'ld' else a.name + 's'}.jsonl"), smoke)
        dev = []
        for v in names:
            dev += episodes(bb, a.name, v, range(40, 50), dev_ts, video_inits=(40,))
        if a.name == "ld":
            dev += episodes(bb, "ld", "vanilla", [40, 41], [2, 12, 22, 32, 42])   # reproduces the earlier d2 runs?
        write(os.path.join(a.out, f"{bb}_{a.name}.jsonl"), dev)
        if a.test:
            test = []
            for v in a.test.split(","):
                test += episodes(bb, "lt", v, range(20), [2, 12, 22, 32, 42, 52, 62], video_inits=(0,))
            write(os.path.join(a.out, f"{bb}_lt.jsonl"), test)
        if a.goal:
            g = []
            for v in a.goal.split(","):
                g += goal_episodes(bb, "lg", v, range(8), [20, 40])
            write(os.path.join(a.out, f"{bb}_lg.jsonl"), g)


if __name__ == "__main__":
    main()
