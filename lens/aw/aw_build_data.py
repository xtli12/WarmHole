"""Build the training set of the self-trained wormhole policy from the recorded teacher episodes.

Input: run directories of the teacher specs ({bb}_teach.jsonl, initial states 20-39 only) with results_*.jsonl and
steps/<id>.npz (bh_client_lens rec_steps).  Only successful episodes are used.  Output directory:
  img.u8, wrist.u8   uint8 memmaps (N, 224, 224, 3): the policy's agent and wrist views at every control step
  meta.npz           per frame: ep, t, prompt (index into prompts.json), act (N, HMAX, 7) executed actions from t on,
                     act_mask (N, HMAX) (a chunk never crosses the switch: before t_s it ends at t_s), uv_b / vis_b and
                     uv_w / vis_w (pixel position of the object the current instruction refers to in the 224 px policy
                     images, visible flags), rel (object - end effector, m), cat (0 before the switch, 1 scripted teacher
                     transit, 2 policy after the hand-back, 3 no-switch rollout, 4 student action of a DAgger episode --
                     src -3 / -4 -- which is never trained on), init
  episodes.json, prompts.json
usage: aw_build_data.py OUT RUN_DIR [RUN_DIR ...] [--hmax 50] [--procs 16]
c5 (11-01, handover 7.11): --self builds the training set of SWITCH SELF-PLAY from the run directories of a trained
policy's own rollouts (make_specs_lens_r21 --selfplay; bh_client_lens14 records every control step and writes the
condition grp = direction / t_s / layout into the results).  Every condition was rolled out K times with different
policy noise; p_c = share of successes in condition c.  A successful episode gets the group-relative weight
w = (1 - p_c) + floor (the positive part of the group-relative advantage: the less sure the policy is in a condition,
the more its own successes there count; the floor keeps successes on newly mastered layouts); failures are not used.
Only frames after the switch are kept (t = t_s, t_s + stride, ...), with the policy's own executed actions as labels,
category 5 and the per-frame weight fw = w (meta.npz key "fw").  The grounding targets are computed exactly as for the
teacher data.  Without --self the behaviour is that of c4 (teacher / DAgger data, no "fw").
c6 (11-01, handover 7.13, round g): --cat_as C (conditional DAgger take-overs and their counterfactual branches,
bh_client_lens15): the teacher frames (scripted transit src -2 and the teacher policy after the hand-back) of every
successful episode become category C with the per-frame weight fw = 3 (transit) / 1 (policy) -- the weights of the
teacher categories 1 / 2 -- and the student's own frames (category 4, never trained on) are not stored; with --need_takeover only
episodes with a recorded take-over (dg_reason k / freeze) are used.  Without --cat_as the behaviour is that of c5.
c7 (11-01, handover 7.15, round i): --policy_max N (with --cat_as): of the teacher frames keep every scripted-transit
frame and only the first N policy frames after the hand-back (the decision part of a relabelled switch; bounds the
size).  Without --policy_max the behaviour is that of c6."""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os

import numpy as np

RES = 224


def project(points, cam_pos, cam_rot, fovy_deg):
    """World points (n, 3) -> pixels (u right, v down) of the 224 px policy image (upright, mirrored) and depth."""
    pc = (np.asarray(points, np.float64) - np.asarray(cam_pos, np.float64)) @ np.asarray(cam_rot, np.float64)
    z = -pc[:, 2]
    f = 0.5 * RES / math.tan(math.radians(float(fovy_deg)) / 2)
    zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = 0.5 * RES + f * pc[:, 0] / zz
    v = 0.5 * RES - f * pc[:, 1] / zz
    return np.stack([RES - u, v], axis=1), z


def project_seq(points, cam_pos, cam_rot, fovy):
    """Per-step camera poses (T, 3), (T, 3, 3), fovy (T,)."""
    pc = np.einsum("ti,tij->tj", np.asarray(points, np.float64) - np.asarray(cam_pos, np.float64),
                   np.asarray(cam_rot, np.float64))
    z = -pc[:, 2]
    f = 0.5 * RES / np.tan(np.radians(np.asarray(fovy, np.float64)) / 2)
    zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = 0.5 * RES + f * pc[:, 0] / zz
    v = 0.5 * RES - f * pc[:, 1] / zz
    return np.stack([RES - u, v], axis=1), z


def episode_meta(path, hmax, cat_as=None):
    z = np.load(path)
    act = z["act"].astype(np.float32)
    T = len(act)
    ts = int(z["ts"])
    sw = z["sw"].astype(bool)
    src = z["src"].astype(np.int32)
    ref = np.where(sw[:, None], z["new_obj"], z["old_obj"]).astype(np.float64)
    uv_b, zb = project(ref, z["bcam_pos"], z["bcam_rot"], z["bfovy"])
    vis_b = (zb > 0.05) & (uv_b[:, 0] >= 0) & (uv_b[:, 0] < RES) & (uv_b[:, 1] >= 0) & (uv_b[:, 1] < RES)
    uv_w, zw = project_seq(ref, z["wcam_pos"], z["wcam_rot"], z["wfovy"])
    vis_w = (zw > 0.02) & (uv_w[:, 0] >= 0) & (uv_w[:, 0] < RES) & (uv_w[:, 1] >= 0) & (uv_w[:, 1] < RES)
    rel = (ref - z["ee"].astype(np.float64)).astype(np.float32)
    chunk = np.zeros((T, hmax, 7), np.float32)
    mask = np.zeros((T, hmax), bool)
    for t in range(T):
        end = ts if (ts >= 0 and t < ts) else T
        n = max(0, min(hmax, end - t))
        chunk[t, :n] = act[t:t + n]
        mask[t, :n] = True
    if ts < 0:
        cat = np.full(T, 3, np.int8)
    else:
        cat = np.where(~sw, 0, np.where(src == -2, 1, 2)).astype(np.int8)
    cat[(src == -3) | (src == -4)] = 4           # DAgger: the student's own actions carry no teacher label
    fw = np.ones(T, np.float64)
    if cat_as is not None:                       # c6: the teacher frames of a round-g episode -> category cat_as
        tf = (cat == 1) | (cat == 2)
        fw = np.where(src == -2, 3.0, 1.0)
        cat[tf] = int(cat_as)
    prompts = (str(z["old_lang"]), str(z["new_lang"]))
    return dict(T=T, ts=ts, sw=sw, act=chunk, act_mask=mask, state=z["state"].astype(np.float32),
                uv_b=uv_b.astype(np.float32), vis_b=vis_b,
                uv_w=uv_w.astype(np.float32), vis_w=vis_w, rel=rel, cat=cat, prompts=prompts, fw=fw)


def _write_images(args):
    path, start, out = args[:3]
    sel = args[3] if len(args) > 3 else None       # c5: only these frame indices of the episode
    z = np.load(path)
    img, wr = z["img"], z["wrist"]
    if sel is not None:
        img, wr = img[sel], wr[sel]
    T = len(img)
    a = np.memmap(os.path.join(out, "img.u8"), dtype=np.uint8, mode="r+")
    a = a.reshape(-1, RES, RES, 3)
    a[start:start + T] = img
    a.flush()
    b = np.memmap(os.path.join(out, "wrist.u8"), dtype=np.uint8, mode="r+").reshape(-1, RES, RES, 3)
    b[start:start + T] = wr
    b.flush()
    return T


def _grp(r):
    g = r.get("grp")
    if g:
        return g
    # fallback: the condition from the id (..._sp<d>_flush_ts<ts>_i<init>_n<k>)
    return "_".join(x for x in str(r["id"]).split("_") if x.startswith(("spA", "spB", "ts", "i")))


def build_self(a):
    """c5: training set of switch self-play (see the module docstring)."""
    rows = {}
    for d in a.runs:
        for p in glob.glob(os.path.join(d, "results_*.jsonl")):
            for ln in open(p):
                try:
                    r = json.loads(ln)
                except Exception:  # noqa: BLE001
                    continue
                r["_dir"] = d
                rows[r["id"]] = r
    good = [r for r in rows.values() if not r.get("error") and not r.get("invalid") and r.get("outcome") is not None
            and r.get("ts") is not None and r.get("switched", True)]
    print(f"{len(rows)} result rows, {len(good)} valid switch episodes, {len(rows) - len(good)} excluded", flush=True)
    groups = {}
    for r in good:
        groups.setdefault(_grp(r), []).append(r)
    pc = {g: float(np.mean([x["outcome"] == "success" for x in rs])) for g, rs in groups.items()}
    ks = sorted({len(rs) for rs in groups.values()})
    print(f"{len(groups)} conditions, rollouts per condition {ks}; success rate per condition: "
          + " ".join(f"{v:.2f}:{sum(abs(x - v) < 1e-9 for x in pc.values())}" for v in sorted(set(pc.values()))), flush=True)
    eps = []
    for r in good:
        if r["outcome"] != "success":
            continue
        f = os.path.join(r["_dir"], "steps", r["id"] + ".npz")
        if not os.path.exists(f):
            continue
        g = _grp(r)
        w = (1.0 - pc[g]) + a.floor
        if w <= 0:
            continue
        eps.append(dict(id=r["id"], path=f, init=int(r["init"]), pair=r.get("pair"), ts=int(r["ts"]), grp=g,
                        p_c=pc[g], w=float(w), variant=r.get("variant"), seed=r.get("seed")))
    if a.max_eps:
        rs = np.random.default_rng(0)
        eps = [eps[i] for i in sorted(rs.choice(len(eps), size=min(a.max_eps, len(eps)), replace=False))]
    print(f"{len(eps)} successful episodes with step records and weight > 0", flush=True)
    with mp.Pool(a.procs) as pool:
        metas = pool.starmap(episode_meta, [(e["path"], a.hmax, getattr(a, "cat_as", None)) for e in eps])
    sels = []
    for e, m in zip(eps, metas):
        ts = int(m["ts"])
        assert ts == e["ts"] and ts >= 0, (e["id"], ts)
        sel = np.arange(ts, m["T"], a.stride, dtype=np.int64)
        sel = sel[m["act_mask"][sel].sum(1) > 0]
        assert bool(m["sw"][sel].all()), e["id"]
        sels.append(sel)
    prompts = sorted({p for m in metas for p in m["prompts"] if p})
    pidx = {p: i for i, p in enumerate(prompts)}
    N = int(sum(len(s) for s in sels))
    print(f"{N} frames (every {a.stride}-th after the switch); prompts: {prompts}", flush=True)
    starts = np.cumsum([0] + [len(s) for s in sels])[:-1]
    for name in ("img.u8", "wrist.u8"):
        mm = np.memmap(os.path.join(a.out, name), dtype=np.uint8, mode="w+", shape=(max(N, 1), RES, RES, 3))
        del mm
    with mp.Pool(a.procs) as pool:
        n_w = pool.map(_write_images, [(e["path"], int(s), a.out, sel) for e, s, sel in zip(eps, starts, sels)])
    assert sum(n_w) == N

    def cat(k):
        return np.concatenate([m[k][sel] for m, sel in zip(metas, sels)])

    ep_idx = np.concatenate([np.full(len(sel), i, np.int32) for i, sel in enumerate(sels)])
    t_idx = np.concatenate([sel.astype(np.int32) for sel in sels])
    prompt = np.concatenate([np.full(len(sel), pidx[m["prompts"][1]], np.int16) for m, sel in zip(metas, sels)])
    init = np.concatenate([np.full(len(sel), e["init"], np.int32) for e, sel in zip(eps, sels)])
    fw = np.concatenate([np.full(len(sel), e["w"], np.float32) for e, sel in zip(eps, sels)])
    np.savez(os.path.join(a.out, "meta.npz"), ep=ep_idx, t=t_idx, prompt=prompt, act=cat("act"),
             act_mask=cat("act_mask"), uv_b=cat("uv_b"), vis_b=cat("vis_b"), uv_w=cat("uv_w"), vis_w=cat("vis_w"),
             rel=cat("rel"), cat=np.full(N, 5, np.int8), init=init, state=cat("state"), fw=fw, N=N)
    for e, s, sel in zip(eps, starts, sels):
        e.update(start=int(s), n=int(len(sel)))
    json.dump(eps, open(os.path.join(a.out, "episodes.json"), "w"), indent=0)
    json.dump(prompts, open(os.path.join(a.out, "prompts.json"), "w"), indent=1)
    json.dump({g: {"p": pc[g], "n": len(groups[g])} for g in sorted(groups)}, open(os.path.join(a.out, "conditions.json"), "w"),
              indent=0)
    by = {}
    for e, sel in zip(eps, sels):
        k = (e["pair"], "late" if e["ts"] >= 52 else "comm", "official" if e["init"] < 10000 else "procedural")
        by.setdefault(k, [0, 0, 0.0])
        by[k][0] += 1
        by[k][1] += len(sel)
        by[k][2] += e["w"] * len(sel)
    for k in sorted(by):
        print(f"  {k}: episodes {by[k][0]}, frames {by[k][1]}, weight mass {by[k][2]:.1f}")
    print("base-view target visible:", float(cat("vis_b").mean()) if N else 0.0, " wrist-view target visible:",
          float(cat("vis_w").mean()) if N else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--hmax", type=int, default=50)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--max_eps", type=int, default=0, help="use only the first N episodes (smoke tests)")
    ap.add_argument("--self", action="store_true", help="c5: switch self-play episodes (group-relative weights, category 5)")
    ap.add_argument("--stride", type=int, default=2, help="c5 --self: keep every stride-th frame after the switch")
    ap.add_argument("--floor", type=float, default=0.1, help="c5 --self: w = (1 - p_c) + floor for successful episodes")
    ap.add_argument("--cat_as", type=int, default=None, help="c6: teacher frames of round-g episodes -> this category")
    ap.add_argument("--need_takeover", action="store_true", help="c6: only episodes with a take-over (dg_reason k / freeze)")
    ap.add_argument("--policy_max", type=int, default=None,
                    help="c7, with --cat_as: keep every transit frame and only the first N policy frames after the hand-back")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.self:
        return build_self(a)
    eps = []
    for d in a.runs:
        res = {}
        for p in glob.glob(os.path.join(d, "results_*.jsonl")):
            for ln in open(p):
                try:
                    r = json.loads(ln)
                except Exception:  # noqa: BLE001
                    continue
                res[r["id"]] = r
        for eid, r in sorted(res.items()):
            f = os.path.join(d, "steps", eid + ".npz")
            if a.need_takeover and r.get("dg_reason") not in ("k", "freeze"):
                continue
            if r.get("outcome") == "success" and not r.get("error") and os.path.exists(f):
                eps.append(dict(id=eid, path=f, init=int(r["init"]), pair=r.get("pair"), ts=r.get("ts"),
                                variant=r.get("variant"), seed=r.get("seed")))
    if a.max_eps:
        rs = np.random.default_rng(0)
        eps = [eps[i] for i in sorted(rs.choice(len(eps), size=min(a.max_eps, len(eps)), replace=False))]
    print(f"{len(eps)} successful episodes with step records", flush=True)
    with mp.Pool(a.procs) as pool:
        metas = pool.starmap(episode_meta, [(e["path"], a.hmax, getattr(a, "cat_as", None)) for e in eps])
    prompts = sorted({p for m in metas for p in m["prompts"] if p})
    pidx = {p: i for i, p in enumerate(prompts)}
    if a.cat_as is not None:
        # c6: only the teacher frames are stored (the student's own frames, category 4, are never trained on); the
        # action chunks were cut from the whole episode before the selection
        sels = [np.where(m["cat"] != 4)[0] for m in metas]
        if a.policy_max is not None:
            # c7 (handover 7.15, round i): every scripted-transit frame (fw 3) and only the first policy_max policy
            # frames after the hand-back (fw 1)
            sels = [x[(m["fw"][x] == 3.0) | (np.cumsum(m["fw"][x] == 1.0) <= a.policy_max)] for m, x in zip(metas, sels)]
    else:
        sels = [np.arange(m["T"]) for m in metas]
    N = int(sum(len(x) for x in sels))
    print(f"{N} frames; prompts: {prompts}", flush=True)
    starts = np.cumsum([0] + [len(x) for x in sels])[:-1]
    for name in ("img.u8", "wrist.u8"):
        mm = np.memmap(os.path.join(a.out, name), dtype=np.uint8, mode="w+", shape=(max(N, 1), RES, RES, 3))
        del mm
    with mp.Pool(a.procs) as pool:
        if a.cat_as is not None:
            n_w = pool.map(_write_images, [(e["path"], int(s), a.out, x) for e, s, x in zip(eps, starts, sels)])
        else:
            n_w = pool.map(_write_images, [(e["path"], int(s), a.out) for e, s in zip(eps, starts)])
    assert sum(n_w) == N
    cat = lambda k: np.concatenate([m[k][x] for m, x in zip(metas, sels)])  # noqa: E731
    ep_idx = np.concatenate([np.full(len(x), i, np.int32) for i, x in enumerate(sels)])
    t_idx = np.concatenate([x.astype(np.int32) for x in sels])
    prompt = np.concatenate([np.where(m["sw"][x], pidx.get(m["prompts"][1], -1), pidx[m["prompts"][0]]).astype(np.int16)
                             for m, x in zip(metas, sels)])
    init = np.concatenate([np.full(len(x), e["init"], np.int16) for e, x in zip(eps, sels)])
    extra = {"fw": cat("fw")} if a.cat_as is not None else {}     # c6: per-frame weights of the round-g categories
    np.savez(os.path.join(a.out, "meta.npz"), ep=ep_idx, t=t_idx, prompt=prompt, act=cat("act"),
             act_mask=cat("act_mask"), uv_b=cat("uv_b"), vis_b=cat("vis_b"), uv_w=cat("uv_w"), vis_w=cat("vis_w"),
             rel=cat("rel"), cat=cat("cat"), init=init, state=cat("state"), N=N, **extra)
    for e, x, s in zip(eps, sels, starts):
        e.update(start=int(s), T=int(len(x)))
    json.dump(eps, open(os.path.join(a.out, "episodes.json"), "w"), indent=0)
    json.dump(prompts, open(os.path.join(a.out, "prompts.json"), "w"), indent=1)
    c = cat("cat")
    print("frames per category (0 before switch, 1 teacher transit, 2 policy after hand-back, 3 no switch, 4 student):",
          {int(k): int((c == k).sum()) for k in np.unique(c)})
    print("base-view target visible:", float(cat("vis_b").mean()), " wrist-view target visible:",
          float(cat("vis_w").mean()))
    print("episodes per variant:", {v: sum(e["variant"] == v for e in eps) for v in sorted({e["variant"] for e in eps})})
    if a.cat_as is not None:
        sel = c == a.cat_as
        f = cat("fw")
        print(f"c6: category {a.cat_as}: {int(sel.sum())} frames, weight mass {float(f[sel].sum()):.0f}"
              f" (transit frames {int((sel & (f == 3.0)).sum())}, policy frames {int((sel & (f == 1.0)).sum())})")


if __name__ == "__main__":
    main()
