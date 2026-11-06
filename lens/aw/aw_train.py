"""Train the self-trained attention-wormhole pi0.5 (or the plain LoRA behaviour-cloning control) on teacher data.

The base pi0.5 LIBERO checkpoint is loaded; LoRA adapters (both experts), the wormhole gates, the [LOC] token and its
read-out head and the small action/time projections are trained; SigLIP and all pretrained transformer weights are
frozen.  Loss = flow-matching action loss (masked: a chunk never crosses the switch)
              + lam_loc * cross-entropy of the [LOC] grounding maps against a Gaussian around the projected object of
                the current instruction (base view; wrist view when the object is in it)
              + lam_coord * smooth-L1 of the target read-out (pixel position in the base view, object - gripper in dm).
Data: aw_build_data.py output; episodes of the validation initial states (--val_inits) are held out (monitoring only).
Augmentation: colour jitter; with --geo_aug also the geometric augmentation of openpi's pi0.5 fine-tuning for the
base camera (random crop of --geo_crop of the side, resized back, rotation within +-geo_rot degrees), applied to the
image AND to the projected object position, so the grounding targets (Gaussian map, visibility, pixel read-out) stay
consistent; the wrist camera is not transformed (as in openpi).
Frames of category >= 4 (actions produced by the student in DAgger episodes, no teacher label) are never trained on.
--data may list several aw_build_data.py outputs (comma-separated, e.g. the teacher data and the DAgger data); they are
concatenated without copying (prompts are merged), frames are sampled by category weight over all of them.
Numerical guard: with --skip_gn G an update whose gradient norm is not below G is skipped like a non-finite one (and
counted in `skipped`); every step whose loss or gradient norm is abnormal is written to <out>/spikes.jsonl with the
frames of its batch and their per-sample flow loss (to tell data problems from numerical ones).
--host_place: the replicated training state and every random key are placed on each card from host memory, one
transfer per card, so that no card-to-card copy is made (on the server, card-to-card copies are corrupted while NCCL
reductions are correct; without this option the state is created on the first card and copied to the others).
--val_only: only the validation metrics of the initial model (identical for the same kind and seed: a fingerprint that
the cards compute correctly), printed as "VAL0ONLY ..."; nothing is trained.
Saves the trainable parameters only (<out>/params_<step>.npz); the server re-loads them on top of the base checkpoint.
Action convention (c10, 11-01): pi0 is trained, like openpi's pi0_libero (extra_delta_transform=True), on actions whose
first 6 dimensions are relative to the current state (actions[:6] - state[:6], then z-score with the checkpoint's norm
stats); aw_server.py c8 adds the state back (AbsoluteActions).  pi0.5 (extra_delta_transform=False) is unchanged.
c9 trained pi0 on absolute actions (step-0 validation flow 4.89 instead of a few hundredths).
c11 (11-01, handover 7.11, switch self-play): frames of category 5 (the policy's own successful post-switch frames,
aw_build_data.py c5 --self) are trained on like teacher frames; a data root may carry per-frame weights "fw" (default
1); --share 5:0.3 gives category 5 a fixed share of the sampling mass (distributed in proportion to fw), the other
categories share the rest in proportion to cat_w x fw as before.  Category 4 (DAgger student actions) is never trained
on.  Without category-5 data and --share the sampling is exactly that of c10.
usage: aw_train.py --data DIR --kind aw|awx|awa|loc|bc --out DIR [--steps 6000 --bs 32 --lr 5e-5 ...]"""
import argparse
import dataclasses
import json
import logging
import os
import pathlib
import queue
import threading
import time

import augmax
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import sharding

import aw_model as AW

RES, GRID, PATCH = 224, 16, 14


def patch_centers():
    c = (np.arange(GRID) + 0.5) * PATCH
    cx, cy = np.meshgrid(c, c)
    return np.stack([cx.ravel(), cy.ravel()], axis=1)


def gauss_targets(uv, sigma):
    d2 = ((patch_centers()[None] - uv[:, None]) ** 2).sum(-1)
    y = np.exp(-d2 / (2 * sigma ** 2))
    return (y / (y.sum(-1, keepdims=True) + 1e-12)).astype(np.float32)


def geo_params(rng, crop, rot):
    """Random crop (side fraction s in [crop, 1], position uniform), resized back to RES, rotation in [-rot, rot] deg."""
    s = rng.uniform(crop, 1.0)
    return s, np.deg2rad(rng.uniform(-rot, rot)), rng.uniform(0, (1 - s) * RES), rng.uniform(0, (1 - s) * RES)


_GRID = np.meshgrid(np.arange(RES) + 0.5, np.arange(RES) + 0.5)       # output pixel centres (x, y)


def geo_warp(img, s, ang, ox, oy):
    """Output pixel p' -> source point o + s * (R(-ang) (p' - c) + c); bilinear, edge-clamped. img (RES, RES, 3) uint8."""
    c = RES / 2.0
    ca, sa = np.cos(-ang), np.sin(-ang)
    jj, ii = _GRID
    px = ca * (jj - c) - sa * (ii - c) + c
    py = sa * (jj - c) + ca * (ii - c) + c
    x = ox + s * px - 0.5
    y = oy + s * py - 0.5
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = (x - x0)[..., None].astype(np.float32)
    fy = (y - y0)[..., None].astype(np.float32)
    x0c, x1c = np.clip(x0, 0, RES - 1), np.clip(x0 + 1, 0, RES - 1)
    y0c, y1c = np.clip(y0, 0, RES - 1), np.clip(y0 + 1, 0, RES - 1)
    im = img.astype(np.float32)
    out = (im[y0c, x0c] * (1 - fx) * (1 - fy) + im[y0c, x1c] * fx * (1 - fy)
           + im[y1c, x0c] * (1 - fx) * fy + im[y1c, x1c] * fx * fy)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def geo_points(uv, s, ang, ox, oy):
    """The same transform for points: source uv -> output pixel coordinates.  uv (n, 2)."""
    c = RES / 2.0
    q = (np.asarray(uv, np.float64) - np.array([ox, oy])) / s
    ca, sa = np.cos(ang), np.sin(ang)
    x = ca * (q[:, 0] - c) - sa * (q[:, 1] - c) + c
    y = sa * (q[:, 0] - c) + ca * (q[:, 1] - c) + c
    return np.stack([x, y], axis=1)


class Data:
    """Frames in RAM-mapped files; weighted sampling of training frames (by category)."""

    def __init__(self, root, horizon, norm_stats, max_token_len, val_inits, cat_w, sigma, exclude_cats=(),
                 quantile=True, geo=None, delta=False, share=None):
        roots = [pathlib.Path(r) for r in str(root).split(",") if r]
        metas = [np.load(r / "meta.npz") for r in roots]
        sizes = [int(mm["N"]) for mm in metas]
        self.N = int(sum(sizes))
        self.imgs = [np.memmap(r / "img.u8", dtype=np.uint8, mode="r", shape=(n, RES, RES, 3)) for r, n in zip(roots, sizes)]
        self.wrists = [np.memmap(r / "wrist.u8", dtype=np.uint8, mode="r", shape=(n, RES, RES, 3))
                       for r, n in zip(roots, sizes)]
        self.root_of = np.concatenate([np.full(n, k, np.int64) for k, n in enumerate(sizes)])
        self.local = np.concatenate([np.arange(n, dtype=np.int64) for n in sizes])
        plists = [json.load(open(r / "prompts.json")) for r in roots]
        self.prompts = sorted({p for pl in plists for p in pl})
        gidx = {p: i for i, p in enumerate(self.prompts)}
        remap = [np.asarray([gidx[p] for p in pl] + [-1], np.int64) for pl in plists]   # index -1 stays -1

        def cat_all(key, fn=None):
            parts = []
            for k, mm in enumerate(metas):
                x = mm[key]
                parts.append(fn(k, x) if fn else x)
            return np.concatenate(parts)

        class _M(dict):
            files = ()
        m = _M()
        for key in ("act", "act_mask", "cat", "init", "uv_b", "vis_b", "uv_w", "vis_w", "rel", "ep", "t"):
            m[key] = cat_all(key)
        m["prompt"] = cat_all("prompt", lambda k, x: remap[k][np.asarray(x, np.int64)])
        if all("state" in mm.files for mm in metas):
            m["state"] = cat_all("state")
            m.files = ("state",)
        # c11: per-frame sampling weights (self-play data); 1 for data roots without them
        m["fw"] = np.concatenate([np.asarray(mm["fw"], np.float64) if "fw" in mm.files else np.ones(n, np.float64)
                                  for mm, n in zip(metas, sizes)])
        m["N"] = self.N
        logging.info("data roots %s: frames %s", [str(r) for r in roots], sizes)
        tok = _tokenizer.PaligemmaTokenizer(max_token_len)
        tt = [tok.tokenize(p) for p in self.prompts]
        self.tok = np.stack([t[0] for t in tt]).astype(np.int32)
        self.tmask = np.stack([t[1] for t in tt]).astype(bool)
        def norm(x, st):        # the policy's own normalisation: quantiles for pi0.5, z-score for pi0
            d = x.shape[-1]
            if quantile:
                q01, q99 = np.asarray(st.q01)[:d], np.asarray(st.q99)[:d]
                return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            return (x - np.asarray(st.mean)[:d]) / (np.asarray(st.std)[:d] + 1e-6)

        act = m["act"][:, :horizon].astype(np.float32)
        if delta:   # openpi DeltaActions(make_bool_mask(6, -1)) of pi0_libero: the first 6 dims relative to the current state
            if "state" not in m.files:
                raise ValueError("delta actions (pi0) need the per-frame state in meta.npz")
            act[..., :6] -= m["state"][:, None, :6].astype(np.float32)
        logging.info("action convention: %s", "delta (first 6 dims minus the state, as openpi pi0_libero)" if delta else "absolute")
        a = norm(act, norm_stats["actions"])
        self.act = np.zeros((self.N, horizon, 32), np.float32)
        self.act[..., :7] = a
        self.state = np.zeros((self.N, 32), np.float32)
        if "state" in m.files:
            self.state[:, :8] = norm(m["state"].astype(np.float32), norm_stats["state"])
        self.amask = m["act_mask"][:, :horizon].astype(bool)
        self.prompt = m["prompt"].astype(np.int64)
        self.cat = m["cat"].astype(np.int64)
        self.init = m["init"].astype(np.int64)
        self.ep, self.t = m["ep"].astype(np.int64), m["t"].astype(np.int64)
        self.sigma, self.geo = sigma, geo
        self.uv_b = m["uv_b"].astype(np.float32)
        self.yb = gauss_targets(m["uv_b"], sigma)
        self.vb = m["vis_b"].astype(np.float32)
        self.yw = gauss_targets(m["uv_w"], sigma)
        self.vw = m["vis_w"].astype(np.float32)
        uvb = np.clip(m["uv_b"], -RES, 2 * RES) / RES
        self.coord = np.concatenate([uvb, 10.0 * m["rel"]], axis=1).astype(np.float32)
        val = np.isin(self.init, np.asarray(val_inits))
        ok = ((self.amask.sum(1) > 0) & (self.prompt >= 0) & ~np.isin(self.cat, np.asarray(exclude_cats, np.int64))
              & (self.cat != 4))                    # category 4: student actions of DAgger episodes (no teacher label)
        self.train_idx = np.where(ok & ~val)[0]
        self.val_idx = np.where(ok & val)[0]
        fw = m["fw"][self.train_idx]
        ct = self.cat[self.train_idx]
        w = np.asarray([cat_w[int(c)] if int(c) < len(cat_w) else 0.0 for c in ct], np.float64) * fw
        share = dict(share or {})
        if share:
            # c11: categories with a fixed share of the sampling mass (in proportion to fw inside the category)
            rest = ~np.isin(ct, np.asarray(list(share), np.int64))
            p = np.zeros(len(ct), np.float64)
            p[rest] = w[rest] / w[rest].sum() * (1.0 - sum(share.values()))
            for c, s in share.items():
                sel = ct == c
                assert sel.any() and fw[sel].sum() > 0, f"--share {c}: no training frames of category {c}"
                p[sel] = fw[sel] / fw[sel].sum() * s
            self.p = p / p.sum()
        else:
            self.p = w / w.sum()
        logging.info("frames %d: train %d, val %d; category shares in training batches %s", self.N, len(self.train_idx),
                     len(self.val_idx), {int(c): round(float(self.p[self.cat[self.train_idx] == c].sum()), 3)
                                         for c in np.unique(self.cat[self.train_idx])})

    def images(self, idx):
        """Frames idx (sorted) from the memmaps of their data roots."""
        img = np.empty((len(idx), RES, RES, 3), np.uint8)
        wr = np.empty((len(idx), RES, RES, 3), np.uint8)
        ro = self.root_of[idx]
        for k in np.unique(ro):
            sel = np.where(ro == k)[0]
            li = self.local[idx[sel]]
            img[sel] = self.imgs[k][li]
            wr[sel] = self.wrists[k][li]
        return img, wr

    def batch(self, idx):
        idx = np.sort(idx)
        img, wr = self.images(idx)
        return {"idx": idx, "img": img, "wrist": wr, "tok": self.tok[self.prompt[idx]],
                "tmask": self.tmask[self.prompt[idx]], "act": self.act[idx], "amask": self.amask[idx],
                "yb": self.yb[idx], "vb": self.vb[idx], "yw": self.yw[idx], "vw": self.vw[idx], "coord": self.coord[idx],
                "cat": self.cat[idx], "state": self.state[idx]}

    def sample(self, rng, bs):
        b = self.batch(rng.choice(self.train_idx, size=bs, p=self.p))
        if self.geo:
            self.geo_augment(b, rng)
        return b

    def geo_augment(self, b, rng):
        """Base camera only: warp the image and move the projected target with it (Gaussian map, visibility, read-out)."""
        crop, rot = self.geo
        img = b["img"].copy()
        uv = b["coord"][:, :2] * RES          # (clipped) projected position of the referred object, base view
        yb, vb, coord = b["yb"].copy(), b["vb"].copy(), b["coord"].copy()
        for k in range(len(img)):
            s, ang, ox, oy = geo_params(rng, crop, rot)
            img[k] = geo_warp(img[k], s, ang, ox, oy)
            u2 = geo_points(uv[k:k + 1], s, ang, ox, oy)[0]
            inside = (0 <= u2[0] < RES) and (0 <= u2[1] < RES)
            yb[k] = gauss_targets(u2[None], self.sigma)[0]
            vb[k] = vb[k] * float(inside)
            coord[k, :2] = np.clip(u2, -RES, 2 * RES) / RES
        b.update(img=img, yb=yb, vb=vb, coord=coord)


def prefetch(data, bs, seed, n=4, threads=3):
    q = queue.Queue(maxsize=n)

    def run(k):
        rng = np.random.default_rng([seed, k])          # independent stream per thread
        while True:
            q.put(data.sample(rng, bs))

    for k in range(threads):
        threading.Thread(target=run, args=(k,), daemon=True).start()
    while True:
        yield q.get()


def make_observation(b, rng=None, train=False):
    """uint8 frames -> openpi Observation (images in [-1, 1]); colour jitter when training."""
    imgs = {}
    for k, key in (("img", "base_0_rgb"), ("wrist", "left_wrist_0_rgb")):
        x = b[k].astype(jnp.float32) / 255.0
        if train:
            rngs = jax.random.split(jax.random.fold_in(rng, 1 if key == "base_0_rgb" else 2), x.shape[0])
            x = jax.vmap(augmax.Chain(augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)))(rngs, x)
        imgs[key] = x * 2.0 - 1.0
    bsz = b["tok"].shape[0]
    return _model.Observation(images=imgs, image_masks={k: jnp.ones((bsz,), jnp.bool_) for k in imgs},
                              state=b["state"], tokenized_prompt=b["tok"],
                              tokenized_prompt_mask=b["tmask"])


def losses(model, rng, b, train, lam_loc, lam_coord):
    r_aug, r_flow = jax.random.split(rng)
    obs = make_observation(b, r_aug, train)
    out = model.aw_losses(r_flow, obs, b["act"], train=train)
    am = b["amask"].astype(jnp.float32)
    flow = (out["flow"] * am).sum() / jnp.maximum(am.sum(), 1.0)
    info = {"flow": flow,
            "flow_ps": (out["flow"] * am).sum(-1) / jnp.maximum(am.sum(-1), 1.0)}     # per sample (spike reports only)
    total = flow
    if "pb" in out:
        ce_b = -(b["yb"][None] * jnp.log(out["pb"] + 1e-9)).sum(-1).mean(0)            # (B,)
        ce_w = -(b["yw"][None] * jnp.log(out["pw"] + 1e-9)).sum(-1).mean(0)
        ent_b = -(b["yb"] * jnp.log(b["yb"] + 1e-12)).sum(-1)
        ent_w = -(b["yw"] * jnp.log(b["yw"] + 1e-12)).sum(-1)
        kl_b = ((ce_b - ent_b) * b["vb"]).sum() / jnp.maximum(b["vb"].sum(), 1.0)
        kl_w = ((ce_w - ent_w) * b["vw"]).sum() / jnp.maximum(b["vw"].sum(), 1.0)
        total = total + lam_loc * (kl_b + 0.5 * kl_w)
        # pointing accuracy of the mean map: argmax patch within 1.5 patches of the target
        c = jnp.asarray(patch_centers())
        tgt = (b["yb"][..., None] * c[None]).sum(1)
        pk = c[jnp.argmax(out["mb"], axis=-1)]
        hit = (jnp.linalg.norm(pk - tgt, axis=-1) < 1.5 * PATCH).astype(jnp.float32)
        info.update(kl_b=kl_b, kl_w=kl_w, point_b=(hit * b["vb"]).sum() / jnp.maximum(b["vb"].sum(), 1.0),
                    img_mass=out["mass"].mean())
    if "coord" in out:
        d = out["coord"] - b["coord"]
        sl1 = jnp.where(jnp.abs(d) < 0.1, 0.5 * d ** 2 / 0.1, jnp.abs(d) - 0.05).mean(-1)
        coord = (sl1 * b["vb"]).sum() / jnp.maximum(b["vb"].sum(), 1.0)
        total = total + lam_coord * coord
        info["coord"] = coord
        info["px_err"] = (jnp.linalg.norm(d[:, :2], axis=-1) * RES * b["vb"]).sum() / jnp.maximum(b["vb"].sum(), 1.0)
        info["rel_err_cm"] = (jnp.linalg.norm(d[:, 2:], axis=-1) * 10.0 * b["vb"]).sum() / jnp.maximum(b["vb"].sum(), 1.0)
    info["total"] = total
    return total, info


@dataclasses.dataclass
class TS:
    step: int
    params: nnx.State
    opt_state: optax.OptState


jax.tree_util.register_dataclass(TS, data_fields=["step", "params", "opt_state"], meta_fields=[])


def load_base(config, ckpt, rng):
    """Model with the base pi0.5 weights; new parameters (LoRA, wormhole, [LOC]) keep their initial values."""
    model = config.create(rng)
    graphdef, state = nnx.split(model)
    flat_model = traverse_util.flatten_dict(state.to_pure_dict())
    loaded = _model.restore_params(pathlib.Path(ckpt) / "params", restore_type=np.ndarray)
    flat_loaded = traverse_util.flatten_dict(loaded)
    partial, missing = {}, []
    for k, v in flat_model.items():
        if k in flat_loaded:
            assert tuple(flat_loaded[k].shape) == tuple(v.shape), (k, flat_loaded[k].shape, v.shape)
            partial[k] = flat_loaded[k]
        else:
            missing.append("/".join(map(str, k)))
    extra = ["/".join(map(str, k)) for k in flat_loaded if k not in flat_model]
    logging.info("loaded %d tensors; new (initialised) %d: %s ...; unused in checkpoint %d %s", len(partial),
                 len(missing), sorted(set(m.split("/")[-1] for m in missing)), len(extra), extra[:5])
    state.replace_by_pure_dict(traverse_util.unflatten_dict(partial))
    model = nnx.merge(graphdef, state)
    if getattr(model, "use_loc", False):
        emb = np.asarray(flat_loaded[("PaliGemma", "llm", "embedder", "input_embedding")], np.float32)
        lo, hi = AW.LOC_TOKEN_IDS
        model.loc_emb.value = jnp.asarray(emb[lo:hi].mean(0) * np.sqrt(emb.shape[1]), jnp.float32)
    return model


def save_trainable(path, params):
    flat = traverse_util.flatten_dict(params.to_pure_dict())
    np.savez(path, **{"/".join(map(str, k)): np.asarray(jax.device_get(v), np.float32) for k, v in flat.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--kind", default="aw", choices=["aw", "awx", "awa", "loc", "bc"])
    ap.add_argument("--bb", default="pi05", choices=["pi05", "pi0"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="", help="base checkpoint (default: the released LIBERO checkpoint of --bb)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_new", type=float, default=5e-4, help="learning rate of the new parameters ([LOC], gates)")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lam_loc", type=float, default=0.1)
    ap.add_argument("--lam_coord", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=14.0, help="target width (px) of the grounding maps")
    ap.add_argument("--cat_w", default="0.5,3,1,0.5", help="sampling weight per frame category 0,1,2,3")
    ap.add_argument("--val_inits", default="38,39")
    ap.add_argument("--exclude_cats", default="", help="frame categories left out (ablation)")
    ap.add_argument("--share", default="", help="c11: fixed sampling share per category, e.g. 5:0.3 (self-play frames)")
    ap.add_argument("--loc_layers", default="6,7,8,9,10,11,12,13")
    ap.add_argument("--loc_heads", default="0,1,2,3")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--geo_aug", action="store_true", help="openpi-style crop / rotation of the base camera (targets moved too)")
    ap.add_argument("--geo_crop", type=float, default=0.95)
    ap.add_argument("--geo_rot", type=float, default=5.0)
    ap.add_argument("--skip_gn", type=float, default=0.0,
                    help="skip (and count) every update whose gradient norm is not below this (0 = only non-finite)")
    ap.add_argument("--spike_flow", type=float, default=1.0, help="training flow loss above this is reported as a spike")
    ap.add_argument("--host_place", action="store_true",
                    help="place the replicated state and the random keys on every card from host memory (no card-to-card copies)")
    ap.add_argument("--val_only", action="store_true", help="print the validation metrics of the initial model and exit")
    ap.add_argument("--smoke", action="store_true", help="tiny dummy-width check of the code paths (CPU)")
    a = ap.parse_args()
    if not a.ckpt:
        a.ckpt = f"{os.environ.get('BH', '/data/blackhole')}/openpi_data/openpi-assets/checkpoints/{a.bb}_libero"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)
    cfg_kw = dict(loc_layers=tuple(int(x) for x in a.loc_layers.split(",")),
                  loc_heads=tuple(int(x) for x in a.loc_heads.split(",")))
    if a.smoke:
        cfg_kw.update(paligemma_variant="dummy", action_expert_variant="dummy", loc_layers=(1, 2), loc_heads=(0, 1))
    config = AW.aw_config(a.kind, a.bb, **cfg_kw)
    json.dump({k: str(v) for k, v in dataclasses.asdict(config).items()}, open(out / "config.json", "w"), indent=1)
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(a.ckpt) / "assets", "physical-intelligence/libero")
    cat_w = [float(x) for x in a.cat_w.split(",")]
    excl = tuple(int(x) for x in a.exclude_cats.split(",") if x != "")
    share = {int(k): float(v) for k, v in (x.split(":") for x in a.share.split(",") if x)}
    data = Data(a.data, config.action_horizon, norm_stats, config.max_token_len,
                tuple(int(x) for x in a.val_inits.split(",")), cat_w, a.sigma, excl, quantile=(a.bb == "pi05"),
                geo=(a.geo_crop, a.geo_rot) if a.geo_aug else None, delta=(a.bb == "pi0"), share=share)

    ndev = jax.device_count()
    assert a.bs % ndev == 0, (a.bs, ndev)
    mesh = sharding.make_mesh(1)
    dsh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    rng = jax.random.key(a.seed)
    if a.smoke:
        model = config.create(rng)
        if model.use_loc:
            model.loc_emb.value = jnp.zeros_like(model.loc_emb.value) + 0.1
    else:
        model = load_base(config, a.ckpt, rng)
    trainable = config.trainable_filter()
    frozen = nnx.All(nnx.Param, nnx.Not(trainable))
    graphdef, params = nnx.split(model)
    params = nnx_utils.state_map(params, frozen, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    tparams = params.filter(trainable)
    n_t = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(tparams))
    n_all = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(params))
    logging.info("trainable parameters %.2fM of %.1fM; devices %d", n_t / 1e6, n_all / 1e6, ndev)
    tflat = traverse_util.flatten_dict(tparams.to_pure_dict())
    labels = traverse_util.unflatten_dict(
        {k: ("new" if any(("worm_" in str(p)) or str(p).startswith("loc_") for p in k) else "base") for k in tflat})

    def sched(peak):
        return optax.warmup_cosine_decay_schedule(0.0, peak, a.warmup, max(a.steps, a.warmup + 1), peak * 0.1)

    def adamw(peak):
        return optax.adamw(sched(peak), b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10)

    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.multi_transform({"base": adamw(a.lr), "new": adamw(a.lr_new)}, labels))
    # an update with a non-finite gradient is skipped (and counted) instead of corrupting the parameters
    tx = optax.apply_if_finite(tx, max_consecutive_errors=100000)
    opt_state = tx.init(tparams.to_pure_dict())
    state = TS(step=0, params=params, opt_state=opt_state)
    rep_devs = list(mesh.devices.flat)

    def replicate(tree):
        """Host copy of every leaf, then one host-to-card transfer per card (no card-to-card copy)."""
        def one(x):
            x = np.asarray(jax.device_get(x))
            return jax.make_array_from_single_device_arrays(x.shape, rep, [jax.device_put(x, d) for d in rep_devs])
        return jax.tree.map(one, tree)

    state = replicate(state) if a.host_place else jax.device_put(state, rep)

    def key_arg(k):
        """A random key as the jitted functions receive it: raw key data placed from host memory with --host_place."""
        return replicate(np.asarray(jax.device_get(jax.random.key_data(k)))) if a.host_place else k

    def train_step(state, rng, b):
        if a.host_place:
            rng = jax.random.wrap_key_data(rng)
        m = nnx.merge(graphdef, state.params)
        diff = nnx.DiffState(0, trainable)

        def loss_fn(mm, r, bb):
            return losses(mm, r, bb, True, a.lam_loc, a.lam_coord)

        (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(m, rng, b)
        tp = state.params.filter(trainable)
        g_pure, tp_pure = grads.to_pure_dict(), tp.to_pure_dict()
        gn = optax.global_norm(g_pure)
        if a.skip_gn > 0:
            # an exploded but finite gradient would be clipped to norm 1 and applied in a meaningless direction: make it
            # non-finite so that apply_if_finite skips (and counts) it like a non-finite one
            bad = jnp.logical_not(gn < a.skip_gn)
            g_pure = jax.tree.map(lambda x: jnp.where(bad, jnp.full_like(x, jnp.nan), x), g_pure)
        updates, new_opt = tx.update(g_pure, state.opt_state, tp_pure)
        new_tp = optax.apply_updates(tp_pure, updates)
        tp.replace_by_pure_dict(new_tp)
        nnx.update(m, tp)
        info["grad_norm"] = gn
        info["skipped"] = new_opt.total_notfinite.astype(jnp.float32)    # updates skipped so far (non-finite / > skip_gn)
        return TS(step=state.step + 1, params=nnx.state(m), opt_state=new_opt), info

    def eval_step(params, rng, b):
        if a.host_place:
            rng = jax.random.wrap_key_data(rng)
        m = nnx.merge(graphdef, params)
        _, info = losses(m, rng, b, False, a.lam_loc, a.lam_coord)
        return info

    jtrain = jax.jit(train_step, in_shardings=(rep, rep, dsh), out_shardings=(rep, rep), donate_argnums=(0,))
    jeval = jax.jit(eval_step, in_shardings=(rep, rep, dsh), out_shardings=rep)
    keys = ("img", "wrist", "tok", "tmask", "act", "amask", "yb", "vb", "yw", "vw", "coord", "state")
    it = prefetch(data, a.bs, a.seed)
    vrng = np.random.default_rng(123)
    vbatches = [data.batch(vrng.choice(data.val_idx, size=a.bs, replace=False)) for _ in range(4)] \
        if len(data.val_idx) >= a.bs else []
    if a.val_only:
        vi = [jax.device_get(jeval(state.params, key_arg(jax.random.key(7)), jax.device_put({k: vb[k] for k in keys}, dsh)))
              for vb in vbatches]
        vavg = {k: float(np.mean([x[k] for x in vi])) for k in vi[0] if k != "flow_ps"}
        logging.info("VAL0ONLY devices %d host_place %s %s", ndev, a.host_place,
                     " ".join(f"{k}={v:.4f}" for k, v in sorted(vavg.items())))
        return
    hist, hidx, t0 = [], [], time.time()
    spike_f = open(out / "spikes.jsonl", "a")
    for step in range(a.steps):
        b = next(it)
        bj = jax.device_put({k: b[k] for k in keys}, dsh)
        with sharding.set_mesh(mesh):
            state, info = jtrain(state, key_arg(jax.random.fold_in(rng, step)), bj)
        hist.append(info)
        hidx.append(b["idx"])
        if step % a.log_every == 0 or step == a.steps - 1:
            h = jax.device_get(hist)
            for j, (x, ix) in enumerate(zip(h, hidx)):
                fl, gn_ = float(x["flow"]), float(x["grad_norm"])
                if not (fl < a.spike_flow) or not (gn_ < (a.skip_gn if a.skip_gn > 0 else 1e3)):
                    fps = np.asarray(x["flow_ps"], np.float64)
                    spike_f.write(json.dumps({
                        "step": step - len(h) + 1 + j, "flow": fl, "grad_norm": gn_, "skipped": float(x["skipped"]),
                        "frames": [[int(data.root_of[i]), int(data.ep[i]), int(data.t[i]), int(data.init[i]),
                                    int(data.cat[i]), float(f)] for i, f in zip(ix, fps)]}) + "\n")
                    spike_f.flush()
            avg = {k: float(np.mean([x[k] for x in h])) for k in h[0] if k != "flow_ps"}
            hist, hidx = [], []
            logging.info("step %d %s | %.2f s/step", step, " ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items())),
                         (time.time() - t0) / (step + 1))
        if vbatches and (step % a.val_every == 0 or step == a.steps - 1):
            vi = [jax.device_get(jeval(state.params, key_arg(jax.random.key(7)), jax.device_put({k: vb[k] for k in keys}, dsh)))
                  for vb in vbatches]
            vavg = {k: float(np.mean([x[k] for x in vi])) for k in vi[0] if k != "flow_ps"}
            logging.info("VAL step %d %s", step, " ".join(f"{k}={v:.4f}" for k, v in sorted(vavg.items())))
        if (step > 0 and step % a.save_every == 0) or step == a.steps - 1:
            save_trainable(out / f"params_{step + 1}.npz", state.params.filter(trainable))
            logging.info("saved %s", out / f"params_{step + 1}.npz")
        if a.smoke and step >= 3:
            save_trainable(out / "params_smoke.npz", state.params.filter(trainable))
            logging.info("smoke done")
            break


if __name__ == "__main__":
    main()
