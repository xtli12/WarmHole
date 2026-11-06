"""Mechanism analysis: does pi0.5's backbone have localization heads?  Same frames and metrics as qwen_heads.py.

For frames at the instruction switch (steps/<id>.npz), the prefix of pi0.5 (base and wrist view + instruction) is run
once per instruction; the attention of the last prefix query over the 16 x 16 base-camera patches is read for every
layer and head.  For the released model (--kind base) the prompt is not padded, so the last query is the instruction's
"\n" token (pi0.5's start-of-answer position, the analogue of Qwen's answer position); for a trained wormhole model
(--kind aw|loc) it is the [LOC] token.  Writes OUT.npz / OUT.json like qwen_heads.py.
usage: pi_heads.py OUT --frames 'GLOB' --kind base|aw|loc [--params P.npz] [--max 200]"""
import argparse
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer

import aw_model as AW
import qwen_heads as QH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--kind", default="base", choices=["base", "bc", "aw", "awx", "awa", "loc"])
    ap.add_argument("--params", default="")
    ap.add_argument("--ckpt", default=(os.environ.get("BH", "/data/blackhole") + "/openpi_data/openpi-assets/checkpoints/pi05_libero"))
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--when", default="switch", choices=["switch", "start"])
    a = ap.parse_args()
    kw = {}
    cj = os.path.join(os.path.dirname(a.params), "config.json") if a.params else ""
    if cj and os.path.exists(cj):
        c = json.load(open(cj))
        kw = {k: tuple(int(x) for x in c[k].strip("()").split(",") if x.strip()) for k in ("loc_layers", "loc_heads")}
    config, model = AW.load_model(a.kind, a.params, a.ckpt, **kw)
    tok = _tokenizer.PaligemmaTokenizer(1000)
    frames = QH.frames_from_steps(glob.glob(a.frames), a.max, a.when)
    print(f"{len(frames)} frames", flush=True)

    from flax import nnx
    gd, st = nnx.split(model)

    def prefix_maps(st, obs):
        m = nnx.merge(gd, st)
        pm, pout, kv, g = m.prefix_pass(obs, m.excl_img_text)
        return jax.nn.softmax(g, axis=-1)[:, :, 0, :, :256]           # attention probabilities (L, b, heads, 256)

    jf = jax.jit(prefix_maps)
    all_maps, rows = [], []
    for k, fr in enumerate(frames):
        z = np.load([p for p in glob.glob(a.frames) if os.path.basename(p)[:-4] == fr["id"]][0])
        wrist = z["wrist"][fr["t"]]
        per = []
        for instr in fr["instr"]:
            t, msk = tok.tokenize(instr)
            n = int(msk.sum())
            if a.kind in ("base", "bc"):
                t, msk = t[:n], msk[:n]                                 # no padding: last query = "\n"
            else:
                t2 = np.zeros(config.max_token_len, np.int32)
                m2 = np.zeros(config.max_token_len, bool)
                t2[:n], m2[:n] = t[:n], True
                t, msk = t2, m2
            imgs = {"base_0_rgb": jnp.asarray(fr["img"][None], jnp.float32) / 127.5 - 1.0,
                    "left_wrist_0_rgb": jnp.asarray(wrist[None], jnp.float32) / 127.5 - 1.0}
            obs = _model.Observation(images=imgs, image_masks={kk: jnp.ones((1,), bool) for kk in imgs},
                                     state=jnp.zeros((1, 32), jnp.float32),
                                     tokenized_prompt=jnp.asarray(t[None], jnp.int32),
                                     tokenized_prompt_mask=jnp.asarray(msk[None], bool))
            g = np.asarray(jf(st, obs))[:, 0]                           # (L, heads, 256)
            per.append(g / (g.sum(-1, keepdims=True) + 1e-12))
        maps = np.stack(per)
        all_maps.append(maps.astype(np.float16))
        rows.append(dict(s_old=QH.scores(maps[0], fr["uv"][[0, 1]]), s_new=QH.scores(maps[1], fr["uv"][[1, 0]])))
        if k % 20 == 0:
            print(k, fr["id"], flush=True)
    M = np.stack(all_maps)
    P_old = np.stack([r["s_old"]["point"] for r in rows])
    P_new = np.stack([r["s_new"]["point"] for r in rows])
    point, switch = (P_old + P_new) / 2, P_old * P_new
    closer = (np.stack([r["s_old"]["closer"] for r in rows]) + np.stack([r["s_new"]["closer"] for r in rows])) / 2
    L, H = point.shape[1:]
    rank = np.argsort(-point.mean(0).ravel())
    top = [dict(layer=int(i // H), head=int(i % H), point=float(point.mean(0).ravel()[i]),
                switch=float(switch.mean(0).ravel()[i]), closer=float(closer.mean(0).ravel()[i])) for i in rank[:15]]
    c = QH.cell_centers()
    uvo = np.stack([fr["uv"][0] for fr in frames])
    uvn = np.stack([fr["uv"][1] for fr in frames])
    pooled = {}
    for kk in (1, 3, 5, 10):
        sel = rank[:kk]
        po = M[:, 0].astype(np.float32).reshape(len(frames), -1, 256)[:, sel].mean(1)
        pn = M[:, 1].astype(np.float32).reshape(len(frames), -1, 256)[:, sel].mean(1)
        ho = np.linalg.norm(c[po.argmax(-1)] - uvo, axis=-1) < 1.5 * QH.CELL
        hn = np.linalg.norm(c[pn.argmax(-1)] - uvn, axis=-1) < 1.5 * QH.CELL
        pooled[f"top{kk}"] = dict(point_old=float(ho.mean()), point_new=float(hn.mean()), switch=float((ho & hn).mean()))
    if a.kind in ("aw", "awx", "awa", "loc"):
        ll, hh = list(config.loc_layers), list(config.loc_heads)
        sel_l = np.zeros((L, H), bool)
        sel_l[np.ix_(ll, hh)] = True
        def pool_sel(x):  # noqa: E306
            y = x.astype(np.float32)[:, sel_l]
            return y.mean(1)
        po, pn = pool_sel(M[:, 0]), pool_sel(M[:, 1])
        ho = np.linalg.norm(c[po.argmax(-1)] - uvo, axis=-1) < 1.5 * QH.CELL
        hn = np.linalg.norm(c[pn.argmax(-1)] - uvn, axis=-1) < 1.5 * QH.CELL
        pooled["designated"] = dict(point_old=float(ho.mean()), point_new=float(hn.mean()), switch=float((ho & hn).mean()))
    summary = dict(kind=a.kind, params=a.params, n_frames=len(frames), layers=int(L), heads=int(H), top_heads=top,
                   pooled=pooled, best_layer_point=[float(x) for x in point.mean(0).max(1)],
                   chance_point=float(np.pi * 1.5 ** 2 / QH.GRID ** 2))
    np.savez_compressed(a.out + ".npz", maps=M, uv=np.stack([fr["uv"] for fr in frames]),
                        ids=np.array([fr["id"] for fr in frames]))
    json.dump(summary, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
