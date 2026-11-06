"""Offline mechanistic analysis on recorded no-switch rollouts (openpi venv + patched overlay).

For every recorded policy query of a rollout under l_old it computes, with shared flow noise,
  * action chunks under l_old and under each paired l_new  -> instruction response Delta_theta
  * attention mass of the action-expert queries on the instruction tokens (alpha_L), on all
    language tokens and on image tokens
  * instruction-free prefix features (empty prompt) for the task-identifiability probe
Outputs one npz per rollout in <out>.
"""
import argparse
import glob
import json
import os
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import openpi.models.gemma as G
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy_config
from openpi.training import config as _config

IMG_TOKENS = 256


def wait_for_gpu(min_free_mb=None, poll=60):
    """The analysis shares the GPU with the policy servers; start only when enough memory is free."""
    if min_free_mb is None:
        min_free_mb = int(os.environ.get("ANA_MIN_FREE_MB", "15000"))
    import subprocess
    while True:
        try:
            gid = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            free = int(subprocess.check_output(
                ["nvidia-smi", f"--id={gid}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).split()[0])
        except Exception:  # noqa: BLE001
            free = 0
        if free >= min_free_mb:
            print(f"GPU free {free} MB -> start", flush=True)
            return
        time.sleep(poll)


def main():
    wait_for_gpu()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--obs_glob", required=True)
    ap.add_argument("--pairs", required=True, help="json: {episode-id-substring: [new prompt, ...]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsamples", type=int, default=4)
    ap.add_argument("--every", type=int, default=1, help="use every k-th recorded query")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(open(args.pairs).read())

    cfg = _config.get_config(args.config)
    pol = policy_config.create_trained_policy(cfg, args.ckpt)
    model = pol._model
    H, D = model.action_horizon, model.action_dim
    tok = _tokenizer.PaligemmaTokenizer(cfg.model.max_token_len)
    graphdef, state = nnx.split(model)

    G.BH_CAPTURE["on"] = True  # read at trace time: the suffix attention reports its probabilities

    def sample(state, observation, noise):
        m = nnx.merge(graphdef, state)
        return m.sample_actions(jax.random.key(0), observation, noise=noise)

    def prefix_feats(state, observation):
        m = nnx.merge(graphdef, state)
        o = _model.preprocess_observation(None, observation, train=False)
        tokens, mask, ar = m.embed_prefix(o)
        attn = _pi0.make_attn_mask(mask, ar)
        pos = jnp.cumsum(mask, axis=1) - 1
        (pout, _), _ = m.PaliGemma.llm([tokens, None], mask=attn, positions=pos)
        base = pout[:, :IMG_TOKENS].mean(axis=1)
        wrist = pout[:, IMG_TOKENS:2 * IMG_TOKENS].mean(axis=1)
        return jnp.concatenate([base, wrist], axis=-1)

    j_sample = jax.jit(sample)
    j_prefix = jax.jit(prefix_feats)

    def span(prompt):
        text = prompt.strip().replace("_", " ").replace("\n", " ")
        return 1, 1 + len(tok._tokenizer.encode(text))

    def prep(img, wrist, st, prompts):
        inps = []
        for p in prompts:
            d = {"observation/image": img, "observation/wrist_image": wrist, "observation/state": st, "prompt": p}
            inps.append(pol._input_transform(jax.tree.map(lambda x: x, d)))
        batch = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *inps)
        return inps, _model.Observation.from_dict(batch)

    files = sorted(glob.glob(args.obs_glob))
    print(f"{len(files)} recorded rollouts", flush=True)
    t0 = time.time()
    for fi, f in enumerate(files):
        eid = pathlib.Path(f).stem
        dst = out / f"{eid}.npz"
        if dst.exists():
            continue
        news = None
        for key, v in pairs.items():
            if key in eid:
                news = v
                break
        if news is None:
            continue
        z = np.load(f, allow_pickle=True)
        p_old = str(z["prompt_old"])
        prompts = [p_old] + list(news)
        P = len(prompts)
        S = args.nsamples
        idx = np.arange(0, len(z["t"]), args.every)
        acts = np.zeros((len(idx), P, S, H, 7), np.float32)
        alpha_instr = np.zeros((len(idx), P), np.float32)
        alpha_lang = np.zeros((len(idx), P), np.float32)
        alpha_img = np.zeros((len(idx), P), np.float32)
        feats = []
        for n, k in enumerate(idx):
            img, wrist, st = z["img"][k], z["wrist"][k], z["state"][k]
            inps, obs1 = prep(img, wrist, st, prompts)
            rep = jax.tree.map(lambda x: jnp.repeat(x, S, axis=0), obs1)  # order: prompt-major
            noise = np.stack([np.random.default_rng(1000 + s).standard_normal((H, D)) for s in range(S)] * P)
            G.BH_CAPTURE["sink"] = []
            x0 = np.asarray(j_sample(state, rep, jnp.asarray(noise, jnp.float32)))
            jax.block_until_ready(x0)
            jax.effects_barrier()
            sink = np.stack(G.BH_CAPTURE["sink"])  # (layers*steps, P*S, keys)
            L = z_len = obs1.tokenized_prompt.shape[1]
            n_img = IMG_TOKENS * len(obs1.images)
            for p in range(P):
                rows = sink[:, p * S:(p + 1) * S, :].mean(axis=(0, 1))
                lo, hi = span(prompts[p])
                alpha_instr[n, p] = rows[n_img + lo:n_img + hi].sum()
                alpha_lang[n, p] = rows[n_img:n_img + L].sum()
                alpha_img[n, p] = rows[:n_img].sum()
                for s in range(S):
                    o = pol._output_transform({"state": np.asarray(inps[p]["state"]), "actions": x0[p * S + s]})
                    acts[n, p, s] = o["actions"][:, :7]
            _, obs0 = prep(img, wrist, st, [""])
            feats.append(np.asarray(j_prefix(state, obs0))[0].astype(np.float16))
        np.savez_compressed(dst, t=z["t"][idx], acts=acts, alpha_instr=alpha_instr, alpha_lang=alpha_lang,
                            alpha_img=alpha_img, feats=np.stack(feats), prompts=np.array(prompts))
        el = time.time() - t0
        print(f"[{fi+1}/{len(files)}] {eid}: {len(idx)} queries, {el/60:.1f} min elapsed", flush=True)
    print("analysis done", flush=True)


if __name__ == "__main__":
    main()
