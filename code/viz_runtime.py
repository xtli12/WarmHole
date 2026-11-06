"""Runtime internals for the figures: what the policy attends to, and what Wormhole computes at a switch.

For one recorded no-switch rollout under l_old (obs npz from the e0 runs) this script
  (1) attention: at several query steps, runs the policy under l_old and under l_new with shared noise and
      records the attention of the action-expert queries over every prefix key (image patches of each
      camera, instruction tokens), per layer (mean over flow steps and samples);
  (2) wormhole: at one committed query step, samples N continuations under l_old (exits) and under l_new
      (to show absorption), retrieves M entries from the entry memory of l_new, solves the OT problem and
      forms the closed-form bridge chunk exactly as the evaluation client does.
Run in the openpi venv with PYTHONPATH=<overlay>:<code dir>. Output: one npz.
"""
import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import openpi.models.gemma as G
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy_config
from openpi.training import config as _config

import wormhole as W

IMG_TOKENS = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--obs", required=True, help="recorded rollout obs npz (e0, under l_old)")
    ap.add_argument("--traj", required=True, help="matching traj npz (for the executed actions)")
    ap.add_argument("--new_prompt", required=True)
    ap.add_argument("--library", required=True, help="entry memory of l_new in this scene")
    ap.add_argument("--attn_steps", default="0,10,20,30,40")
    ap.add_argument("--switch_step", type=int, default=30)
    ap.add_argument("--n_exit", type=int, default=16)
    ap.add_argument("--n_attn", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = _config.get_config(args.config)
    pol = policy_config.create_trained_policy(cfg, args.ckpt)
    model = pol._model
    H, D = model.action_horizon, model.action_dim
    tok = _tokenizer.PaligemmaTokenizer(cfg.model.max_token_len)
    graphdef, state = nnx.split(model)
    G.BH_CAPTURE["on"] = True

    def sample(state, observation, noise):
        m = nnx.merge(graphdef, state)
        return m.sample_actions(jax.random.key(0), observation, noise=noise)

    j_sample = jax.jit(sample)

    def prep(img, wrist, st, prompts):
        inps = []
        for p in prompts:
            d = {"observation/image": img, "observation/wrist_image": wrist, "observation/state": st, "prompt": p}
            inps.append(pol._input_transform(jax.tree.map(lambda x: x, d)))
        batch = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *inps)
        return inps, _model.Observation.from_dict(batch)

    def run(img, wrist, st, prompts, seeds):
        """Samples one chunk per (prompt, seed); returns env-space actions and the captured attention."""
        P, S = len(prompts), len(seeds)
        inps, obs1 = prep(img, wrist, st, prompts)
        rep = jax.tree.map(lambda x: jnp.repeat(x, S, axis=0), obs1)  # prompt-major
        noise = np.stack([np.random.default_rng(1000 + s).standard_normal((H, D)) for s in seeds] * P)
        G.BH_CAPTURE["sink"] = []
        x0 = np.asarray(j_sample(state, rep, jnp.asarray(noise, jnp.float32)))
        jax.block_until_ready(x0)
        jax.effects_barrier()
        sink = np.stack(G.BH_CAPTURE["sink"])  # (steps*layers, P*S, keys)
        acts = np.zeros((P, S, H, 7), np.float32)
        for p in range(P):
            for s in range(S):
                o = pol._output_transform({"state": np.asarray(inps[p]["state"]), "actions": x0[p * S + s]})
                acts[p, s] = o["actions"][:, :7]
        return acts, sink, obs1

    z = np.load(args.obs, allow_pickle=True)
    tr = np.load(args.traj)
    p_old = str(z["prompt_old"])
    prompts = [p_old, args.new_prompt]
    t_rec = np.asarray(z["t"])
    res = {"prompts": np.array(prompts), "t_rec": t_rec}

    # ---------------- (1) attention over prefix keys ----------------
    steps = [int(s) for s in args.attn_steps.split(",")]
    attn, imgs, wrists, kept = [], [], [], []
    tok_ids = None
    for st_ in steps:
        k = int(np.argmin(np.abs(t_rec - st_)))
        acts, sink, obs1 = run(z["img"][k], z["wrist"][k], z["state"][k], prompts, list(range(args.n_attn)))
        n_calls = sink.shape[0]
        n_steps = 10
        L = n_calls // n_steps
        per = sink.reshape(n_steps, L, len(prompts), args.n_attn, -1).mean(axis=(0, 3))  # (L, P, keys)
        attn.append(per.astype(np.float32))
        imgs.append(z["img"][k])
        wrists.append(z["wrist"][k])
        kept.append(int(t_rec[k]))
        if tok_ids is None:
            tok_ids = np.asarray(obs1.tokenized_prompt)  # (P, Ltok)
            tok_mask = np.asarray(obs1.tokenized_prompt_mask)
            n_img = IMG_TOKENS * len(obs1.images)
            img_names = list(obs1.images.keys())
            img_masks = {kk: bool(np.asarray(v)[0]) for kk, v in obs1.image_masks.items()}
        print(f"attention at t={t_rec[k]}: layers={L}, keys={per.shape[-1]}", flush=True)
    pieces = [[tok._tokenizer.id_to_piece(int(i)) for i in row] for row in tok_ids]
    res.update(attn=np.stack(attn), attn_t=np.array(kept), attn_img=np.stack(imgs), attn_wrist=np.stack(wrists),
               tok_ids=tok_ids, tok_mask=tok_mask, tok_pieces=np.array(pieces, dtype=object), n_img=n_img,
               img_names=np.array(img_names), img_masks=np.array([img_masks[n] for n in img_names]))
    print("prompt pieces:", [pc for pc, m in zip(pieces[1], tok_mask[1]) if m][:40], flush=True)

    # ---------------- (2) wormhole internals at a committed state ----------------
    k = int(np.argmin(np.abs(t_rec - args.switch_step)))
    ts = int(t_rec[k])
    seeds = list(range(args.n_exit))
    acts, _, _ = run(z["img"][k], z["wrist"][k], z["state"][k], prompts, seeds)
    X_old, X_new = acts[0][:, :10].astype(np.float64), acts[1][:, :10].astype(np.float64)
    stv = np.asarray(z["state"][k], np.float64)
    p, g = stv[:3], float(stv[6] - stv[7])
    lib = W.EntryLibrary(args.library, 10)
    Y, wY, idx = lib.retrieve(p, g, m=32)
    a_prev = np.asarray(tr["act"][ts - 1], np.float64) if ts >= 1 else np.zeros(7)
    x_hat, d_ot, plan = W.ot_axis(X_old, Y, wY, a_prev, eta=0.1, anchor=True, use_ot=True)
    br, _, _ = W.bridge_chunk(X_old, Y, wY, a_prev, eta=0.1, ramp=3)
    # same quantities at the start of the episode, where the instruction is still read
    acts0, _, _ = run(z["img"][0], z["wrist"][0], z["state"][0], prompts, seeds)
    res.update(ts=ts, ee=p, grip=g, X_old=X_old, X_new=X_new, Y=Y, wY=wY, Y_states=lib.states[idx], plan=plan,
               x_hat=x_hat, d_ot=d_ot, bridge=br, a_prev=a_prev, X0_old=acts0[0][:, :10], X0_new=acts0[1][:, :10],
               ee0=np.asarray(z["state"][0][:3], np.float64), img_switch=z["img"][k], ee_traj=tr["ee"],
               new_obj=tr["new_obj"][0] if "new_obj" in tr.files else np.full(3, np.nan),
               old_obj=tr["old_obj"][0])
    diff_c = float(np.linalg.norm(X_new.mean(0) - X_old.mean(0)))
    diff_0 = float(np.linalg.norm(acts0[1].mean(0) - acts0[0].mean(0)))
    print(f"t={ts}: |mean new - mean old| = {diff_c:.3f} (at t=0: {diff_0:.3f}); |d_OT| = {np.linalg.norm(d_ot):.3f}",
          flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **res)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
