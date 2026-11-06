"""Policy server for the instruction black-hole experiments.

Runs in the openpi venv with the patched overlay first on PYTHONPATH (the overlay lets the
Gemma attention accept an additive float mask). Speaks the standard openpi websocket protocol;
an optional dict under the key "bh" in each request selects the sampling mode:
  {"noise_seed": int}                         deterministic flow noise
  {"mode": "cig", "prompt_old": str, "w": w}  contrastive instruction guidance
  {"mode": "bias", "beta": b}                 additive attention bias from action queries
                                              to the instruction tokens
"""
import argparse
import logging
import time

import einops
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

BIG_NEG = -2.3819763e38
IMG_TOKENS = 256


def sample_custom(model, observation, noise, w, key_bias, cig=False, num_steps=10):
    """Flow sampling (same integrator as Pi0.sample_actions) with
    - optional contrastive guidance when the batch holds [new, old] conditions:
      v = v_new + w * (v_new - v_old), applied to a single shared latent;
    - an additive bias on the attention logits from the action-expert queries to prefix keys."""
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    b = observation.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time = carry
        xb = jnp.broadcast_to(x_t, (b,) + x_t.shape[1:]) if cig else x_t
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, xb, jnp.broadcast_to(time, b))
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        p_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full = jnp.concatenate([p_mask, suffix_attn_mask], axis=-1)
        bias = jnp.concatenate([key_bias, jnp.zeros((b, suffix_tokens.shape[1]), jnp.float32)], axis=-1)
        fmask = jnp.where(full, bias[:, None, :], BIG_NEG).astype(jnp.float32)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens], mask=fmask, positions=pos, kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        v = model.action_out_proj(suffix_out[:, -model.action_horizon:])
        if cig:
            v = v[0:1] + w * (v[0:1] - v[1:2])
        return x_t + dt * v, time + dt

    def cond(carry):
        return carry[1] >= -dt / 2

    x0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x0


class BHPolicy:
    def __init__(self, config_name, ckpt_dir):
        self.cfg = _config.get_config(config_name)
        self.policy = policy_config.create_trained_policy(self.cfg, ckpt_dir)
        self.model = self.policy._model
        self.H, self.D = self.model.action_horizon, self.model.action_dim
        graphdef, state = nnx.split(self.model)

        def fun(state, observation, noise, w, key_bias, cig):
            return sample_custom(nnx.merge(graphdef, state), observation, noise, w, key_bias, cig=cig)

        self._state = state
        self._jfn = jax.jit(fun, static_argnames=("cig",))
        self.tok = _tokenizer.PaligemmaTokenizer(self.cfg.model.max_token_len)
        self._rng = np.random.default_rng(0)

    @property
    def metadata(self):
        return self.policy.metadata

    def noise(self, seed):
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        return rng.standard_normal((1, self.H, self.D)).astype(np.float32)

    def instr_span(self, prompt):
        """Positions (within the language block) of the instruction tokens, after BOS."""
        text = prompt.strip().replace("_", " ").replace("\n", " ")
        return 1, 1 + len(self.tok._tokenizer.encode(text))

    def prepare(self, obs, prompts):
        inps = []
        for p in prompts:
            o = dict(obs)
            o["prompt"] = p
            inps.append(self.policy._input_transform(jax.tree.map(lambda x: x, o)))
        batch = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *inps)
        return inps, _model.Observation.from_dict(batch)

    def run_custom(self, obs, prompts, noise, w=0.0, beta=0.0, cig=False):
        inps, observation = self.prepare(obs, prompts)
        L = observation.tokenized_prompt.shape[1]
        P = IMG_TOKENS * len(observation.images) + L
        key_bias = np.zeros((len(prompts), P), np.float32)
        if beta:
            lo, hi = self.instr_span(prompts[0])
            key_bias[:, P - L + lo:P - L + hi] = beta
        x0 = np.asarray(self._jfn(self._state, observation, jnp.asarray(noise), jnp.float32(w),
                                  jnp.asarray(key_bias), cig=cig))
        outs = [self.policy._output_transform({"state": np.asarray(inps[0]["state"]), "actions": x})["actions"]
                for x in x0]
        return {"actions": outs[0], "samples": np.stack(outs)}

    def infer(self, obs):
        ctl = obs.pop("bh", None) or {}
        mode = ctl.get("mode", "plain")
        noise = self.noise(ctl.get("noise_seed"))
        t0 = time.monotonic()
        if mode == "plain":
            out = self.policy.infer(obs, noise=noise[0])
        elif mode == "cig":
            out = self.run_custom(obs, [obs["prompt"], ctl["prompt_old"]], noise, w=float(ctl.get("w", 1.0)), cig=True)
        elif mode == "multi":
            n = int(ctl.get("n", 8))
            seeds = [int(ctl.get("noise_seed", 0)) * 1000 + k for k in range(n)]
            nz = np.concatenate([self.noise(sd) for sd in seeds], axis=0)
            prompt = ctl.get("prompt", obs["prompt"])
            out = self.run_custom(obs, [prompt] * n, nz)
        elif mode == "bias":
            out = self.run_custom(obs, [obs["prompt"]], noise, beta=float(ctl.get("beta", 1.0)))
        elif mode == "custom_plain":
            out = self.run_custom(obs, [obs["prompt"]], noise)
        else:
            raise ValueError(mode)
        out = dict(out)
        out.pop("state", None)
        out["policy_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8810)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    import openpi
    logging.info("openpi package: %s", openpi.__file__)
    pol = BHPolicy(args.config, args.ckpt)
    if args.selftest:
        rng = np.random.default_rng(1)
        obs = {"observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
               "observation/wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
               "observation/state": rng.standard_normal(8).astype(np.float32),
               "prompt": "put the bowl on the plate"}
        a = pol.infer({**obs, "bh": {"noise_seed": 3}})["actions"]
        b = pol.infer({**obs, "bh": {"noise_seed": 3, "mode": "custom_plain"}})["actions"]
        c = pol.infer({**obs, "bh": {"noise_seed": 3, "mode": "cig", "prompt_old": "turn on the stove", "w": 0.0}})["actions"]
        d = pol.infer({**obs, "bh": {"noise_seed": 3, "mode": "bias", "beta": 0.0}})["actions"]
        e = pol.infer({**obs, "bh": {"noise_seed": 3, "mode": "bias", "beta": 2.0}})["actions"]
        f = pol.infer({**obs, "bh": {"noise_seed": 3, "mode": "cig", "prompt_old": "turn on the stove", "w": 1.0}})["actions"]
        g = pol.infer({**obs, "bh": {"noise_seed": 3, "mode": "multi", "n": 4}})
        print("multi samples", g["samples"].shape, "spread", float(g["samples"].std(axis=0).mean()), flush=True)
        for name, x in [("custom_plain", b), ("cig w=0", c), ("bias 0", d), ("bias 2", e), ("cig w=1", f)]:
            print(f"selftest max|plain - {name}| = {float(np.abs(a - x).max()):.5f}", flush=True)
        for k in range(3):
            t0 = time.time()
            pol.infer({**obs, "bh": {"noise_seed": k}})
            print("plain infer s", round(time.time() - t0, 3), flush=True)
        for k in range(3):
            t0 = time.time()
            pol.infer({**obs, "bh": {"noise_seed": k, "mode": "cig", "prompt_old": "turn on the stove", "w": 1.0}})
            print("cig infer s", round(time.time() - t0, 3), flush=True)
        return
    # compile every sampling mode before accepting clients: a JIT compile inside a request would
    # block the event loop long enough for the clients' websocket keepalive to time out
    rng = np.random.default_rng(0)
    dummy = {"observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
             "observation/wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
             "observation/state": rng.standard_normal(8).astype(np.float32), "prompt": "put the bowl on the plate"}
    for ctl in ({"noise_seed": 1}, {"mode": "multi", "n": 16, "noise_seed": 1},
                {"mode": "cig", "prompt_old": "turn on the stove", "w": 1.0}, {"mode": "bias", "beta": 2.0}):
        t0 = time.time()
        pol.infer({**dummy, "bh": dict(ctl)})
        logging.info("warm-up %s: %.1fs", ctl.get("mode", "plain"), time.time() - t0)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=pol, host="127.0.0.1", port=args.port, metadata=pol.metadata)
    logging.info("serving %s on port %d", args.config, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
