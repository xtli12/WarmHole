"""Attention lens: shared pieces for the attention-guided escape (openpi venv + patched overlay).

The idea: inside the black hole the new instruction is still read, but it no longer changes *where in the image*
the action expert looks.  The lens (1) measures the attention of the action-expert queries over the image patches
under the new and under the old instruction, (2) turns the difference -- measured now, remembered from an attentive
moment, or predicted by a learned focus network -- into an additive bias on the attention logits from the action
queries to the image keys, and (3) re-generates the action chunk under that bias.

Token layout of the openpi pi0 / pi0.5 prefix for LIBERO: 3 x 256 image tokens (base camera, left wrist, masked right
wrist), then the language tokens.  The bias is applied to the first 512 keys (base + wrist); language and suffix keys
are left untouched.
"""
import math

import einops
import jax
import jax.numpy as jnp
import numpy as np
import openpi.models.gemma as G
from openpi.models import model as _model
from openpi.models import pi0 as _pi0

from lens_geom import (ACT_SCALE_CM, GRID, IMG_TOKENS, N_IMG, PATCH, RES, displacement_cm,  # noqa: F401
                       object_map, patch_centers, project_to_policy_image, robust_unit)

BIG_NEG = -2.3819763e38


def flow(model, observation, noise, key_bias, num_steps=10, vbias=None):
    """Flow sampling (the integrator of Pi0.sample_actions) with an additive bias on the attention logits from the
    action-expert queries to the keys; key_bias: (b, P) over the prefix keys, or (b, P + n_suffix) over prefix and
    suffix keys (P = number of prefix tokens; the pi0 suffix starts with its state token)."""
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    b = observation.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, b))
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        p_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full = jnp.concatenate([p_mask, suffix_attn_mask], axis=-1)
        if key_bias.shape[1] == prefix_tokens.shape[1]:      # prefix keys only: no bias on the suffix keys
            bias = jnp.concatenate([key_bias, jnp.zeros((b, suffix_tokens.shape[1]), jnp.float32)], axis=-1)
        else:                                                # bias over prefix and suffix keys (pi0 state token)
            bias = key_bias
        fmask = jnp.where(full, bias[:, None, :], BIG_NEG).astype(jnp.float32)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens], mask=fmask, positions=pos, kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        v = model.action_out_proj(suffix_out[:, -model.action_horizon:])
        if vbias is not None:          # guidance: over the integration the final actions move by +vbias
            v = v - vbias
        return x_t + dt * v, time + dt

    def cond(carry):
        return carry[1] >= -dt / 2

    x0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x0


def prefix_image_features(model, observation):
    """Instruction-free contextual image features: the backbone's outputs at the image positions for the prefix
    built with the given (empty) prompt.  Returns (b, 512, width)."""
    o = _model.preprocess_observation(None, observation, train=False)
    tokens, mask, ar = model.embed_prefix(o)
    attn = _pi0.make_attn_mask(mask, ar)
    pos = jnp.cumsum(mask, axis=1) - 1
    (pout, _), _ = model.PaliGemma.llm([tokens, None], mask=attn, positions=pos)
    return pout[:, :N_IMG]


def prefix_pass(model, observation):
    """Backbone pass over the prefix only (used with the prefix attention capture of the overlay)."""
    o = _model.preprocess_observation(None, observation, train=False)
    tokens, mask, ar = model.embed_prefix(o)
    attn = _pi0.make_attn_mask(mask, ar)
    pos = jnp.cumsum(mask, axis=1) - 1
    (pout, _), _ = model.PaliGemma.llm([tokens, None], mask=attn, positions=pos)
    return pout.astype(jnp.float32).mean(axis=(1, 2))


class Lens:
    """The frozen policy plus jitted samplers with and without attention capture."""

    def __init__(self, config_name=None, ckpt_dir=None, policy=None, cfg=None):
        """Either load the checkpoint (config_name, ckpt_dir) or reuse an already loaded openpi policy."""
        from flax import nnx
        from openpi.policies import policy_config
        from openpi.training import config as _config
        from openpi.models import tokenizer as _tokenizer

        self.cfg = cfg if cfg is not None else _config.get_config(config_name)
        self.policy = policy if policy is not None else policy_config.create_trained_policy(self.cfg, ckpt_dir)
        self.model = self.policy._model
        self.H, self.D = self.model.action_horizon, self.model.action_dim
        self.tok = _tokenizer.PaligemmaTokenizer(self.cfg.model.max_token_len)
        graphdef, state = nnx.split(self.model)
        self._state = state

        def make():  # distinct function objects: separate trace caches for the capturing and the plain sampler
            def fun(state, observation, noise, key_bias):
                return flow(nnx.merge(graphdef, state), observation, noise, key_bias)
            return fun

        def feats(state, observation):
            return prefix_image_features(nnx.merge(graphdef, state), observation)

        def ppass(state, observation):
            return prefix_pass(nnx.merge(graphdef, state), observation)

        def guided(state, observation, noise, key_bias, vbias):
            return flow(nnx.merge(graphdef, state), observation, noise, key_bias, vbias=vbias)

        self._j_cap = jax.jit(make())
        self._j_plain = jax.jit(make())
        self._j_feats = jax.jit(feats)
        self._j_pcap = jax.jit(ppass)
        self._j_phead = jax.jit(lambda state, observation: ppass(state, observation))   # own trace cache
        self._j_guided = jax.jit(guided)
        self._alpha = None
        self.pi05 = bool(getattr(self.model, "pi05", False))
        self.n_suffix = self.H + (0 if self.pi05 else 1)     # pi0: state token + action tokens

    # ------------------------------------------------------------------ inputs
    def prepare(self, elems, off=None):
        """elems: list of openpi LIBERO input dicts (observation/image, observation/wrist_image, observation/state,
        prompt); off: optional list (per element) of camera names whose tokens are masked out of the whole model,
        as the unused right wrist camera is.  Returns the transformed inputs and a batched Observation."""
        inps = [self.policy._input_transform(jax.tree.map(lambda x: x, dict(e))) for e in elems]
        if off is not None:
            for inp, cams in zip(inps, off):
                for cam in cams or ():
                    inp["image_mask"][cam] = np.False_
        batch = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *inps)
        return inps, _model.Observation.from_dict(batch)

    def n_prefix(self, observation):
        return IMG_TOKENS * len(observation.images) + observation.tokenized_prompt.shape[1]

    def noise(self, seeds):
        return np.stack([np.random.default_rng(int(s)).standard_normal((self.H, self.D)) for s in seeds]).astype(
            np.float32)

    def to_actions(self, inps, x0):
        return np.stack([self.policy._output_transform({"state": np.asarray(inp["state"]), "actions": np.asarray(x)})
                         ["actions"][:, :7] for inp, x in zip(inps, x0)])

    # ------------------------------------------------------------------ sampling
    def prefix_attention_heads(self, observation):
        """Backbone attention of every head from the language positions (768:) to the base-camera patches (:256):
        (layers, b, heads, n_text, 256).  Needs the per-head capture of the patched overlay."""
        G.BH_CAPTURE["prefix_heads"] = True
        G.BH_CAPTURE["sink_heads"] = []
        try:
            jax.block_until_ready(self._j_phead(self._state, observation))
            jax.effects_barrier()
            out = np.stack(G.BH_CAPTURE["sink_heads"]).astype(np.float32)
        finally:
            G.BH_CAPTURE["prefix_heads"] = False
            G.BH_CAPTURE["sink_heads"] = []
        return out

    def action_scale(self, inp):
        """Per-dimension slope of the output transform (normalised -> robot actions), first 7 dimensions."""
        if self._alpha is None:
            z = np.zeros((self.H, self.D), np.float32)
            a0 = self.policy._output_transform({"state": np.asarray(inp["state"]), "actions": z})["actions"][0, :7]
            a1 = self.policy._output_transform({"state": np.asarray(inp["state"]), "actions": z + 1})["actions"][0, :7]
            self._alpha = np.asarray(a1 - a0, np.float64)
        return self._alpha

    def guidance(self, inp, direction, lam, steps):
        """vbias that moves the first `steps` actions of the chunk by lam * direction (robot action units, xyz)."""
        al = self.action_scale(inp)
        vb = np.zeros((self.H, self.D), np.float32)
        d = np.asarray(direction, np.float64)[:3]
        vb[:int(steps), :3] = (float(lam) * d / al[:3]).astype(np.float32)
        return vb

    def sample_guided(self, observation, noise, key_bias, vbias):
        x0 = self._j_guided(self._state, observation, jnp.asarray(noise, jnp.float32),
                            jnp.asarray(key_bias, jnp.float32), jnp.asarray(vbias, jnp.float32))
        return np.asarray(jax.block_until_ready(x0))

    def sample(self, observation, noise, key_bias, capture=False):
        """Returns raw flow outputs x0 (b, H, D) and, with capture, the attention of the action queries over all
        keys, averaged over heads and queries: (layers x flow steps, b, keys)."""
        G.BH_CAPTURE["on"] = bool(capture)          # read when a sampler is traced for a new input shape
        G.BH_CAPTURE["sink"] = []
        fn = self._j_cap if capture else self._j_plain
        x0 = fn(self._state, observation, jnp.asarray(noise, jnp.float32), jnp.asarray(key_bias, jnp.float32))
        x0 = np.asarray(jax.block_until_ready(x0))
        sink = None
        if capture:
            jax.effects_barrier()
            sink = np.stack(G.BH_CAPTURE["sink"]).astype(np.float32)
        G.BH_CAPTURE["on"] = False
        G.BH_CAPTURE["sink"] = []
        return x0, sink

    def image_features(self, observation):
        return np.asarray(self._j_feats(self._state, observation))

    def prefix_attention(self, observation):
        """Backbone attention from the language positions (queries 768:) to the base and wrist patches (keys :512),
        averaged over heads: (layers, b, n_text, 512).  Needs the prefix capture of the patched overlay."""
        G.BH_CAPTURE["prefix"] = True
        G.BH_CAPTURE["sink_prefix"] = []
        try:
            jax.block_until_ready(self._j_pcap(self._state, observation))
            jax.effects_barrier()
            out = np.stack(G.BH_CAPTURE["sink_prefix"]).astype(np.float32)
        finally:
            G.BH_CAPTURE["prefix"] = False
            G.BH_CAPTURE["sink_prefix"] = []
        return out


def word_sets(lens, observation):
    """Language positions of a batch of two prompts (x, y): every valid position ('all') and the positions where the
    two instructions differ ('phrase', longest common prefix / suffix removed)."""
    tm = np.asarray(observation.tokenized_prompt_mask)
    tok = np.asarray(observation.tokenized_prompt)
    sp = lens.tok._tokenizer
    ix, iy = np.where(tm[0])[0], np.where(tm[1])[0]
    px = [sp.id_to_piece(int(i)) for i in tok[0][ix]]
    py = [sp.id_to_piece(int(i)) for i in tok[1][iy]]
    p = 0
    while p < min(len(px), len(py)) and px[p] == py[p]:
        p += 1
    s = 0
    while s < min(len(px), len(py)) - p and px[len(px) - 1 - s] == py[len(py) - 1 - s]:
        s += 1
    ph_x, ph_y = ix[p:len(px) - s], iy[p:len(py) - s]
    if len(ph_x) == 0 or len(ph_y) == 0:      # identical instructions: fall back to every position
        ph_x, ph_y = ix, iy
    return {"all": (ix, iy), "phrase": (ph_x, ph_y)}


class LearnedLens:
    """Non-negative readout of the per-head grounding differences (trained by lens_train_heads.py)."""

    def __init__(self, path):
        z = np.load(path)
        self.w, self.scale = z["w"].astype(np.float32), z["scale"].astype(np.float32)
        self.words = [str(x) for x in z["words"]]

    def map(self, lens, observation):
        """observation: batch (x = new instruction, y = old instruction) -> M (256): where x draws attention that y
        does not, as weighted by the learned heads."""
        PH = lens.prefix_attention_heads(observation)                 # (L, 2, heads, n_text, 256)
        ws = word_sets(lens, observation)
        feats = []
        for w in self.words:
            ix, iy = ws[w]
            feats.append(PH[:, 0][:, :, ix].mean(2) - PH[:, 1][:, :, iy].mean(2))   # (L, heads, 256)
        F = np.stack(feats) / self.scale[..., None]                    # (W, L, heads, 256)
        return np.einsum("wlhp,wlh->p", F, self.w)


# ---------------------------------------------------------------------- maps and biases
def image_attention(sink, rows=None):
    """Attention mass of the action queries on the 512 image keys (base, wrist), mean over layers and flow steps
    (and over the given batch rows).  sink: (calls, b, keys)."""
    a = sink[:, :, :N_IMG].mean(axis=0)
    return a if rows is None else a[rows].mean(axis=0)


def key_bias_from_map(m, n_prefix, beta):
    """Additive logit bias over the prefix keys from a 512-patch map in [-1, 1]."""
    kb = np.zeros(n_prefix, np.float32)
    kb[:N_IMG] = beta * np.asarray(m, np.float32)
    return kb
