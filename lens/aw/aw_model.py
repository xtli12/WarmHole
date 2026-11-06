"""Attention-wormhole pi0.5 (AW-pi0.5): a self-trained instruction-grounding pathway inside the policy.

Mechanism imitated from grounding VLMs (Qwen2.5-VL): the position that has to produce a location attends, in a few
middle-layer "localization heads", to the image patches of the object the instruction refers to, and later layers read
the location out of those patches.  AW-pi0.5 gets the same structure:

  [LOC] token   one learned token appended to the prefix (after the instruction).  It attends to everything before it
                (images + instruction); nothing in the backbone attends to it (causal block), so the computation of all
                original prefix tokens is unchanged.  Its attention over the base / wrist image patches in the
                designated localization heads (`loc_layers` x `loc_heads`) is the grounding map; it is trained to cover
                the object the current instruction refers to (simulator object positions projected into the cameras),
                and a small head reads the target location out of its final state.
  wormhole      the action expert reaches the grounding in two ways: it attends to the [LOC] token at every layer (with
                a learned per-layer logit offset `worm_locgate`), and its attention to the image patches is biased by
                worm_gamma[l] * log(256 * map) (per-layer gates, initialised at 0).  When the instruction changes, the
                [LOC] map moves to the new object at once and the action expert's attention is routed there: the
                chunk is regenerated from the new grounding (truncate-and-reconstruct = re-query on the switch).

With use_loc=False, worm=False the model is exactly pi0.5 (two-pass forward: prefix pass + suffix pass with the KV
cache, which is the same computation as openpi's joint forward), used for the plain LoRA behaviour-cloning control.
Only the two real cameras (base, left wrist) are embedded; the masked third camera of the LIBERO inputs is dropped,
which leaves the outputs unchanged (masked tokens are never attended to and do not advance positions).
c10 (11-01, handover 7.12): contrastive instruction guidance at inference (sample_actions(cig_obs=, cig_w=)): the same
observation with the old instruction gives a second velocity field v_old on the same noisy actions, and every Euler step
uses v + w (v - v_old) (the paper's CIG, Section "inference-time remedies").  For kind awx the instruction reaches the
action expert only through [LOC] (excl_text, excl_img_text), so v - v_old is exactly the contribution of the wormhole
token.  With cig_obs=None (or w = 0) the sampling is that of c9 (identical arithmetic).  Training is unchanged."""
import dataclasses

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
import openpi.models.siglip as _siglip
import openpi.shared.nnx_utils as nnx_utils

import aw_gemma as _awg

CAMS = ("base_0_rgb", "left_wrist_0_rgb")
N_PATCH = 256
LOC_TOKEN_IDS = (256000, 257024)   # PaliGemma <loc0000> ... <loc1023>: initial value of the [LOC] embedding


@dataclasses.dataclass(frozen=True)
class AWConfig(pi0_config.Pi0Config):
    use_loc: bool = True
    worm: bool = True
    coord_head: bool = True
    loc_layers: tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12, 13)
    loc_heads: tuple[int, ...] = (0, 1, 2, 3)
    worm_clip: float = 4.0
    # wormhole-exclusive language pathway ('awx'): the action expert does not see the instruction tokens, and the image
    # tokens do not read the instruction in the backbone -- the instruction reaches the actions only through [LOC]
    excl_text: bool = False
    excl_img_text: bool = False
    # grounding map of the designated heads: 'mass' = attention mass pooled over the heads, then normalised per camera
    # (the trained 'aw' model); 'head' = every head's attention renormalised over each camera's patches, then averaged
    # (numerically safe when the heads look mostly at the instruction, as [LOC] must in 'awx')
    loc_norm: str = "mass"
    # attention-only wormhole ('awa'): the action expert does not see the [LOC] key/value either, so the instruction can
    # reach the actions only by re-weighting the action expert's attention over the image patches (the grounding map)
    excl_loc: bool = False
    # one wormhole gate per attention head of the action expert (instead of one per layer)
    worm_heads: bool = False

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05 if self.pi05 else _model.ModelType.PI0

    @override
    def create(self, rng) -> "AWPi0":
        return AWPi0(self, rngs=nnx.Rngs(rng))

    def trainable_filter(self) -> nnx.filterlib.Filter:
        """LoRA adapters of both experts, the wormhole gates, the [LOC] token and head, and the small action/time
        projections; SigLIP and all pretrained transformer weights stay frozen."""
        return nnx.All(
            nnx.Param,
            nnx.Any(
                nnx_utils.PathRegex(".*lora.*"),
                nnx_utils.PathRegex(".*worm_.*"),
                nnx_utils.PathRegex("loc_.*"),
                nnx_utils.PathRegex("(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|state_proj|"
                                    "action_time_mlp_in|action_time_mlp_out)/.*"),
            ),
        )


def aw_config(kind: str, bb: str = "pi05", **kw) -> AWConfig:
    """kind: 'aw' (full attention wormhole), 'awx' (attention wormhole as the only language pathway into the action
    expert: [LOC] key/value + attention bias), 'awa' (attention-only wormhole: the instruction reaches the action expert
    only through the attention bias of the grounding map), 'loc' ([LOC] token + grounding loss, no attention bias),
    'bc' (plain LoRA behaviour cloning, no [LOC], no bias), 'base' (the released policy, no adapters).  bb: pi05 | pi0."""
    if bb == "pi05":
        base = dict(pi05=True, action_horizon=10, discrete_state_input=False, max_token_len=48)
    elif bb == "pi0":
        base = dict(pi05=False, action_horizon=50, discrete_state_input=False, max_token_len=48)
    else:
        raise ValueError(bb)
    base.update(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    if kind == "base":           # the released pi0.5 itself (no adapters): mechanism analysis of the original model
        base.update(use_loc=False, worm=False, coord_head=False, paligemma_variant="gemma_2b",
                    action_expert_variant="gemma_300m")
    elif kind == "aw":
        base.update(use_loc=True, worm=True)
    elif kind == "awx":
        base.update(use_loc=True, worm=True, excl_text=True, excl_img_text=True, loc_norm="head", worm_heads=True)
    elif kind == "awa":
        base.update(use_loc=True, worm=True, excl_text=True, excl_img_text=True, loc_norm="head", worm_heads=True,
                    excl_loc=True)
    elif kind == "loc":
        base.update(use_loc=True, worm=False)
    elif kind == "bc":
        base.update(use_loc=False, worm=False, coord_head=False)
    else:
        raise ValueError(kind)
    base.update(kw)
    return AWConfig(**base)


class AWPi0(_model.BaseModel):
    def __init__(self, config: AWConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = bool(config.pi05)
        self.n_suffix = config.action_horizon + (0 if self.pi05 else 1)     # pi0: one state token before the actions
        self.use_loc = bool(config.use_loc)
        self.use_worm = bool(config.worm and config.use_loc)
        self.use_coord = bool(config.coord_head and config.use_loc)
        self.loc_layers = tuple(int(x) for x in config.loc_layers)
        self.loc_heads = tuple(int(x) for x in config.loc_heads)
        self.worm_clip = float(config.worm_clip)
        self.excl_text = bool(config.excl_text)
        self.excl_img_text = bool(config.excl_img_text)
        self.loc_norm = str(config.loc_norm)
        self.excl_loc = bool(config.excl_loc and config.use_loc)
        pg = _awg.get_config(config.paligemma_variant)
        ae = _awg.get_config(config.action_expert_variant)
        llm = nnx_bridge.ToNNX(
            _awg.Module(configs=[pg, ae], embed_dtype=config.dtype, adarms=self.pi05, aw=self.use_worm,
                        aw_heads=bool(config.worm_heads), n_read=2 * N_PATCH)
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, self.pi05])
        img = nnx_bridge.ToNNX(
            _siglip.Module(num_classes=pg.width, variant="So400m/14", pool_type="none", scan=True, dtype_mm=config.dtype)
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, ae.width, rngs=rngs)
        if self.pi05:
            self.time_mlp_in = nnx.Linear(ae.width, ae.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(ae.width, ae.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, ae.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * ae.width, ae.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(ae.width, ae.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(ae.width, config.action_dim, rngs=rngs)
        if self.use_loc:
            self.loc_emb = nnx.Param(jax.random.normal(rngs.params(), (pg.width,), jnp.float32))
        if self.use_coord:
            self.loc_head_in = nnx.Linear(pg.width, 512, rngs=rngs)
            self.loc_head_out = nnx.Linear(512, 5, rngs=rngs)
        self.deterministic = True

    # ------------------------------------------------------------------ prefix: images, instruction, [LOC]
    def embed_prefix(self, obs: _model.Observation, img_no_text: bool = False):
        tokens, input_mask, ar_mask = [], [], []
        for name in CAMS:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]
        txt = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(txt)
        input_mask.append(obs.tokenized_prompt_mask)
        # img_no_text (probe): the instruction becomes its own causal block, so image tokens no longer read it
        ar_mask += [bool(img_no_text)] + [False] * (txt.shape[1] - 1)
        if self.use_loc:
            b = txt.shape[0]
            tokens.append(jnp.broadcast_to(self.loc_emb.value.astype(txt.dtype)[None, None, :], (b, 1, txt.shape[-1])))
            input_mask.append(jnp.ones((b, 1), jnp.bool_))
            ar_mask += [True]          # [LOC] sees the whole prefix; no prefix token sees [LOC]
        return jnp.concatenate(tokens, axis=1), jnp.concatenate(input_mask, axis=1), jnp.array(ar_mask)

    def prefix_pass(self, obs: _model.Observation, img_no_text: bool = False):
        pt, pm, par = self.embed_prefix(obs, img_no_text)
        attn = _pi0.make_attn_mask(pm, par)
        pos = jnp.cumsum(pm, axis=1) - 1
        (pout, _), kv, g = self.PaliGemma.llm([pt, None], mask=attn, positions=pos)
        return pm, pout, kv, g

    def ground_maps(self, g):
        """[LOC] attention of the designated localization heads -> per-layer grounding maps over the 16x16 patches of
        the base and the wrist camera, (n_layers, b, 256) each, and the share of [LOC] attention on the images."""
        sel = g[jnp.array(self.loc_layers)]                               # (Ls, b, K, G, S) masked logits
        sel = sel.reshape(sel.shape[0], sel.shape[1], -1, sel.shape[-1])   # (Ls, b, heads, S)
        sel = sel[:, :, jnp.array(self.loc_heads)]                         # (Ls, b, h, S)
        probs = jax.nn.softmax(sel, axis=-1)                               # attention over all prefix keys
        mass = probs[..., : 2 * N_PATCH].sum(-1).mean(-1)                  # (Ls, b)
        if self.loc_norm == "head":
            pb = jax.nn.softmax(sel[..., :N_PATCH], axis=-1).mean(2)
            pw = jax.nn.softmax(sel[..., N_PATCH: 2 * N_PATCH], axis=-1).mean(2)
            return pb, pw, mass
        p = probs[..., : 2 * N_PATCH].sum(2)                               # (Ls, b, 512) mass pooled over heads
        base, wrist = p[..., :N_PATCH] + 1e-6, p[..., N_PATCH:] + 1e-6
        return base / base.sum(-1, keepdims=True), wrist / wrist.sum(-1, keepdims=True), mass

    def wormhole(self, g, pm):
        pb, pw, mass = self.ground_maps(g)
        mb, mw = pb.mean(0), pw.mean(0)                                    # (b, 256)
        c = self.worm_clip
        lb = jnp.clip(jnp.log(N_PATCH * mb + 1e-6), -c, c)
        lw = jnp.clip(jnp.log(N_PATCH * mw + 1e-6), -c, c)
        b, p = pm.shape
        h = self.n_suffix
        img_logmap = jnp.concatenate([lb, lw, jnp.zeros((b, p - 2 * N_PATCH + h), lb.dtype)], axis=1)
        loc_onehot = jnp.zeros((b, p + h), lb.dtype).at[:, p - 1].set(1.0)
        worm = (img_logmap, loc_onehot) if self.use_worm else None
        return worm, {"pb": pb, "pw": pw, "mb": mb, "mw": mw, "mass": mass}

    # ------------------------------------------------------------------ suffix: noisy actions -> velocity
    def embed_suffix(self, state, noisy_actions, timestep):
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = _pi0.posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = nnx.swish(self.time_mlp_in(time_emb))
            time_emb = nnx.swish(self.time_mlp_out(time_emb))
            mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
            ar = jnp.array([True] + [False] * (self.action_horizon - 1))
            return action_tokens, mask, ar, time_emb
        # pi0: a state token, then action tokens mixed with the time embedding by an MLP (no adaRMS)
        state_token = self.state_proj(state)[:, None, :]
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        at = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        at = self.action_time_mlp_out(nnx.swish(self.action_time_mlp_in(at)))
        tokens = jnp.concatenate([state_token, at], axis=1)
        mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        ar = jnp.array([True] + [True] + [False] * (self.action_horizon - 1))
        return tokens, mask, ar, None

    def velocity(self, pm, kv, x_t, time, worm, key_mask=None, state=None):
        st, sm, sar, ada = self.embed_suffix(state, x_t, time)
        s_attn = _pi0.make_attn_mask(sm, sar)
        p_attn = einops.repeat(pm if key_mask is None else key_mask, "b p -> b s p", s=st.shape[1])
        full = jnp.concatenate([p_attn, s_attn], axis=-1)
        if self.excl_text:
            # the action expert must not learn the instruction from its own positions (their offset would otherwise be
            # the number of instruction tokens): fixed positions after the padded prefix
            pos = jnp.full((st.shape[0], 1), pm.shape[1], jnp.int32) + jnp.cumsum(sm, axis=-1) - 1
        else:
            pos = jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
        (_, sout), _, _ = self.PaliGemma.llm([None, st], mask=full, positions=pos, kv_cache=kv,
                                             adarms_cond=[None, ada], worm=worm)
        return self.action_out_proj(sout[:, -self.action_horizon:])

    def text_hidden(self, pm, observation):
        """Key mask of the action expert without the instruction tokens (positions are unchanged)."""
        n_txt = observation.tokenized_prompt.shape[1]
        return pm.at[:, 2 * N_PATCH: 2 * N_PATCH + n_txt].set(False)

    def coord(self, pout):
        h = self.loc_head_in(pout[:, -1].astype(jnp.float32))
        return self.loc_head_out(nnx.gelu(h))

    # ------------------------------------------------------------------ training
    def aw_losses(self, rng, observation: _model.Observation, actions, *, train: bool = True):
        """Per-example losses: flow (b, H) and, with [LOC], the grounding cross-entropies (b,) per camera and the
        target read-out (b, 5).  Targets of the grounding losses are passed by the caller."""
        noise_rng, time_rng = jax.random.split(rng, 2)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions
        pm, pout, kv, g = self.prefix_pass(observation, self.excl_img_text)
        worm, aux = (None, {})
        if self.use_loc:
            worm, aux = self.wormhole(g, pm)
        key_mask = None
        if self.excl_text or self.excl_loc:
            key_mask = self.text_hidden(pm, observation) if self.excl_text else pm
            if self.excl_loc:
                key_mask = key_mask.at[:, -1].set(False)
        v_t = self.velocity(pm, kv, x_t, time, worm, key_mask=key_mask, state=observation.state)
        out = {"flow": jnp.mean(jnp.square(v_t - u_t), axis=-1)}
        if self.use_loc:
            out.update(pb=aux["pb"], pw=aux["pw"], mb=aux["mb"], mw=aux["mw"], mass=aux["mass"])
        if self.use_coord:
            out["coord"] = self.coord(pout)
        return out

    @override
    def compute_loss(self, rng, observation, actions, *, train: bool = False):
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=CAMS)
        return self.aw_losses(rng, observation, actions, train=train)["flow"]

    # ------------------------------------------------------------------ inference
    @override
    def sample_actions(self, rng, observation: _model.Observation, *, num_steps: int = 10, noise=None,
                       return_aux: bool = False, donor: _model.Observation | None = None, worm_scale=1.0,
                       hide_loc: bool = False, hide_text: bool = False, img_no_text: bool = False,
                       cig_obs: _model.Observation | None = None, cig_w=0.0):
        """Flow sampling.  Diagnostics of the wormhole (inference only): worm_scale multiplies the attention bias
        (0 closes it); hide_loc removes the [LOC] key from the action expert's attention (positions unchanged);
        hide_text removes the instruction tokens' keys from the action expert's attention (language can then reach the
        actions only through instruction-conditioned image tokens or [LOC]); img_no_text stops the image tokens from
        reading the instruction in the backbone (then the instruction reaches the actions only through its own tokens
        or [LOC]);
        donor = the same observation with another instruction: its [LOC] state (key/value at every layer) and its
        grounding map are transplanted into this pass, while all image and instruction tokens stay those of
        `observation` (does the action expert follow the wormhole or the instruction tokens?)."""
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=CAMS)
        dt = -1.0 / num_steps
        b = observation.tokenized_prompt.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (b, self.action_horizon, self.action_dim))
        img_no_text = bool(img_no_text or self.excl_img_text)
        hide_text = bool(hide_text or self.excl_text)
        hide_loc = bool(hide_loc or self.excl_loc)
        pm, pout, kv, g = self.prefix_pass(observation, img_no_text)
        if donor is not None and self.use_loc:
            donor = _model.preprocess_observation(None, donor, train=False, image_keys=CAMS)
            _, dpout, dkv, dg = self.prefix_pass(donor, img_no_text)
            kk, vv = kv
            kv = (kk.at[:, :, -1].set(dkv[0][:, :, -1]), vv.at[:, :, -1].set(dkv[1][:, :, -1]))
            pout, g = dpout, dg
        worm, aux = (None, {})
        if self.use_loc:
            worm, aux = self.wormhole(g, pm)
            if worm is not None:
                worm = (worm[0] * worm_scale, worm[1])
        key_mask = None
        if (hide_loc and self.use_loc) or hide_text:
            key_mask = pm
            if hide_loc and self.use_loc:
                key_mask = key_mask.at[:, -1].set(False)
            if hide_text:
                key_mask = self.text_hidden(key_mask, observation)
        cig_on = cig_obs is not None
        if cig_on:     # c10: the old-instruction branch of contrastive instruction guidance (same images, same noise)
            cig_obs = _model.preprocess_observation(None, cig_obs, train=False, image_keys=CAMS)
            pm_o, _, kv_o, g_o = self.prefix_pass(cig_obs, img_no_text)
            worm_o = None
            if self.use_loc:
                worm_o, _ = self.wormhole(g_o, pm_o)
                if worm_o is not None:
                    worm_o = (worm_o[0] * worm_scale, worm_o[1])
            key_mask_o = None
            if (hide_loc and self.use_loc) or hide_text:
                key_mask_o = pm_o
                if hide_loc and self.use_loc:
                    key_mask_o = key_mask_o.at[:, -1].set(False)
                if hide_text:
                    key_mask_o = self.text_hidden(key_mask_o, cig_obs)

        def step(carry):
            x_t, time = carry
            tb = jnp.broadcast_to(time, b)
            v = self.velocity(pm, kv, x_t, tb, worm, key_mask, state=observation.state)
            if cig_on:
                v_o = self.velocity(pm_o, kv_o, x_t, tb, worm_o, key_mask_o, state=cig_obs.state)
                v = v + cig_w * (v - v_o)
            return x_t + dt * v, time + dt

        def cond(carry):
            return carry[1] >= -dt / 2

        x0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        if not return_aux:
            return x0
        info = {}
        if self.use_loc:
            info = {"mb": aux["mb"], "mw": aux["mw"], "mass": aux["mass"].T}
            if self.use_coord:
                info["coord"] = self.coord(pout)
        return x0, info


def load_model(kind, params_path, ckpt, **cfg_kw):
    """Base pi0.5 LIBERO weights with the trained parameters (aw_train.py params_<step>.npz) on top."""
    import pathlib

    import flax.traverse_util as traverse_util
    import numpy as np

    config = aw_config(kind, **cfg_kw)
    model = nnx.eval_shape(config.create, jax.random.key(0))
    graphdef, state = nnx.split(model)
    flat_model = traverse_util.flatten_dict(state.to_pure_dict())
    base = traverse_util.flatten_dict(_model.restore_params(pathlib.Path(ckpt) / "params", dtype=jnp.bfloat16))
    trained = {}
    if params_path:
        z = np.load(params_path)
        trained = {tuple(k.split("/")): z[k] for k in z.files}
    full, missing = {}, []
    for k in flat_model:
        if k in trained:
            full[k] = jnp.asarray(trained[k], jnp.float32)
        elif k in base:
            full[k] = base[k]
        else:
            missing.append("/".join(k))
    if missing:
        raise ValueError(f"parameters neither trained nor in the base checkpoint: {missing[:10]}")
    unused = [k for k in trained if k not in flat_model]
    if unused:
        raise ValueError(f"trained parameters unknown to the model: {unused[:10]}")
    state.replace_by_pure_dict(traverse_util.unflatten_dict(full))
    return config, nnx.merge(graphdef, state)
