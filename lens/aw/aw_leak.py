"""No-leak check of the wormhole-exclusive kinds (CPU, dummy widths, random weights).

For awa (attention-only wormhole) the actions may depend on the instruction ONLY through the attention bias of the
grounding map; for awx only through [LOC] (its key/value and the bias).  Two instructions of different token lengths,
the same images and the same flow noise:
  awa, bias closed (worm_scale 0)            -> actions must be identical
  awx, [LOC] hidden and bias closed           -> identical
  awa / awx with the wormhole open            -> different (the instruction does reach the actions)
  aw  (instruction tokens visible)            -> different
The wormhole gates are set to non-zero values (they are 0 at initialisation), and so is every other zero-initialised
weight: at initialisation pi0.5's adaRMS gates are 0, so the action expert would ignore all keys and every check would
trivially give identical actions.
usage: aw_leak.py"""
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer

import aw_model as AW

PROMPTS = ("pick up the black bowl on the stove and place it on the plate",
           "pick up the black bowl on the wooden cabinet and place it on the plate")


def build(kind):
    cfg = AW.aw_config(kind, "pi05", paligemma_variant="dummy", action_expert_variant="dummy", loc_layers=(1, 2),
                       loc_heads=(0, 1))
    model = cfg.create(jax.random.key(0))
    gd, st = nnx.split(model)
    flat = traverse_util.flatten_dict(st.to_pure_dict())
    rng = np.random.default_rng(1)
    for k in flat:
        v = np.asarray(flat[k])
        if k[-1] in ("worm_gamma", "worm_locgate"):
            flat[k] = jnp.asarray(rng.uniform(0.5, 1.5, v.shape), jnp.float32)
        elif v.size and not np.any(v):
            # zero-initialised weights (LoRA B, the adaRMS modulation of pi0.5 -- whose zero gates would make the
            # action expert ignore every key --, biases): random values, so that a dependence can show up at all
            flat[k] = jnp.asarray(rng.normal(0.0, 0.05, v.shape), v.dtype)
    st.replace_by_pure_dict(traverse_util.unflatten_dict(flat))
    return cfg, nnx.merge(gd, st)


def obs_for(cfg, prompt, imgs):
    tok = _tokenizer.PaligemmaTokenizer(cfg.max_token_len)
    t, m = tok.tokenize(prompt)
    return _model.Observation(images=imgs, image_masks={k: jnp.ones((1,), bool) for k in imgs},
                              state=jnp.zeros((1, 32), jnp.float32), tokenized_prompt=jnp.asarray(t[None], jnp.int32),
                              tokenized_prompt_mask=jnp.asarray(m[None], bool))


def main():
    rng = np.random.default_rng(0)
    imgs = {k: jnp.asarray(rng.uniform(-1, 1, (1, 224, 224, 3)), jnp.float32) for k in AW.CAMS}
    noise = jnp.asarray(rng.standard_normal((1, 10, 32)), jnp.float32)
    tok = _tokenizer.PaligemmaTokenizer(48)
    print("instruction tokens:", [int(tok.tokenize(p)[1].sum()) for p in PROMPTS])
    checks = [("awa", dict(worm_scale=0.0), "same"), ("awx", dict(worm_scale=0.0, hide_loc=True), "same"),
              ("awa", dict(), "differ"), ("awx", dict(), "differ"), ("awx", dict(worm_scale=0.0), "differ"),
              ("aw", dict(), "differ")]
    models, ok = {}, True
    for kind, kw, want in checks:
        if kind not in models:
            models[kind] = build(kind)
        cfg, model = models[kind]
        outs = [np.asarray(model.sample_actions(jax.random.key(0), obs_for(cfg, p, imgs), noise=noise, **kw))
                for p in PROMPTS]
        d = float(np.abs(outs[0] - outs[1]).max())
        good = (d < 1e-6) if want == "same" else (d > 1e-4)
        ok &= good
        print(f"{kind:4s} {str(kw):40s} max|a(A)-a(B)| = {d:.3e}  expected {want:6s} -> {'OK' if good else 'FAIL'}",
              flush=True)
    print("leak check", "passed" if ok else "FAILED")


if __name__ == "__main__":
    main()
