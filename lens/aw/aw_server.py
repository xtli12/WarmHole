"""Policy server of the self-trained attention-wormhole pi0.5 (and of the plain LoRA control).

Loads the base pi0.5 LIBERO checkpoint, overlays the trained parameters (aw_train.py params_<step>.npz) and serves the
standard openpi websocket protocol with the standard LIBERO input/output transforms (same normalisation as the base
policy).  Request options under "bh": noise_seed (deterministic flow noise), info=True (also return the [LOC]
grounding maps of the base / wrist camera and the target read-out).  No external model, no scripted motion: the
actions are the model's own samples.
c8 (11-01): for pi0 the output transforms include openpi's AbsoluteActions (first 6 dims plus the state), as for the
released pi0_libero (extra_delta_transform=True) and as aw_train.py c10 trains pi0; pi0.5 is unchanged.
c9 (11-01, handover 7.12): request option bh.mode = "cig" with bh.prompt_old and bh.w -- contrastive instruction
guidance (aw_model): the observation is also passed with the old instruction and every flow step uses
v + w (v - v_old); the client sends it after the switch only (method cig).  Everything else is c8.
usage: aw_server.py --kind aw|awx|awa|loc|bc --params PATH.npz --port 8601 [--ckpt BASE_CKPT]"""
import os
import argparse
import json
import logging
import pathlib
import time

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.policies import libero_policy
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints

import aw_model as AW


class AWPolicy:
    def __init__(self, kind, params_path, ckpt, cfg_json=None, model=None, config=None, bb="pi05"):
        self.kind = kind
        kw = {"bb": bb}
        if cfg_json and pathlib.Path(cfg_json).exists():
            c = json.load(open(cfg_json))
            for k in ("loc_layers", "loc_heads"):
                if k in c:
                    kw[k] = tuple(int(x) for x in c[k].strip("()").split(",") if x.strip())
        if model is not None:
            self.config, self.model = config, model
        else:
            self.config, self.model = AW.load_model(kind, params_path, ckpt, **kw)
        logging.info("AW policy %s with trained parameters from %s", kind, params_path)
        norm_stats = _checkpoints.load_norm_stats(pathlib.Path(ckpt) / "assets", "physical-intelligence/libero")
        mc = self.config
        self._quantile = mc.model_type != _model.ModelType.PI0     # openpi: quantile norm for pi0.5, z-score for pi0
        self._in = _transforms.compose([
            libero_policy.LiberoInputs(model_type=mc.model_type),
            _transforms.Normalize(norm_stats, use_quantiles=self._quantile),
            _transforms.InjectDefaultPrompt(None),
            _transforms.ResizeImages(224, 224),
            _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(mc.max_token_len), discrete_state_input=False),
            _transforms.PadStatesAndActions(mc.action_dim),
        ])
        outs = [libero_policy.LiberoOutputs()]
        if mc.model_type == _model.ModelType.PI0:    # openpi pi0_libero: extra_delta_transform=True (c8)
            outs = [_transforms.AbsoluteActions(_transforms.make_bool_mask(6, -1))] + outs
        self._out = _transforms.compose(outs)
        self._unnorm = _transforms.Unnormalize(norm_stats, use_quantiles=self._quantile)
        gd, st = nnx.split(self.model)

        def fn(st, obs, noise, worm_scale, hide_loc, donor, hide_text, img_no_text, cig_obs, cig_w):
            m = nnx.merge(gd, st)
            return m.sample_actions(jax.random.key(0), obs, noise=noise, return_aux=True, worm_scale=worm_scale,
                                    hide_loc=hide_loc, donor=donor, hide_text=hide_text, img_no_text=img_no_text,
                                    cig_obs=cig_obs, cig_w=cig_w)

        self._st = st
        self._jfn = jax.jit(fn, static_argnames=("hide_loc", "hide_text", "img_no_text"))
        self.H, self.D = self.model.action_horizon, self.model.action_dim
        self._rng = np.random.default_rng(0)

    @property
    def metadata(self):
        return {"aw_kind": self.kind}

    def infer(self, obs):
        ctl = obs.pop("bh", None) or {}
        seed = ctl.get("noise_seed")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        noise = rng.standard_normal((1, self.H, self.D)).astype(np.float32)
        t0 = time.monotonic()
        inputs = self._in(jax.tree.map(lambda x: x, obs))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        observation = _model.Observation.from_dict(inputs)
        donor = None
        if ctl.get("donor_prompt"):
            d = dict(obs)
            d["prompt"] = str(ctl["donor_prompt"])
            di = jax.tree.map(lambda x: jnp.asarray(x)[None], self._in(jax.tree.map(lambda x: x, d)))
            donor = _model.Observation.from_dict(di)
        cig, cig_w = None, 0.0
        if ctl.get("mode") == "cig" and ctl.get("prompt_old"):     # c9: contrastive instruction guidance
            d = dict(obs)
            d["prompt"] = str(ctl["prompt_old"])
            ci = jax.tree.map(lambda x: jnp.asarray(x)[None], self._in(jax.tree.map(lambda x: x, d)))
            cig, cig_w = _model.Observation.from_dict(ci), float(ctl.get("w", 1.0))
        x0, info = self._jfn(self._st, observation, jnp.asarray(noise), jnp.float32(ctl.get("worm_scale", 1.0)),
                             hide_loc=bool(ctl.get("hide_loc", False)), donor=donor,
                             hide_text=bool(ctl.get("hide_text", False)), img_no_text=bool(ctl.get("img_no_text", False)),
                             cig_obs=cig, cig_w=jnp.float32(cig_w))
        x0 = np.asarray(x0[0])
        out = self._unnorm({"state": np.asarray(inputs["state"][0]), "actions": x0})
        out = self._out(out)
        res = {"actions": out["actions"], "policy_timing": {"infer_ms": (time.monotonic() - t0) * 1000}}
        if ctl.get("info") and info:
            res["aw"] = {k: np.asarray(v[0]) for k, v in info.items()}
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="aw", choices=["aw", "awx", "awa", "loc", "bc", "base"])
    ap.add_argument("--params", default="")
    ap.add_argument("--bb", default="pi05", choices=["pi05", "pi0"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--config_json", default="")
    ap.add_argument("--port", type=int, default=8601)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    a.ckpt = a.ckpt or f"{os.environ.get('BH', '/data/blackhole')}/openpi_data/openpi-assets/checkpoints/{a.bb}_libero"
    cj = a.config_json or (str(pathlib.Path(a.params).parent / "config.json") if a.params else "")
    pol = AWPolicy(a.kind, a.params, a.ckpt, cj, bb=a.bb)
    rng = np.random.default_rng(0)
    dummy = {"observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
             "observation/wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
             "observation/state": rng.standard_normal(8).astype(np.float32),
             "prompt": "pick up the black bowl on the stove and place it on the plate"}
    for k in range(3):
        t0 = time.time()
        r = pol.infer({**dummy, "bh": {"noise_seed": 1, "info": True}})
        logging.info("warm-up %d: %.2fs actions %s %s", k, time.time() - t0, r["actions"].shape,
                     {kk: np.asarray(v).shape for kk, v in r.get("aw", {}).items()})
    # c9: compile the guided path too, and check it: w = 0 must give exactly the plain actions, w > 0 must not
    old_p = "pick up the black bowl on the wooden cabinet and place it on the plate"
    plain = pol.infer({**dummy, "bh": {"noise_seed": 3}})["actions"]
    for w in (0.0, 1.0):
        t0 = time.time()
        g = pol.infer({**dummy, "bh": {"noise_seed": 3, "mode": "cig", "prompt_old": old_p, "w": w}})["actions"]
        logging.info("CIG check w=%.1f: %.2fs, max |guided - plain| = %.6g", w, time.time() - t0,
                     float(np.abs(np.asarray(g) - np.asarray(plain)).max()))
    if a.selftest:
        return
    server = websocket_policy_server.WebsocketPolicyServer(policy=pol, host="127.0.0.1", port=a.port,
                                                           metadata=pol.metadata)
    logging.info("serving AW-pi0.5 (%s) on port %d", a.kind, a.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
