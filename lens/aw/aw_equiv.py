"""Correctness checks of the attention-wormhole model code against the released openpi pi0.5 policy.
(1) kind=base (no adapters) must reproduce the released policy's actions for the same inputs and flow noise (two real
cameras, prompt padded to 48 instead of 200: masked tokens change nothing); (2) kind=aw / bc at initialisation (LoRA
adapters at their small random init, [LOC] token, wormhole gates at 0) stay close to it; (3) the [LOC] readout of a
prefix pass has the expected shape.
usage: aw_equiv.py STEPS_NPZ [--ckpt DIR]"""
import os
import argparse
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.training import config as _config

import aw_model as AW
import aw_server as AS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps")
    ap.add_argument("--ckpt", default=(os.environ.get("BH", "/data/blackhole") + "/openpi_data/openpi-assets/checkpoints/pi05_libero"))
    a = ap.parse_args()
    z = np.load(a.steps)
    t = int(np.argmax(z["sw"])) if z["sw"].any() else 20
    obs = {"observation/image": z["img"][t], "observation/wrist_image": z["wrist"][t],
           "observation/state": z["state"][t], "prompt": str(z["new_lang"] or z["old_lang"])}
    cfg = _config.get_config("pi05_libero")
    pol = policy_config.create_trained_policy(cfg, a.ckpt)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal((10, 32)).astype(np.float32)
    ref = pol.infer(dict(obs), noise=noise)["actions"]
    print("released policy actions[0]", np.round(ref[0], 4), flush=True)
    res = {}
    import aw_train as AT
    for kind in ("base", "bc", "aw"):
        t0 = time.time()
        if kind == "base":
            p = AS.AWPolicy(kind, "", a.ckpt)
        else:   # untrained adapters: initial LoRA / [LOC] / gates on top of the base weights
            config = AW.aw_config(kind)
            p = AS.AWPolicy(kind, "", a.ckpt, model=AT.load_base(config, a.ckpt, jax.random.key(0)), config=config)
        o = dict(obs)
        inputs = p._in(jax.tree.map(lambda x: x, o))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        ob = _model.Observation.from_dict(inputs)
        x0, info = p._jfn(p._st, ob, jnp.asarray(noise[None]), jnp.float32(1.0), hide_loc=False, donor=None,
                          hide_text=False, img_no_text=False)
        out = p._out(p._unnorm({"state": np.asarray(inputs["state"][0]), "actions": np.asarray(x0[0])}))["actions"]
        d = float(np.abs(out - ref).max())
        res[kind] = d
        print(f"{kind}: max |actions - released| = {d:.5f}  (mean {float(np.abs(out - ref).mean()):.5f}); "
              f"aux {[(k, tuple(np.asarray(v).shape)) for k, v in info.items()]}; {time.time() - t0:.1f}s", flush=True)
    print("RESULT", res)


if __name__ == "__main__":
    main()
