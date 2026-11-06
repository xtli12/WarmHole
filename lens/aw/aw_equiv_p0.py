"""aw_equiv_p0.py -- correctness check of the pi0 path of aw_server.py against the released openpi pi0 LIBERO policy
(the pi0 counterpart of aw_equiv.py, which checked pi0.5 in 121: kind=base max |diff| 0.0046, bc at init 0.0043).
Same observation, same flow noise: (1) openpi's create_trained_policy(get_config("pi0_libero")) -- its output transforms
include AbsoluteActions (extra_delta_transform=True); (2) AWPolicy kind=base with the installed aw_server.py; (3) the
same model output post-processed without AbsoluteActions (= aw_server.py c7, for reference); (4) kind=bc at its
initialisation (small random LoRA) with the installed server transforms.
usage: aw_equiv_p0.py STEPS_NPZ [--ckpt DIR]"""
import os
import argparse
import gc
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.training import config as _config

import aw_model as AW
import aw_server as AS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps")
    ap.add_argument("--ckpt", default=(os.environ.get("BH", "/data/blackhole") + "/openpi_data/openpi-assets/checkpoints/pi0_libero"))
    a = ap.parse_args()
    z = np.load(a.steps)
    t = int(np.argmax(z["sw"])) if z["sw"].any() else 20
    obs = {"observation/image": z["img"][t], "observation/wrist_image": z["wrist"][t],
           "observation/state": z["state"][t], "prompt": str(z["new_lang"] or z["old_lang"])}
    print("observation from", a.steps, "t", t, "state", np.round(z["state"][t], 3), flush=True)
    cfg = _config.get_config("pi0_libero")
    print("pi0_libero extra_delta_transform =", cfg.data.extra_delta_transform, flush=True)
    pol = policy_config.create_trained_policy(cfg, a.ckpt)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal((cfg.model.action_horizon, 32)).astype(np.float32)
    ref = pol.infer(dict(obs), noise=noise)["actions"]
    print("released policy actions", ref.shape, "[0]", np.round(ref[0], 4), flush=True)
    del pol
    gc.collect()
    import aw_train as AT
    res = {}
    for kind in ("base", "bc"):
        t0 = time.time()
        if kind == "base":
            p = AS.AWPolicy(kind, "", a.ckpt, bb="pi0")
        else:   # untrained adapters on top of the base weights
            config = AW.aw_config(kind, "pi0")
            p = AS.AWPolicy(kind, "", a.ckpt, model=AT.load_base(config, a.ckpt, jax.random.key(0)), config=config, bb="pi0")
        inputs = p._in(jax.tree.map(lambda x: x, dict(obs)))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        ob = _model.Observation.from_dict(inputs)
        x0, _ = p._jfn(p._st, ob, jnp.asarray(noise[None]), jnp.float32(1.0), hide_loc=False, donor=None,
                       hide_text=False, img_no_text=False)
        un = p._unnorm({"state": np.asarray(inputs["state"][0]), "actions": np.array(x0[0])})
        out = p._out({k: np.array(v) for k, v in un.items()})["actions"]
        d = float(np.abs(out - ref).max())
        res[kind] = d
        print(f"{kind} (installed server transforms): actions {out.shape}, max |actions - released| = {d:.5f} "
              f"(mean {float(np.abs(out - ref).mean()):.5f}); {time.time() - t0:.1f}s", flush=True)
        if kind == "base":
            c7 = libero_policy.LiberoOutputs()({k: np.array(v) for k, v in un.items()})["actions"]
            d7 = np.abs(c7 - ref).max(axis=0)
            res["base_without_absolute"] = float(d7.max())
            print("base without AbsoluteActions (server c7): max |diff| per dim", np.round(d7, 3), flush=True)
        del p
        gc.collect()
    print("RESULT", res)


if __name__ == "__main__":
    main()
