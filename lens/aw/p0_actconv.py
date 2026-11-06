"""p0_actconv.py -- CPU only, read-only.  Does the released pi0 LIBERO checkpoint expect delta actions?
openpi trains pi0_libero with an extra DeltaActions transform (actions[:6] -= state[:6], AbsoluteActions at inference);
pi05_libero does not.  aw_train.py / aw_server.py apply only the z-score normalisation for pi0.  Here: the openpi config
flags, and the second moment of the normalised teacher actions per dimension, raw versus delta, over valid chunk steps.
A convention that matches the checkpoint's norm stats gives a second moment near 1 for every dimension."""
import os
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, (os.environ.get("BH", "/data/blackhole") + "/openpi/src"))
from openpi.training import config as C          # noqa: E402
from openpi.shared import normalize as N         # noqa: E402

CK = pathlib.Path((os.environ.get("BH", "/data/blackhole") + "/openpi_data/openpi-assets/checkpoints"))
for name in ("pi0_libero", "pi05_libero"):
    c = C.get_config(name)
    print(name, "| data config:", type(c.data).__name__, "| extra_delta_transform =", getattr(c.data, "extra_delta_transform", "n/a"),
          "| action_horizon =", c.model.action_horizon)

for bb, data, quant in (("pi0", "data_pi0", False), ("pi05", "data_pi05", True)):
    st = N.load(CK / f"{bb}_libero" / "assets" / "physical-intelligence" / "libero")
    sa, ss = st["actions"], st["state"]
    m = np.load(f"{os.environ.get('BH', '/data/blackhole')}/lens/aw/{data}/meta.npz")
    print(f"\n== {bb}: {data} keys {sorted(m.files)}; act {m['act'].shape}, state {m['state'].shape if 'state' in m.files else None}")
    print("  norm stats actions mean", np.round(np.asarray(sa.mean)[:7], 3), "std", np.round(np.asarray(sa.std)[:7], 3))
    print("  norm stats state   mean", np.round(np.asarray(ss.mean)[:8], 3), "std", np.round(np.asarray(ss.std)[:8], 3))
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(int(m["N"]), size=min(20000, int(m["N"])), replace=False))
    act = m["act"][idx].astype(np.float64)                       # (n, H, 7) executed actions
    msk = m["act_mask"][idx].astype(bool)
    stt = m["state"][idx].astype(np.float64)                     # (n, 8)
    dl = act.copy()
    dl[..., :6] -= stt[:, None, :6]

    def nz(x):
        if quant:
            q01, q99 = np.asarray(sa.q01)[:7], np.asarray(sa.q99)[:7]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return (x - np.asarray(sa.mean)[:7]) / (np.asarray(sa.std)[:7] + 1e-6)

    for lab, x in (("raw  ", act), ("delta", dl)):
        z = nz(x)[msk]
        print(f"  {lab}: second moment per dim", np.round((z ** 2).mean(0), 3), "| mean", np.round(z.mean(0), 2),
              "| overall", round(float((z ** 2).mean()), 3))
    print("  raw actions: mean", np.round(act[msk].mean(0), 3), "std", np.round(act[msk].std(0), 3))
    print("  delta actions: mean", np.round(dl[msk].mean(0), 3), "std", np.round(dl[msk].std(0), 3))
print("\nDONE")
