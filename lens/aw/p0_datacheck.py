"""p0_datacheck.py -- CPU only.  Training-data path of aw_train.py c10 against c9:
(1) pi0.5 (no delta): every array of the Data object identical between c9 and c10;
(2) pi0: second moment of the normalised actions over valid chunk steps, c9 (absolute) versus c10 (delta, as openpi
    pi0_libero); a convention matching the checkpoint's norm stats gives values near 1.
usage: p0_datacheck.py AW_TRAIN_C9 AW_TRAIN_C10   (run with PYTHONPATH=$A, CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu)"""
import os
import importlib.util
import pathlib
import sys

import numpy as np
from openpi.training import checkpoints as _checkpoints

A = pathlib.Path((os.environ.get("BH", "/data/blackhole") + "/lens/aw"))
CK = pathlib.Path((os.environ.get("BH", "/data/blackhole") + "/openpi_data/openpi-assets/checkpoints"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T9, T10 = load(sys.argv[1], "aw_train_c9"), load(sys.argv[2], "aw_train_c10")
CW, VAL = [0.5, 3.0, 1.0, 0.5], (38, 39)
ok = True
ns = _checkpoints.load_norm_stats(CK / "pi05_libero" / "assets", "physical-intelligence/libero")
d9 = T9.Data(str(A / "data_pi05"), 10, ns, 48, VAL, CW, 14.0, (), quantile=True, geo=None)
d10 = T10.Data(str(A / "data_pi05"), 10, ns, 48, VAL, CW, 14.0, (), quantile=True, geo=None, delta=False)
for k in ("act", "state", "amask", "prompt", "cat", "train_idx", "val_idx", "p", "yb", "vb", "yw", "vw", "coord", "tok", "tmask"):
    same = np.array_equal(getattr(d9, k), getattr(d10, k))
    ok &= same
    print(f"pi0.5 {k:9s} identical c9 / c10: {same}")
del d9, d10
ns0 = _checkpoints.load_norm_stats(CK / "pi0_libero" / "assets", "physical-intelligence/libero")
for lab, T, kw in (("c9 (absolute)", T9, {}), ("c10 (delta)", T10, {"delta": True})):
    d = T.Data(str(A / "data_pi0"), 50, ns0, 48, VAL, CW, 14.0, (), quantile=False, geo=None, **kw)
    z = d.act[..., :7][d.amask]
    m2 = (z ** 2).mean(0)
    print(f"pi0 {lab}: second moment per dim {np.round(m2, 3)}, overall {float(m2.mean()):.3f}; "
          f"padding dims 7-31 all zero: {not d.act[..., 7:].any()}; train {len(d.train_idx)}, val {len(d.val_idx)}")
    if "delta" in kw:
        ok &= bool(m2.mean() < 2.0)
    del d
print("DATACHECK", "PASS" if ok else "FAIL")
