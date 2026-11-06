"""Add a per-head prefix-attention capture to the patched openpi overlay (off unless BH_CAPTURE['prefix_heads'] is
set while a function is traced).  It records, per layer, the attention of every head from the language positions
(queries q0:) to the base-camera keys (:k1), without averaging: (b, heads, n_text, k1).
usage: patch_overlay_heads.py <overlay dir>"""
import pathlib
import shutil
import sys

p = pathlib.Path(sys.argv[1]) / "openpi" / "models" / "gemma.py"
s = p.read_text()
if "_bh_capture_heads" in s:
    print("already patched")
    sys.exit(0)
shutil.copy(p, p.with_suffix(".py.bak_heads"))
old = "        if BH_CAPTURE.get('prefix', False) and kv_cache is None:\n"
assert old in s, "prefix capture not found (run patch_overlay_prefix.py first)"
s = s.replace(old, (
    "        if BH_CAPTURE.get('prefix_heads', False) and kv_cache is None:\n"
    "            _ph = probs.astype(jnp.float32)[:, :, :, BH_CAPTURE.get('q0', 768):, :BH_CAPTURE.get('k1', 256)]\n"
    "            jax.debug.callback(_bh_capture_heads, _ph.reshape((_ph.shape[0], -1) + _ph.shape[3:]))\n") + old, 1)
hook = "def _bh_capture_prefix(x):\n"
assert hook in s
s = s.replace(hook, ("def _bh_capture_heads(x):\n"
                     "    import numpy as _np\n"
                     "    BH_CAPTURE.setdefault('sink_heads', []).append(_np.asarray(x))\n\n\n") + hook, 1)
p.write_text(s)
print("patched", p)
