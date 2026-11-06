"""Add a prefix-attention capture to the patched openpi overlay (off unless BH_CAPTURE['prefix'] is set while a
function is traced).  It records, per layer, the attention from the language positions (queries 768:) to the base
and wrist image keys (:512), averaged over heads.  usage: patch_overlay_prefix.py <overlay dir>"""
import pathlib
import shutil
import sys

p = pathlib.Path(sys.argv[1]) / "openpi" / "models" / "gemma.py"
s = p.read_text()
if "_bh_capture_prefix" in s:
    print("already patched")
    sys.exit(0)
shutil.copy(p, p.with_suffix(".py.bak_prefix"))
old = "        if BH_CAPTURE['on'] and kv_cache is not None:\n"
assert old in s, "suffix capture line not found (run patch_overlay.py first)"
s = s.replace(old, (
    "        if BH_CAPTURE.get('prefix', False) and kv_cache is None:\n"
    "            jax.debug.callback(_bh_capture_prefix, probs.astype(jnp.float32)"
    "[:, :, :, BH_CAPTURE.get('q0', 768):, :BH_CAPTURE.get('k1', 512)].mean(axis=(1, 2)))\n") + old, 1)
hook = "def _bh_capture(x):\n"
assert hook in s
s = s.replace(hook, ("def _bh_capture_prefix(x):\n"
                     "    import numpy as _np\n"
                     "    BH_CAPTURE.setdefault('sink_prefix', []).append(_np.asarray(x))\n\n\n") + hook, 1)
p.write_text(s)
print("patched", p)
