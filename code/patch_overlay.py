"""Patch a copy of openpi so that Gemma attention accepts an additive float mask and can
report attention probabilities of the action-expert queries (used only by the analysis)."""
import pathlib
import sys

p = pathlib.Path(sys.argv[1]) / "openpi" / "models" / "gemma.py"
s = p.read_text()
if "BH_CAPTURE" in s:
    print("already patched")
    sys.exit(0)
old = "        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)\n"
assert old in s, "mask line not found"
s = s.replace(old, (
    "        if jnp.issubdtype(attn_mask.dtype, jnp.floating):\n"
    "            masked_logits = logits + attn_mask[:, :, None, :, :].astype(logits.dtype)\n"
    "        else:\n"
    "            masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)\n"))
old2 = "        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)\n"
assert old2 in s, "softmax line not found"
s = s.replace(old2, old2 + (
    "        if BH_CAPTURE['on'] and kv_cache is not None:\n"
    "            jax.debug.callback(_bh_capture, probs.astype(jnp.float32).mean(axis=(1, 2, 3)))\n"))
hook = ("BH_CAPTURE = {'on': False, 'sink': []}\n\n\n"
        "def _bh_capture(x):\n"
        "    import numpy as _np\n"
        "    BH_CAPTURE['sink'].append(_np.asarray(x))\n\n\n")
anchor = "@at.typecheck\nclass Attention(nn.Module):"
if anchor not in s:
    anchor = "class Attention(nn.Module):"
s = s.replace(anchor, hook + anchor, 1)
n = s.count('mask: at.Bool[at.Array, "b t s"]')
s = s.replace('mask: at.Bool[at.Array, "b t s"]', 'mask: at.Bool[at.Array, "b t s"] | at.Float[at.Array, "b t s"]')
print("relaxed", n, "mask annotations")
p.write_text(s)
print("patched", p)
