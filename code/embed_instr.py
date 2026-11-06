"""Mean PaliGemma token embedding of an instruction (conditioning input of the bridge head)."""
import sys

import jax.numpy as jnp
import numpy as np

from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy_config
from openpi.training import config as _config

cfg_name, ckpt, out, old, new = sys.argv[1:6]
cfg = _config.get_config(cfg_name)
pol = policy_config.create_trained_policy(cfg, ckpt)
tok = _tokenizer.PaligemmaTokenizer(cfg.model.max_token_len)


def emb(text):
    ids = tok._tokenizer.encode(text.strip().lower())
    e = pol._model.PaliGemma.llm(jnp.asarray([ids], jnp.int32), method="embed")
    return np.asarray(e[0], np.float32).mean(0)


np.savez(out, e_old=emb(old), e_new=emb(new))
print("saved", out)
