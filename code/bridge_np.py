"""Numpy forward pass of the trained bridge head (used inside the simulator client)."""
import math

import numpy as np


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _tau_embed(tau, dim=16):
    freqs = np.exp(np.linspace(0, math.log(100.0), dim // 2))
    a = tau * freqs * 2 * math.pi
    return np.concatenate([np.sin(a), np.cos(a)])


class BridgeNP:
    def __init__(self, path):
        z = np.load(path)
        self.W = {k.replace("__", "."): z[k].astype(np.float64) for k in z.files if "__" in k}
        self.s_mu, self.s_sd = z["s_mu"].astype(np.float64), z["s_sd"].astype(np.float64)
        self.S = int(z["S"])
        self.eo = self._lin("emb_old", z["e_old"].astype(np.float64))
        self.en = self._lin("emb_new", z["e_new"].astype(np.float64))

    def _lin(self, name, x):
        return x @ self.W[name + ".weight"].T + self.W[name + ".bias"]

    def velocity(self, z, tau, s, d):
        h = np.concatenate([z.ravel(), _tau_embed(tau), (s - self.s_mu) / self.s_sd, d.ravel(), self.eo, self.en])
        h = _silu(self._lin("net.0", h))
        h = _silu(self._lin("net.2", h))
        h = _silu(self._lin("net.4", h))
        return self._lin("net.6", h).reshape(z.shape)

    def traverse(self, x0, s, d):
        z = np.array(x0, dtype=np.float64)
        for k in range(self.S):
            z = z + self.velocity(z, k / self.S, s, d) / self.S
        return z
