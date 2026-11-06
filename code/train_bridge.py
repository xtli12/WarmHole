"""Train the wormhole bridge head v_psi(z, tau | state, d_OT, l_old, l_new) from OT-matched pairs.

Training states come only from held-out rollouts (initial states 20-39). For each state chi on an
old-behavior rollout: Mouth A = stored old continuations retrieved near chi, Mouth B = entry
segments of the new behavior retrieved near chi; entropic OT couples them; every matched pair
(i, j) yields straight-path targets u_ij = y_j - x_i (Eq. bridge), the entry target T(x_i)
(Eq. entry) and the tidal terms (Eq. tidal). The trained weights are exported to .npz for a
numpy forward pass inside the simulator client.

usage: train_bridge.py LIB_OLD.npz LIB_NEW.npz EMB.npz OUT.npz [--steps 20000]
LIB_*.npz: states (n,4), segs (n,H,7), ep (n,), t (n,)  (from build_library.py --with-meta)
EMB.npz  : e_old (d,), e_new (d,) instruction embeddings
"""
import argparse
import math

import numpy as np
import torch
import torch.nn as nn

import wormhole as W


def tau_embed(tau, dim=16):
    freqs = torch.exp(torch.linspace(0, math.log(100.0), dim // 2, device=tau.device))
    a = tau[:, None] * freqs[None] * 2 * math.pi
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class BridgeHead(nn.Module):
    def __init__(self, H=10, A=7, emb_in=2048, emb=32, width=512):
        super().__init__()
        self.H, self.A = H, A
        self.emb_old = nn.Linear(emb_in, emb)
        self.emb_new = nn.Linear(emb_in, emb)
        d_in = H * A * 2 + 16 + 4 + 2 * emb
        self.net = nn.Sequential(nn.Linear(d_in, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, H * A))

    def forward(self, z, tau, s, d, e_old, e_new):
        h = torch.cat([z.flatten(1), tau_embed(tau), s, d.flatten(1), self.emb_old(e_old), self.emb_new(e_new)], -1)
        return self.net(h).view(-1, self.H, self.A)


def sinkhorn_log(C, a, b, eta, iters=300):
    """Log-domain Sinkhorn: the plain-domain version underflows (whole rows of exp(-C/eta) become 0)
    for states whose exits are far from every entry, which produced NaN targets in the first run."""
    def lse(x, axis):
        m = np.max(x, axis=axis, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))
    la, lb = np.log(a), np.log(np.maximum(b, 1e-300))
    f, g = np.zeros(len(a)), np.zeros(len(b))
    for _ in range(iters):
        f = eta * (la - lse((g[None, :] - C) / eta, 1))
        g = eta * (lb - lse((f[:, None] - C) / eta, 0))
    return np.exp((f[:, None] + g[None, :] - C) / eta)


def make_pairs(lib_old, lib_new, rng, n_states=3000, N=16, M=32, H=10):
    """Sample training states on old-behavior rollouts (before the old object is grasped)."""
    S_old, X_old = lib_old["states"], lib_old["segs"][:, :H]
    S_new, X_new = lib_new["states"], lib_new["segs"][:, :H]
    # candidate states: gripper still open (width > 0.06) -> before the grasp
    cand = np.where(S_old[:, 3] > 0.06)[0]
    picks = rng.choice(cand, size=min(n_states, len(cand)), replace=False)
    rows = []
    skipped = 0
    for k in picks:
        p, g = S_old[k, :3], S_old[k, 3]
        d2o = np.sum((S_old[:, :3] - p) ** 2, 1) / 0.05 ** 2 + ((S_old[:, 3] - g) / 0.01) ** 2
        io = np.argsort(d2o)[:N]
        X = X_old[io].astype(np.float64)
        d2n = np.sum((S_new[:, :3] - p) ** 2, 1) / 0.05 ** 2 + ((S_new[:, 3] - g) / 0.01) ** 2
        jn = np.argsort(d2n)[:M]
        wY = np.exp(-0.5 * (d2n[jn] - d2n[jn].min()))
        wY /= wY.sum()
        Y = X_new[jn].astype(np.float64)
        a_prev = lib_old["segs"][k - 1, 0] if k > 0 else np.zeros(7)
        C = W.ot_cost(X, Y, a_prev)
        C = C / (np.median(C) + 1e-9)
        pi = sinkhorn_log(C, np.full(N, 1.0 / N), wY, 0.1)
        rs = pi.sum(1, keepdims=True)
        if not (np.all(np.isfinite(pi)) and np.all(rs > 0)):
            skipped += 1
            continue
        Tx = (pi / rs) @ Y.reshape(M, -1)  # barycentric entry of each exit
        for i in range(N):
            d_i = Tx[i].reshape(H, 7) - X[i]
            for j in np.argsort(-pi[i])[:4]:
                rows.append((X[i], Y[j], pi[i, j] * N, d_i, Tx[i].reshape(H, 7), np.r_[p, g], a_prev))
    print(f"training states: {len(picks) - skipped} used, {skipped} skipped (degenerate plan)", flush=True)
    cols = list(zip(*rows))
    return [np.asarray(c, np.float32) for c in cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lib_old")
    ap.add_argument("lib_new")
    ap.add_argument("emb")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--S", type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    lo, ln, em = np.load(args.lib_old), np.load(args.lib_new), np.load(args.emb)
    X, Y, w, D, T, St, Ap = make_pairs(lo, ln, rng)
    ok = np.all(np.isfinite(np.concatenate([a.reshape(len(a), -1) for a in (X, Y, w[:, None], D, T, St, Ap)], 1)), 1)
    X, Y, w, D, T, St, Ap = (a[ok] for a in (X, Y, w, D, T, St, Ap))
    print("pairs", X.shape[0], f"({(~ok).sum()} non-finite removed)", flush=True)
    s_mu, s_sd = St.mean(0), St.std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = lambda a: torch.as_tensor(a, device=dev)
    X, Y, w, D, T, St, Ap = map(t, (X, Y, w, D, T, (St - s_mu) / s_sd, Ap))
    e_old = t(em["e_old"].astype(np.float32))[None]
    e_new = t(em["e_new"].astype(np.float32))[None]
    model = BridgeHead(emb_in=e_old.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    n = X.shape[0]
    for step in range(args.steps):
        idx = torch.randint(0, n, (256,), device=dev)
        x, y, ww, d, tx, s, ap_ = X[idx], Y[idx], w[idx], D[idx], T[idx], St[idx], Ap[idx]
        eo, en = e_old.expand(len(idx), -1), e_new.expand(len(idx), -1)
        tau = torch.rand(len(idx), device=dev)
        zt = (1 - tau[:, None, None]) * x + tau[:, None, None] * y
        v = model(zt, tau, s, d, eo, en)
        l_bridge = (ww * ((v - (y - x)) ** 2).sum((1, 2))).mean()
        z = x.clone()
        for k in range(args.S):
            tk = torch.full((len(idx),), k / args.S, device=dev)
            z = z + model(z, tk, s, d, eo, en) / args.S
        l_entry = ((z - tx) ** 2).sum((1, 2)).mean()
        l_tidal = ((z[:, 0, :6] - ap_[:, :6]) ** 2).sum(-1).mean() + \
            ((z[:, 2:, :6] - 2 * z[:, 1:-1, :6] + z[:, :-2, :6]) ** 2).sum(-1).mean()
        loss = l_bridge + 0.5 * l_entry + 0.1 * l_tidal
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == args.steps - 1:
            print(f"step {step} bridge {l_bridge.item():.4f} entry {l_entry.item():.4f} tidal {l_tidal.item():.4f}",
                  flush=True)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(args.out, s_mu=s_mu, s_sd=s_sd, e_old=em["e_old"], e_new=em["e_new"], S=args.S,
             **{k.replace(".", "__"): v for k, v in sd.items()})
    print("saved", args.out)


if __name__ == "__main__":
    main()
