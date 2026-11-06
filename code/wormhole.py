"""Training-free OT wormhole traversal (numpy only; used by the LIBERO client).

Mouth A  : N continuation chunks of the base policy under the old instruction (exits).
Mouth B  : M entry segments retrieved from successful executions of the new instruction whose
           start states are compatible with the current state, with compatibility weights.
Throat   : entropic OT plan between exits and entries under a manipulation-aware cost
           (end-effector path, gripper, junction velocity, entry smoothness).
Axis     : anchored barycentric displacement d_OT = sum_ij pi_ij a_i (y_j - x_i) / sum_ij pi_ij a_i.
Traversal: bridge chunk x_hat + s(k) d_OT with a short ramp s(k) (tidal regularization).
"""
import numpy as np


class EntryLibrary:
    def __init__(self, path, horizon):
        z = np.load(path)
        self.states = z["states"].astype(np.float64)      # (n, 4): ee xyz, gripper width
        self.segs = z["segs"][:, :horizon].astype(np.float64)  # (n, H, 7)

    def retrieve(self, p, g, m=32, sigma_p=0.05, sigma_g=0.01):
        d2 = np.sum((self.states[:, :3] - p) ** 2, axis=1) / sigma_p ** 2 \
            + ((self.states[:, 3] - g) / sigma_g) ** 2
        idx = np.argsort(d2)[:m]
        w = np.exp(-0.5 * (d2[idx] - d2[idx].min()))
        return self.segs[idx], w / w.sum(), idx


def sinkhorn(C, a, b, eta, iters=300):
    K = np.exp(-(C - C.min()) / eta)
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(iters):
        u = a / (K @ v + 1e-300)
        v = b / (K.T @ u + 1e-300)
    return u[:, None] * K * v[None, :]


def _path(x):
    return np.cumsum(x[..., :3], axis=-2)


def ot_cost(X, Y, a_prev, lam=(1.0, 0.5, 0.5, 0.2)):
    n = X.shape[0]
    PX, PY = _path(X), _path(Y)
    c_p = ((PX[:, None] - PY[None]) ** 2).sum(-1).mean(-1)
    c_g = ((X[:, None, :, 6] - Y[None, :, :, 6]) ** 2).mean(-1)
    c_v = np.repeat(((Y[:, 0, :6] - a_prev[None, :6]) ** 2).sum(-1)[None], n, axis=0)
    c_s = np.repeat((np.diff(Y[..., :6], 2, axis=1) ** 2).sum(-1).mean(-1)[None], n, axis=0)
    return sum(l * t / (np.median(t) + 1e-9) for l, t in zip(lam, (c_p, c_g, c_v, c_s)))


def ot_axis(X, Y, wY, a_prev, eta=0.1, anchor=True, use_ot=True):
    """Throat + anchored axis. Returns the anchor x_hat, the axis d_OT and the plan."""
    n, m = X.shape[0], Y.shape[0]
    x_hat = X.mean(axis=0)
    if use_ot:
        C = ot_cost(X, Y, a_prev)
        C = C / (np.median(C) + 1e-9)
        pi = sinkhorn(C, np.full(n, 1.0 / n), wY, eta)
    else:  # no throat: independent coupling (difference of marginal means)
        pi = np.full((n, 1), 1.0 / n) * wY[None, :]
    if anchor:
        da = ((X - x_hat) ** 2).sum(axis=(-1, -2))
        alpha = np.exp(-da / (2 * (np.median(da) + 1e-9)))
    else:
        alpha = np.ones(n)
    Wt = pi * alpha[:, None]
    Wt = Wt / Wt.sum()
    d_ot = np.einsum("ij,ijhk->hk", Wt, Y[None] - X[:, None])
    return x_hat, d_ot, pi


def bridge_chunk(X, Y, wY, a_prev, eta=0.1, ramp=3, anchor=True, use_ot=True, head=None, state=None):
    """X: (N,H,7) exits, Y: (M,H,7) entries, wY: (M,) entry weights, a_prev: last executed action.
    Closed-form traversal x_hat + s(k) d_OT, or the learned bridge head integrated from x_hat."""
    x_hat, d_ot, pi = ot_axis(X, Y, wY, a_prev, eta, anchor, use_ot)
    if head is not None:
        br = head.traverse(x_hat, state, d_ot)
    else:
        s = np.minimum(1.0, (np.arange(X.shape[1]) + 1.0) / max(1, ramp))[:, None]
        br = x_hat + s * d_ot
    br = np.array(br)
    br[:, 6] = np.where(br[:, 6] >= 0, 1.0, -1.0)
    return br.astype(np.float32), d_ot.astype(np.float32), pi


# ---------------------------------------------------------------------------------------------------
# v2: state-space throat. The v1 axis transfers how the entries move but leaves the robot where it is,
# so the hand-over state is usually far from every execution of the new instruction (median 12-21 cm
# for the failures of v1). v2 chooses an entry *state* on the far side and carries the robot there.
# ---------------------------------------------------------------------------------------------------
class EntryLibrary2:
    """Mouth B for the state-space throat (see build_library2.py)."""

    def __init__(self, path, horizon=10):
        z = np.load(path)
        self.pos = z["pos"].astype(np.float64)
        self.aa = z["aa"].astype(np.float64)
        self.grip = z["grip"].astype(np.float64)
        self.t = z["t"].astype(np.int64)
        self.t_close = z["t_close"].astype(np.int64)
        self.T = z["T"].astype(np.int64)
        self.ep = z["ep"].astype(np.int64)
        self.segs = z["segs"][:, :horizon].astype(np.float64)

    def approach(self, margin):
        """States at least `margin` steps before the gripper first closes (nothing grasped yet)."""
        return np.where(self.t <= self.t_close - margin)[0]

    def manifold_dist(self, p):
        """Distance of an end-effector position to the approach phase of the new task."""
        idx = self.approach(1)
        return float(np.sqrt(((self.pos[idx] - p) ** 2).sum(axis=1)).min())


def lift_travel(P, E, clear):
    """Length of the lift-transit-settle path from each point of P (n,3) to each point of E (m,3)."""
    ztop = np.maximum(P[:, None, 2], E[None, :, 2]) + clear
    return (ztop - P[:, None, 2]) + np.linalg.norm(P[:, None, :2] - E[None, :, :2], axis=-1) + (ztop - E[None, :, 2])


def sinkhorn_log(C, a, b, eta, iters=300):
    """Entropic OT in the log domain (no underflow for small eta or tiny marginals)."""
    la, lb = np.log(a + 1e-300), np.log(b + 1e-300)
    f, g = np.zeros_like(a), np.zeros_like(b)
    M = -C / eta
    for _ in range(iters):
        f = la - _lse(M + g[None, :], axis=1)
        g = lb - _lse(M + f[:, None], axis=0)
    return np.exp(M + f[:, None] + g[None, :])


def _lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def aa2quat(aa):
    """Axis-angle (n,3) -> quaternion (n,4) in (x, y, z, w) order, as robosuite uses."""
    aa = np.atleast_2d(aa)
    ang = np.linalg.norm(aa, axis=1, keepdims=True)
    ax = np.where(ang > 1e-9, aa / np.maximum(ang, 1e-12), np.array([[1.0, 0.0, 0.0]]))
    return np.concatenate([ax * np.sin(ang / 2), np.cos(ang / 2)], axis=1)


def mean_quat(q, w):
    q = q * np.sign(q @ q[np.argmax(w)])[:, None]  # same hemisphere as the heaviest one
    m = (w[:, None] * q).sum(axis=0)
    return m / np.linalg.norm(m)


def select_mouth(lib, p, X=None, margin=10, v=0.015, lam=1.0, kappa=5.0, clear=0.03, m=64, eta=0.1,
                 scale=0.0121, use_ot=True):
    """Mouth B weights and the target of the traversal.

    Each approach state s_j of the new task is scored by the expected number of steps to finish the
    new task through it: the length of the lift-transit-settle path from the robot to s_j at speed v,
    plus lam times its time-to-go T_j - t_j. b_j ~ exp(-score_j / kappa) over the m best states.
    With exits X (N,H,7) the plan of Eq. (ot) couples the end points of the exits with the mouths under
    the same score, and the target is the anchored transport-weighted entry state (Eq. (axis) in state
    space); without X it is the b-weighted mean. Returns a dict with target position and orientation."""
    idx = lib.approach(margin)
    E = lib.pos[idx]
    rem = (lib.T[idx] - lib.t[idx]).astype(np.float64)
    c0 = lift_travel(p[None], E, clear)[0] / v + lam * rem
    pre = np.argsort(c0)[:m]
    E, rem, c0, sel = E[pre], rem[pre], c0[pre], idx[pre]
    b = np.exp(-(c0 - c0.min()) / kappa)
    b = b / b.sum()
    plan = None
    if use_ot and X is not None:
        n = X.shape[0]
        Q = p[None] + scale * X[:, :, :3].sum(axis=1)  # where each exit would bring the gripper
        C = lift_travel(Q, E, clear) / v + lam * rem[None]
        C = C / (np.median(C) + 1e-9)
        plan = sinkhorn_log(C, np.full(n, 1.0 / n), b, eta)
        x_hat = X.mean(axis=0)
        da = ((X - x_hat) ** 2).sum(axis=(-1, -2))
        alpha = np.exp(-da / (2 * (np.median(da) + 1e-9)))
        w = (plan * alpha[:, None]).sum(axis=0)
        w = w / w.sum()
    else:
        w = b
    pos = (w[:, None] * E).sum(axis=0)
    quat = mean_quat(aa2quat(lib.aa[sel]), w)
    return {"pos": pos, "quat": quat, "grip": float((w * lib.grip[sel]).sum()), "w": w, "idx": sel, "plan": plan,
            "t": float((w * lib.t[sel]).sum()), "rem": float((w * rem).sum()), "score": float((w * c0).sum()),
            "travel": float((w * lift_travel(p[None], E, clear)[0]).sum())}
