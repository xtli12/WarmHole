"""Attention lens: numpy-only geometry shared by the policy side and the LIBERO client.

Patch layout of the 224 px policy image (SigLIP): 16 x 16 patches of 14 px, row-major (p = row * 16 + col).
The openpi LIBERO preprocessing flips the raw render in both axes, so the policy image is the upright camera image
mirrored left-right.
"""
import math

import numpy as np

IMG_TOKENS = 256
N_IMG = 2 * IMG_TOKENS            # base + left wrist keys steered by the lens
GRID, PATCH, RES = 16, 14, 224
ACT_SCALE_CM = 1.21               # commanded end-effector displacement per action unit and control step


def patch_centers():
    c = (np.arange(GRID) + 0.5) * PATCH
    cx, cy = np.meshgrid(c, c)
    return np.stack([cx.ravel(), cy.ravel()], axis=1)


def project_to_policy_image(points, cam_pos, cam_rot, fovy_deg, mirror=True):
    """Pixel coordinates (u right, v down) of world points in the 224 px policy image of the agent-view camera."""
    pc = (np.asarray(points, np.float64) - np.asarray(cam_pos, np.float64)) @ np.asarray(cam_rot, np.float64)
    z = -pc[:, 2]
    f = 0.5 * RES / math.tan(math.radians(fovy_deg) / 2)
    u = 0.5 * RES + f * pc[:, 0] / z
    v = 0.5 * RES - f * pc[:, 1] / z
    if mirror:
        u = RES - u
    return np.stack([u, v], axis=1)


def object_map(uv_new, uv_old, sigma=1.2 * PATCH):
    """Oracle map on the base camera: +1 around the new object, -1 around the old object (wrist patches 0)."""
    c = patch_centers()

    def g(uv):
        return np.exp(-((c - np.asarray(uv)) ** 2).sum(axis=1) / (2 * sigma ** 2))

    m = np.zeros(N_IMG, np.float32)
    m[:IMG_TOKENS] = g(uv_new) - g(uv_old)
    return m / (np.abs(m).max() + 1e-12)


def robust_unit(m, q=0.99, eps=1e-12):
    """Scale a map so that its q-quantile of |m| is 1, then clip to [-1, 1]."""
    m = np.asarray(m, np.float32)
    s = np.quantile(np.abs(m), q) + eps
    return np.clip(m / s, -1.0, 1.0)


def displacement_cm(actions):
    """Commanded end-effector displacement of a chunk (cm), from the pose actions."""
    return ACT_SCALE_CM * np.asarray(actions)[..., :, :3].sum(axis=-2)


def policy_depth(depth_raw):
    """Raw MuJoCo agent-view depth buffer (as returned in the observation, upside down) -> the same 180-degree
    rotation the policy image gets ([::-1, ::-1]); still at the render resolution."""
    return np.ascontiguousarray(np.asarray(depth_raw, np.float64).squeeze()[::-1, ::-1])


def real_depth(d, znear, zfar):
    """MuJoCo normalised depth buffer -> metric depth along the optical axis (m); znear, zfar in metres
    (model.vis.map.znear / zfar times model.stat.extent)."""
    return znear / (1.0 - d * (1.0 - znear / zfar))


def unproject_policy_pixels(uv, depth_m, cam_pos, cam_rot, fovy_deg, mirror=True):
    """Inverse of project_to_policy_image: policy-image pixels (n, 2) with metric depths (n,) -> world points."""
    uv = np.atleast_2d(np.asarray(uv, np.float64))
    z = np.asarray(depth_m, np.float64).reshape(-1)
    f = 0.5 * RES / math.tan(math.radians(fovy_deg) / 2)
    u = RES - uv[:, 0] if mirror else uv[:, 0]
    xc = (u - 0.5 * RES) * z / f
    yc = -(uv[:, 1] - 0.5 * RES) * z / f
    pc = np.stack([xc, yc, -z], axis=1)
    return np.asarray(cam_pos, np.float64)[None] + pc @ np.asarray(cam_rot, np.float64).T


def sample_depth(depth_pol, uv):
    """Metric depth at policy-image pixels (n, 2) from a rotated depth image of any square resolution."""
    n = depth_pol.shape[0]
    uv = np.atleast_2d(np.asarray(uv, np.float64))
    c = np.clip((uv[:, 0] * n / RES).astype(int), 0, n - 1)
    r = np.clip((uv[:, 1] * n / RES).astype(int), 0, n - 1)
    return depth_pol[r, c]


def map_target(m, depth_pol, cam_pos, cam_rot, fovy_deg, frac=0.5, sub=4):
    """3-D target of a base-camera focus map m (256,): the pixels of the patches whose value is at least `frac` of
    the maximum are lifted to the world with the depth image; returns the focus-weighted mean of their xy, the
    highest point (object top) and the support of the region.  `sub` x `sub` pixels are sampled per patch."""
    m = np.asarray(m, np.float64)[:IMG_TOKENS]
    if m.max() <= 0:                       # nothing attended more under this instruction: fall back to the peak
        sel = np.array([int(np.argmax(m))])
        m = np.ones_like(m)
    else:
        sel = np.where(m >= frac * m.max())[0]
    offs = (np.arange(sub) + 0.5) * PATCH / sub
    pts, ws = [], []
    for p in sel:
        r0, c0 = (p // GRID) * PATCH, (p % GRID) * PATCH
        uu, vv = np.meshgrid(c0 + offs, r0 + offs)
        uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
        X = unproject_policy_pixels(uv, sample_depth(depth_pol, uv), cam_pos, cam_rot, fovy_deg)
        pts.append(X)
        ws.append(np.full(len(X), m[p] + 1e-12))
    P, W = np.concatenate(pts), np.concatenate(ws)
    xy = (W[:, None] * P[:, :2]).sum(0) / W.sum()
    ztop = float(np.quantile(P[:, 2], 0.9))
    return {"xy": xy, "ztop": ztop, "n_patches": int(len(sel)), "points": P}


def unproject_pixels(uv, depth_m, cam_pos, cam_rot, fovy_deg, res):
    """Upright (not mirrored) pixels (n, 2) of a res x res camera image with metric depths (n,) -> world points."""
    uv = np.atleast_2d(np.asarray(uv, np.float64))
    z = np.asarray(depth_m, np.float64).reshape(-1)
    f = 0.5 * res / math.tan(math.radians(fovy_deg) / 2)
    pc = np.stack([(uv[:, 0] - 0.5 * res) * z / f, -(uv[:, 1] - 0.5 * res) * z / f, -z], axis=1)
    return np.asarray(cam_pos, np.float64)[None] + pc @ np.asarray(cam_rot, np.float64).T


def project_pixels(points, cam_pos, cam_rot, fovy_deg, res):
    """World points -> upright (not mirrored) pixels of a res x res camera image."""
    pc = (np.asarray(points, np.float64) - np.asarray(cam_pos, np.float64)) @ np.asarray(cam_rot, np.float64)
    z = -pc[:, 2]
    f = 0.5 * res / math.tan(math.radians(fovy_deg) / 2)
    return np.stack([0.5 * res + f * pc[:, 0] / z, 0.5 * res - f * pc[:, 1] / z], axis=1)


def box_cloud(bbox, depth_up, cam_pos, cam_rot, fovy_deg, step=1):
    """World points of every `step`-th pixel of a box in an upright res x res image with its metric depth:
    array (nv, nu, 3) laid out like the pixels, and a validity mask (finite, nearer than 10 m)."""
    res = depth_up.shape[0]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    us = np.arange(max(0, int(math.floor(x1))), min(res - 1, int(math.ceil(x2))) + 1, step)
    vs = np.arange(max(0, int(math.floor(y1))), min(res - 1, int(math.ceil(y2))) + 1, step)
    if len(us) < 2 or len(vs) < 2:
        return None, None
    uu, vv = np.meshgrid(us, vs)
    d = depth_up[vv.ravel(), uu.ravel()]
    uv = np.stack([uu.ravel() + 0.5, vv.ravel() + 0.5], 1)          # pixel centres
    P = unproject_pixels(uv, d, cam_pos, cam_rot, fovy_deg, res).reshape(len(vs), len(us), 3)
    ok = np.isfinite(d).reshape(len(vs), len(us)) & (d.reshape(len(vs), len(us)) < 10.0)
    return P, ok


def _label(mask):
    """4-connected components of a boolean image: (labels, n)."""
    try:
        from scipy import ndimage
        return ndimage.label(mask)
    except Exception:  # noqa: BLE001  (no scipy: breadth-first search)
        lab = np.zeros(mask.shape, np.int32)
        n = 0
        H, W = mask.shape
        for r0, c0 in zip(*np.nonzero(mask)):
            if lab[r0, c0]:
                continue
            n += 1
            lab[r0, c0] = n
            stack = [(r0, c0)]
            while stack:
                r, c = stack.pop()
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not lab[rr, cc]:
                        lab[rr, cc] = n
                        stack.append((rr, cc))
        return lab, n


def _core(shape, core):
    nv, nu = shape
    v0, v1 = int(nv * (1 - core) / 2), max(int(nv * (1 - core) / 2) + 1, int(math.ceil(nv * (1 + core) / 2)))
    u0, u1 = int(nu * (1 - core) / 2), max(int(nu * (1 - core) / 2) + 1, int(math.ceil(nu * (1 + core) / 2)))
    m = np.zeros(shape, bool)
    m[v0:v1, u0:u1] = True
    return m


def _pick_component(mask, core_mask):
    """The connected part of mask that covers most of the box core (ties: the larger one)."""
    lab, n = _label(mask)
    if n == 0:
        return None
    best, key = None, None
    for k in range(1, n + 1):
        comp = lab == k
        kk = (int((comp & core_mask).sum()), int(comp.sum()))
        if key is None or kk > key:
            best, key = comp, kk
    return best


def _circle_xy(R):
    """Least-squares circle through points R (n, 2) (Kasa fit): centre, radius."""
    A = np.c_[2 * R, np.ones(len(R))]
    b = (R ** 2).sum(1)
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    r2 = c[2] + c[0] ** 2 + c[1] ** 2
    return c[:2], float(math.sqrt(r2)) if r2 > 0 else float("nan")


def bbox_target(bbox, depth_up, cam_pos, cam_rot, fovy_deg, method="inner", inner=0.5, step=2, core=0.6,
                band=0.015, q_top=0.95, lift=0.01, exclude=None):
    """3-D target of the object in a box of an upright res x res image with its upright metric depth image.
    method  inner: median xy of the points in the inner part of the box, 90 % height quantile there (object top);
            rim:   the object top z_top is the q_top height quantile of the box core; the points within `band`
                   below it that are connected to the core (the rim of a bowl, the top of a box) give the centre as
                   the midpoint of their xy extent (2-98 %);
            rimc:  as rim, centre of a circle fitted to those points (round objects);
            seg:   the object is the part of the box higher than its support (median height of the box border + lift)
                   connected to the core; centre = midpoint of its xy extent;
            rimt:  as rimc, but for a box much larger or taller than an object (a box that holds the support, e.g. the
                   whole cabinet the bowl stands on) the top is taken over the whole box and the largest connected
                   part of the top band is used (the object on top of its support).
    exclude: (point, radius) -- world points within radius of point are ignored (the robot's own gripper).
    All methods return {"xy", "ztop", "center_uv", "n"} (+ "radius" for rimc / rimt)."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if method == "inner":
        res = depth_up.shape[0]
        hw, hh = max(2.0, (x2 - x1) * inner / 2), max(2.0, (y2 - y1) * inner / 2)
        us = np.arange(max(0, cx - hw), min(res - 1, cx + hw) + 1e-6, step)
        vs = np.arange(max(0, cy - hh), min(res - 1, cy + hh) + 1e-6, step)
        uu, vv = np.meshgrid(us, vs)
        uv = np.stack([uu.ravel(), vv.ravel()], 1)
        d = depth_up[np.clip(uv[:, 1].astype(int), 0, res - 1), np.clip(uv[:, 0].astype(int), 0, res - 1)]
        P = unproject_pixels(uv, d, cam_pos, cam_rot, fovy_deg, res)
        return {"xy": np.median(P[:, :2], axis=0), "ztop": float(np.quantile(P[:, 2], 0.9)),
                "center_uv": [cx, cy], "n": int(len(P))}
    P, ok = box_cloud(bbox, depth_up, cam_pos, cam_rot, fovy_deg, step=1)
    if P is None or ok.sum() < 8:
        return bbox_target(bbox, depth_up, cam_pos, cam_rot, fovy_deg, method="inner")
    if exclude is not None:
        ok = ok & (np.linalg.norm(P - np.asarray(exclude[0], np.float64)[None, None], axis=-1) > float(exclude[1]))
        if ok.sum() < 8:
            return bbox_target(bbox, depth_up, cam_pos, cam_rot, fovy_deg, method="inner")
    z = P[..., 2]
    cm = _core(z.shape, core) & ok
    if cm.sum() < 4:
        cm = ok
    big = method == "rimt" and ((y2 - y1) > 1.6 * (x2 - x1) or (x2 - x1) * (y2 - y1) > 0.06 * depth_up.shape[0] ** 2)
    ztop = float(np.quantile(z[ok] if big else z[cm], 0.98 if big else q_top))
    if method in ("rim", "rimc", "rimt"):
        mask = ok & (z >= ztop - band) & (z <= ztop + 0.03)
    elif method == "seg":
        bw = max(2, int(0.06 * min(z.shape)))
        ring = np.zeros(z.shape, bool)
        ring[:bw], ring[-bw:], ring[:, :bw], ring[:, -bw:] = True, True, True, True
        zsup = float(np.median(z[ring & ok])) if (ring & ok).any() else float(np.quantile(z[ok], 0.1))
        mask = ok & (z >= min(zsup + lift, ztop - 0.005)) & (z <= ztop + 0.03)
    else:
        raise ValueError(method)
    comp = _pick_component(mask, np.ones_like(cm) if big else cm)
    if comp is None or comp.sum() < 8:
        return bbox_target(bbox, depth_up, cam_pos, cam_rot, fovy_deg, method="inner")
    R = P[comp][:, :2]
    out = {"ztop": ztop, "center_uv": [cx, cy], "n": int(len(R)), "big": bool(big)}
    lo, hi = np.quantile(R, 0.02, axis=0), np.quantile(R, 0.98, axis=0)
    out["xy"] = (lo + hi) / 2
    if method in ("rimc", "rimt") and len(R) >= 12:
        c, r = _circle_xy(R)
        if np.isfinite(r) and 0.01 < r < 0.2 and np.all(np.abs(c - out["xy"]) < 0.1):
            out["xy"], out["radius"] = c, r
    return out
