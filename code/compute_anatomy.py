"""Turn downloaded raw results into the CSVs behind the anatomy figure and tables.

usage: compute_anatomy.py RAW_DIR OUT_DIR BACKBONE [BACKBONE ...]
RAW_DIR/<bb>_e0/{results_*.jsonl,traj/*.npz}, RAW_DIR/<bb>_e1/..., RAW_DIR/<bb>_ana/*.npz
"""
import collections
import csv
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from summarize import escape_time, lift_time, load_results, switch_metrics, wilson  # noqa: E402

SP_TARGET = {"A->B:old": "stove", "B->A:new": "stove", "A->B:new": "cabinet", "B->A:old": "cabinet"}


def boot_ci(vals, n=2000, seed=0):
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_csv(path, rows, fields):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def sweep_rows(bb, rows):
    out = []
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("tag") != "e1" or "outcome" not in r:
            continue
        group = "spatial" if r["suite"] == "libero_spatial" else r["pair"]
        by[(group, r["ts"])].append(r)
        if r["suite"] == "libero_goal":
            by[("goal", r["ts"])].append(r)
    for (group, ts), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        valid = [r for r in rs if r["outcome"] != "done_before_switch"]
        k = sum(r["outcome"] == "success" for r in valid)
        p, lo, hi = wilson(k, len(valid))
        out.append(dict(backbone=bb, pair=group, ts=ts, n=len(valid), success=k,
                        capture=sum(r["outcome"] == "capture" for r in valid),
                        timeout=sum(r["outcome"] == "timeout" for r in valid),
                        done_before=len(rs) - len(valid), p_esc=round(p, 4), lo=round(lo, 4), hi=round(hi, 4)))
    return out


def lift_rows(bb, rows, raw):
    out = []
    for r in rows:
        if r.get("tag") != "e0" or r.get("outcome") != "success" or r["suite"] != "libero_spatial":
            continue
        z = np.load(os.path.join(raw, "traj", r["id"] + ".npz"))
        lt = lift_time(z, "old_obj")
        if lt is not None:
            out.append(dict(backbone=bb, pair="spatial", id=r["id"], lift_t=lt))
    return out


def library(rows, raw, pair_keys, H=5):
    states, segs = [], []
    for r in rows:
        if r.get("pair") not in pair_keys or r.get("outcome") != "success":
            continue
        z = np.load(os.path.join(raw, "traj", r["id"] + ".npz"))
        ee, gr, act = z["ee"], z["grip"], z["act"]
        for t in range(len(act) - H):
            states.append(np.r_[ee[t], gr[t, 0] - gr[t, 1]])
            segs.append(act[t:t + H, :6].reshape(-1))
    return np.asarray(states), np.asarray(segs)


def transmission_rows(bb, rows, raw_e0, ana_dir, raw_lib=None, H=5, sig_p=0.05, sig_g=0.01, knn=20):
    """rho_theta = <Delta, d*> / |d*|^2 on spatial no-switch rollouts (both directions, both scenes)."""
    lib_rows = rows
    out = []
    per_t = collections.defaultdict(lambda: collections.defaultdict(list))
    # entry libraries: executions of the *new* instruction in the same layout
    libs = {
        "stove": library(lib_rows, raw_e0, {"A->B:old", "B->A:new"}, H),
        "cabinet": library(lib_rows, raw_e0, {"A->B:new", "B->A:old"}, H),
    }
    for r in rows:
        if r.get("tag") != "e0" or r["suite"] != "libero_spatial" or r.get("outcome") != "success":
            continue
        f = os.path.join(ana_dir, r["id"] + ".npz")
        if not os.path.exists(f):
            continue
        a = np.load(f, allow_pickle=True)
        z = np.load(os.path.join(raw_e0, "traj", r["id"] + ".npz"))
        tgt_old = SP_TARGET[r["pair"]]
        tgt_new = "cabinet" if tgt_old == "stove" else "stove"
        st_lib, seg_lib = libs[tgt_new]
        acts = a["acts"]  # (K, P, S, Hc, 7)
        t_q = a["t"]
        alpha = a["alpha_instr"][:, 0]
        a0 = alpha[0] if alpha[0] > 0 else np.nan
        d0 = None
        for k, t in enumerate(t_q):
            e_old = acts[k, 0, :, :H, :6].mean(axis=0).reshape(-1)
            e_new = acts[k, 1, :, :H, :6].mean(axis=0).reshape(-1)
            delta = e_new - e_old
            p = z["ee"][t]
            g = z["grip"][t, 0] - z["grip"][t, 1]
            d2 = np.sum((st_lib[:, :3] - p) ** 2, axis=1) / sig_p ** 2 + ((st_lib[:, 3] - g) / sig_g) ** 2
            idx = np.argsort(d2)[:knn]
            w = np.exp(-0.5 * (d2[idx] - d2[idx].min()))
            e_pnew = (w[:, None] * seg_lib[idx]).sum(0) / w.sum()
            dstar = e_pnew - e_old
            rho = float(delta @ dstar / (dstar @ dstar + 1e-9))
            sens = float(np.linalg.norm(delta))
            if d0 is None:
                d0 = sens if sens > 0 else np.nan
            tb = int(t // 10 * 10)
            per_t[tb]["rho"].append(rho)
            per_t[tb]["sens"].append(sens / d0)
            per_t[tb]["alpha"].append(alpha[k] / a0)
    for tb in sorted(per_t):
        d = per_t[tb]
        if len(d["rho"]) < 8:
            continue
        rm, rl, rh = boot_ci(d["rho"])
        sm, sl, sh = boot_ci(d["sens"])
        am, al, ah = boot_ci(d["alpha"])
        out.append(dict(backbone=bb, pair="spatial", t=tb, n=len(d["rho"]), rho_mean=round(rm, 4), rho_lo=round(rl, 4),
                        rho_hi=round(rh, 4), sens_mean=round(sm, 4), sens_lo=round(sl, 4), sens_hi=round(sh, 4),
                        alpha_mean=round(am, 4), alpha_lo=round(al, 4), alpha_hi=round(ah, 4)))
    return out


def ridge_probe(Xtr, ytr, Xte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    classes = np.unique(ytr)
    Y = (ytr[:, None] == classes[None]).astype(np.float64) * 2 - 1
    A = Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1])
    Wt = np.linalg.solve(A, Xtr.T @ Y)
    return classes[np.argmax(Xte @ Wt, axis=1)]


def probe_rows(bb, rows, ana_dir, reps=20, seed=0, setting="spatial"):
    eps = []
    suite = "libero_spatial" if setting == "spatial" else "libero_goal"
    for r in rows:
        if r.get("tag") != "e0" or r["suite"] != suite or r.get("outcome") != "success":
            continue
        f = os.path.join(ana_dir, r["id"] + ".npz")
        if os.path.exists(f):
            a = np.load(f, allow_pickle=True)
            lab = SP_TARGET[r["pair"]] if setting == "spatial" else r["pair"]
            eps.append((lab, a["t"], a["feats"].astype(np.float32)))
    if len(eps) < 8:
        return []
    labels = np.array([e[0] for e in eps])
    rng = np.random.default_rng(seed)
    tbins = sorted({int(t // 10 * 10) for _, ts, _ in eps for t in ts})
    out = []
    for tb in tbins:
        X, y, g = [], [], []
        for gi, (lab, ts, F) in enumerate(eps):
            ks = [k for k, t in enumerate(ts) if int(t // 10 * 10) == tb]
            if ks:
                X.append(F[ks[0]])
                y.append(lab)
                g.append(gi)
        X, y = np.asarray(X), np.asarray(y)
        if len(np.unique(y)) < 2 or min(collections.Counter(y).values()) < 3:
            continue
        accs = []
        for _ in range(reps):
            perm = rng.permutation(len(X))
            nte = max(2, len(X) // 4)
            te, tr = perm[:nte], perm[nte:]
            if len(np.unique(y[tr])) < 2:
                continue
            accs.append(float(np.mean(ridge_probe(X[tr], y[tr], X[te]) == y[te])))
        m, lo, hi = boot_ci(accs)
        out.append(dict(backbone=bb, setting=setting, t=tb, n=len(X), acc_mean=round(m, 4), acc_lo=round(lo, 4),
                        acc_hi=round(hi, 4), chance=round(max(collections.Counter(y).values()) / len(y), 4)))
    return out


def main(raw, out_dir, backbones):
    os.makedirs(out_dir, exist_ok=True)
    for f in ("sweep.csv", "lift.csv", "transmission.csv", "probe.csv"):
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            os.remove(p)
    for bb in backbones:
        e0 = os.path.join(raw, f"{bb}_e0")
        e1 = os.path.join(raw, f"{bb}_e1")
        ana = os.path.join(raw, f"{bb}_ana")
        r0 = load_results(e0) if os.path.isdir(e0) else []
        r1 = load_results(e1) if os.path.isdir(e1) else []
        if r1:
            write_csv(os.path.join(out_dir, "sweep.csv"), sweep_rows(bb, r1),
                      ["backbone", "pair", "ts", "n", "success", "capture", "timeout", "done_before", "p_esc", "lo", "hi"])
        if r0:
            write_csv(os.path.join(out_dir, "lift.csv"), lift_rows(bb, r0, e0), ["backbone", "pair", "id", "lift_t"])
        if r0 and os.path.isdir(ana):
            write_csv(os.path.join(out_dir, "transmission.csv"), transmission_rows(bb, r0, e0, ana),
                      ["backbone", "pair", "t", "n", "rho_mean", "rho_lo", "rho_hi", "sens_mean", "sens_lo", "sens_hi",
                       "alpha_mean", "alpha_lo", "alpha_hi"])
            for setting in ("spatial", "goal"):
                write_csv(os.path.join(out_dir, "probe.csv"), probe_rows(bb, r0, ana, setting=setting),
                          ["backbone", "setting", "t", "n", "acc_mean", "acc_lo", "acc_hi", "chance"])
    print("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
