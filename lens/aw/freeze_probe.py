"""What does a trained policy do at its own B->A freeze point?  (one GPU, read-only; handover 7.13 diagnostic)

c1 (11-01).  Takes the frozen frames of the self-play T1 failures of one model (B->A, never within 6 cm of the new
bowl; the frame 5 steps before the end) and, for comparison, the frames of successful B->A self-play episodes at their
closest pass by the median T1 end pose (within 4 cm).  For every frame (aw_server_c9.AWPolicy = the serving code,
4 policy noises): the mean translation of the first 5 actions (the executed part of a chunk) under the new instruction,
under the old instruction and, for awx, with [LOC] hidden from the action expert (no instruction at all), projected on
the horizontal unit vector toward the new bowl and toward the old bowl (in units of the action, 1 = 5 cm per step);
the share of noises that move toward the new bowl; for awx the [LOC] read-out of object - gripper (dm) against the
truth (error in cm, cosine with the true horizontal direction).
usage: freeze_probe.py KIND PARAMS RUN_GLOB [N]"""
import collections
import glob
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, (os.environ.get("BH", "/data/blackhole") + "/lens/aw"))
import aw_server_c9 as S  # noqa: E402


def load_rows(run_glob):
    rows = {}
    for d in sorted(glob.glob(run_glob)):
        for p in glob.glob(os.path.join(d, "results_*.jsonl")):
            for ln in open(p):
                try:
                    r = json.loads(ln)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("error") or r.get("outcome") is None or r.get("pair") != "B->A" or int(r.get("ts") or 0) < 12:
                    continue
                r["_dir"] = d
                rows[r["id"]] = r
    return rows


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0


def main():
    kind, params, run_glob = sys.argv[1:4]
    nmax = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    rows = load_rows(run_glob)
    t1, ok = [], []
    for r in rows.values():
        f = os.path.join(r["_dir"], "steps", r["id"] + ".npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        sw = np.where(z["sw"] > 0)[0]
        if len(sw) == 0:
            continue
        s = int(sw[0])
        ee, new = z["ee"].astype(float), z["new_obj"].astype(float)
        dxy = np.linalg.norm(ee[s:, :2] - new[s:, :2], axis=1)
        if r["outcome"] != "success" and not (dxy < 0.06).any() and r["outcome"] != "capture":
            t1.append((r, f, len(ee) - 6))
        elif r["outcome"] == "success":
            ok.append((r, f, s))
    t1 = t1[:nmax]
    if not t1:
        print("no T1 failures found")
        return
    endp = np.median([np.load(f)["ee"][k] for _, f, k in t1], axis=0)
    near = []
    for r, f, s in ok:
        ee = np.load(f)["ee"].astype(float)
        d = np.linalg.norm(ee[s:] - endp, axis=1)
        j = int(np.argmin(d))
        if d[j] < 0.04:
            near.append((r, f, s + j))
    near = near[:nmax]
    print(f"{kind} {params}: {len(t1)} frozen frames (T1), {len(near)} frames of successes passing the T1 end pose"
          f" {np.round(endp, 3).tolist()} m within 4 cm")
    ckpt = (os.environ.get("BH", "/data/blackhole") + "/openpi_data/openpi-assets/checkpoints/pi05_libero")
    pol = S.AWPolicy(kind, params, ckpt, str(pathlib.Path(params).parent / "config.json"), bb="pi05")

    def run(fr, prompt, seed, **bh):
        obs = {"observation/image": fr["img"], "observation/wrist_image": fr["wrist"], "observation/state": fr["state"],
               "prompt": prompt, "bh": {"noise_seed": seed, "info": True, **bh}}
        return pol.infer(obs)

    for label, frames in (("frozen (T1)", t1), ("successes at the same place", near)):
        st = collections.defaultdict(list)
        for r, f, k in frames:
            z = np.load(f)
            fr = dict(img=z["img"][k], wrist=z["wrist"][k], state=z["state"][k])
            ee, nb, ob = z["ee"][k].astype(float), z["new_obj"][k].astype(float), z["old_obj"][k].astype(float)
            un, uo = unit(nb[:2] - ee[:2]), unit(ob[:2] - ee[:2])
            conds = [("new", str(z["new_lang"]), {}), ("old", str(z["old_lang"]), {})]
            if kind in ("awx", "aw", "awa"):
                conds.append(("noLOC", str(z["new_lang"]), {"hide_loc": True, "worm_scale": 0.0}))
            for name, prompt, bh in conds:
                v = []
                for seed in range(4):
                    res = run(fr, prompt, seed, **bh)
                    a = np.asarray(res["actions"], np.float64)[:5, :3].mean(0)
                    v.append(a)
                    if name == "new" and seed == 0 and "aw" in res and "coord" in res["aw"]:
                        rel = np.asarray(res["aw"]["coord"], np.float64)[2:5] / 10.0
                        st["readout_err_cm"].append(100 * float(np.linalg.norm(rel - (nb - ee))))
                        st["readout_cos_new"].append(float(unit(rel[:2]) @ un))
                v = np.asarray(v)
                st[name + "_to_new"].append(float(np.mean(v[:, :2] @ un)))
                st[name + "_to_old"].append(float(np.mean(v[:, :2] @ uo)))
                st[name + "_xy"].append(float(np.mean(np.linalg.norm(v[:, :2], axis=1))))
                st[name + "_z"].append(float(np.mean(v[:, 2])))
                st[name + "_share_new"].append(float(np.mean(v[:, :2] @ un > 0)))
                st[name + "_spread"].append(float(np.linalg.norm(v[:, :2].std(0))))
        med = lambda k: float(np.median(st[k])) if st[k] else float("nan")  # noqa: E731
        print(f"-- {label}: {len(frames)} frames")
        for name in ("new", "old", "noLOC"):
            if not st[name + "_xy"]:
                continue
            print(f"   {name:5s}: toward new bowl {med(name + '_to_new'):+.3f}, toward old bowl {med(name + '_to_old'):+.3f},"
                  f" |xy| {med(name + '_xy'):.3f}, z {med(name + '_z'):+.3f}; noises moving toward the new bowl"
                  f" {med(name + '_share_new'):.2f}; spread over noises {med(name + '_spread'):.3f}")
        if st["readout_err_cm"]:
            print(f"   [LOC] read-out of object - gripper: error {med('readout_err_cm'):.1f} cm, cosine with the true"
                  f" horizontal direction to the new bowl {med('readout_cos_new'):+.2f}")


if __name__ == "__main__":
    main()
