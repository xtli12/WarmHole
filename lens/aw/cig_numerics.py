"""Why did the round-f self-check see w = 0 differ from plain sampling, and what does CIG do to the actions?  (one GPU)

c1 (11-01, after the round-f smoke).  Loads one trained policy with aw_server_c9.AWPolicy (exactly the serving code) and
takes real observations from a self-play step record (the frames after the switch of B->A episodes: the images, the
state and both instructions).  Same noise everywhere.  Prints, per frame:
  plain twice (same compiled program)                      -> must be identical (is sampling deterministic?)
  CIG w = 0 vs plain                                       -> differs only if the guided program computes v differently
  CIG with prompt_old = the NEW prompt, w = 0 vs w = 1     -> the two branches identical: v - v_old should vanish
  CIG w in {0.25, 0.5, 1} vs the straight-flow prediction (1 + w) a_new - w a_old (a_old = plain with the old prompt)
  |translation| of plain, old-prompt and guided actions, and the share of action entries outside [-1, 1]
usage: cig_numerics.py KIND PARAMS STEPS_GLOB [N_FRAMES]"""
import os
import glob
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, (os.environ.get("BH", "/data/blackhole") + "/lens/aw"))
import aw_server_c9 as S  # noqa: E402


def main():
    kind, params, sglob = sys.argv[1:4]
    nf = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    ckpt = (os.environ.get("BH", "/data/blackhole") + "/openpi_data/openpi-assets/checkpoints/pi05_libero")
    pol = S.AWPolicy(kind, params, ckpt, str(pathlib.Path(params).parent / "config.json"), bb="pi05")
    frames = []
    for f in sorted(glob.glob(sglob)):
        if "_spBA_" not in f:
            continue
        z = np.load(f)
        sw = np.where(z["sw"] > 0)[0]
        if len(sw) < 40:
            continue
        k = int(sw[0]) + 30
        frames.append(dict(img=z["img"][k], wrist=z["wrist"][k], state=z["state"][k], new=str(z["new_lang"]),
                           old=str(z["old_lang"]), src=f.split("/")[-1]))
        if len(frames) >= nf:
            break
    print(f"{kind}: {len(frames)} frames (B->A, 30 steps after the switch) from {sglob}")

    def act(fr, prompt, **bh):
        obs = {"observation/image": fr["img"], "observation/wrist_image": fr["wrist"], "observation/state": fr["state"],
               "prompt": prompt, "bh": {"noise_seed": 3, **bh}}
        return np.asarray(pol.infer(obs)["actions"], dtype=np.float64)

    mx = lambda a, b: float(np.abs(a - b).max())  # noqa: E731
    t0 = time.time()
    for i, fr in enumerate(frames):
        p1, p2 = act(fr, fr["new"]), act(fr, fr["new"])
        ao = act(fr, fr["old"])
        g = {w: act(fr, fr["new"], mode="cig", prompt_old=fr["old"], w=w) for w in (0.0, 0.25, 0.5, 1.0)}
        s0 = act(fr, fr["new"], mode="cig", prompt_old=fr["new"], w=0.0)
        s1 = act(fr, fr["new"], mode="cig", prompt_old=fr["new"], w=1.0)
        lin = {w: (1 + w) * p1 - w * ao for w in g}
        tr = lambda a: float(np.abs(a[:, :3]).mean())  # noqa: E731
        out = lambda a: float((np.abs(a[:, :6]) > 1.0).mean())  # noqa: E731
        print(f"[{i}] {fr['src'][:60]}")
        print(f"    plain vs plain {mx(p1, p2):.3g} | CIG w=0 vs plain {mx(g[0.0], p1):.3g} | same-prompt CIG w=0 vs w=1"
              f" {mx(s0, s1):.3g}, w=0 vs plain {mx(s0, p1):.3g} | new vs old prompt (plain) {mx(p1, ao):.3g}")
        print("    CIG vs straight-flow prediction (1+w) a_new - w a_old: " +
              ", ".join(f"w={w}: {mx(g[w], lin[w]):.3g} (CIG vs plain {mx(g[w], p1):.3g})" for w in (0.25, 0.5, 1.0)))
        print(f"    |translation| plain {tr(p1):.3f}, old prompt {tr(ao):.3f}, CIG 0.25 {tr(g[0.25]):.3f}, 0.5 {tr(g[0.5]):.3f},"
              f" 1.0 {tr(g[1.0]):.3f} | entries outside [-1, 1]: plain {out(p1):.2f}, CIG 0.5 {out(g[0.5]):.2f}, 1.0 {out(g[1.0]):.2f}"
              f" | gripper plain {p1[:, 6].mean():+.2f} old {ao[:, 6].mean():+.2f} CIG 1.0 {g[1.0][:, 6].mean():+.2f}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
