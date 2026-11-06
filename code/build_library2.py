"""Entry memory for the state-space throat (Mouth B of Wormhole, v2).

Stores every state of the successful no-switch executions of the new instruction (held-out library
rollouts, initial states 20-39) together with what the traversal needs to choose a mouth:
  pos (n,3)   end-effector position
  aa  (n,3)   end-effector orientation (axis-angle, as in the policy state)
  grip (n,)   gripper opening
  t, t_close, T, ep   step index, end of the approach phase (first step at which the gripper is commanded
                      to close; for tasks without a grasp, first step at which the task's key object moves),
                      step at which the task was completed, episode index
  segs (n,H,7)        the actions executed from that state (used by the v1 axis and for logging)

usage: build_library2.py RUN_DIR PAIR_KEY OUT.npz [H] [TAG]
"""
import glob
import json
import sys

import numpy as np

run_dir, pair_key, out = sys.argv[1], sys.argv[2], sys.argv[3]
H = int(sys.argv[4]) if len(sys.argv) > 4 else 10
tag = sys.argv[5] if len(sys.argv) > 5 else "lib"
rows = [json.loads(l) for p in glob.glob(f"{run_dir}/results_*.jsonl") for l in open(p) if l.strip()]
cols = {k: [] for k in ("pos", "aa", "grip", "t", "t_close", "T", "ep", "segs")}
n_ep = 0
for r in sorted(rows, key=lambda r: r["id"]):
    if r.get("pair") != pair_key or r.get("outcome") != "success" or r.get("tag") != tag:
        continue
    z = np.load(f"{run_dir}/traj/{r['id']}.npz")
    ee, aa, gr, act = z["ee"], z["aa"], z["grip"], z["act"]
    closing = np.where(act[:, 6] > 0)[0]
    if len(closing):
        t_close = int(closing[0])
    else:  # no grasp (e.g. opening a drawer): the approach ends when the task's key object first moves
        o = z["old_obj"] if "old_obj" in z.files else np.full((len(act), 3), np.nan)
        moved = np.where(np.linalg.norm(o - o[0], axis=1) > 0.01)[0] if not np.isnan(o).any() else []
        t_close = int(moved[0]) if len(moved) else len(act)
    T = int(r.get("t_old") or len(act))
    for t in range(len(act)):
        seg = act[t:t + H]
        if len(seg) < H:
            seg = np.concatenate([seg, np.repeat(seg[-1:], H - len(seg), axis=0)])
        cols["pos"].append(ee[t])
        cols["aa"].append(aa[t])
        cols["grip"].append(gr[t, 0] - gr[t, 1])
        cols["t"].append(t)
        cols["t_close"].append(t_close)
        cols["T"].append(T)
        cols["ep"].append(n_ep)
        cols["segs"].append(seg)
    n_ep += 1
arr = {k: np.asarray(v, np.float32 if k in ("pos", "aa", "grip", "segs") else np.int32) for k, v in cols.items()}
# v1-compatible fields, so the same file also works with EntryLibrary
arr["states"] = np.concatenate([arr["pos"], arr["grip"][:, None]], axis=1).astype(np.float32)
np.savez(out, **arr)
tc = np.unique(np.stack([arr["ep"], arr["t_close"]], 1), axis=0)[:, 1]
print(f"{out}: {n_ep} episodes, {len(arr['t'])} states, gripper closes at step {np.median(tc):.0f} (median)")
