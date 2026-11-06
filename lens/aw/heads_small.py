"""Compact copy of a head-analysis result for plotting: the maps of the top-k heads (and, for a trained model, the
mean map of its designated [LOC] heads), the object positions, the frame ids, and the frames themselves.
usage: heads_small.py RESULT_BASE OUT.npz --frames 'GLOB' [--k 5] [--designated "6-13:0-3"]"""
import argparse
import glob
import json
import os

import numpy as np


def rng(s):
    lo, hi = (int(x) for x in s.split("-"))
    return list(range(lo, hi + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--designated", default="")
    a = ap.parse_args()
    z = np.load(a.base + ".npz")
    js = json.load(open(a.base + ".json"))
    M = z["maps"]                                                  # (F, 2, L, H, 256) float16
    top = [(h["layer"], h["head"]) for h in js["top_heads"][: a.k]]
    out = {"top": np.stack([M[:, :, l, h] for l, h in top], axis=2), "top_heads": np.array(top),
           "uv": z["uv"], "ids": z["ids"]}
    if a.designated:
        ls, hs = a.designated.split(":")
        sel = M[:, :, rng(ls)][:, :, :, rng(hs)].astype(np.float32)
        out["designated"] = sel.mean(axis=(2, 3)).astype(np.float16)
    files = {os.path.basename(p)[:-4]: p for p in glob.glob(a.frames)}
    imgs = []
    for fid in z["ids"]:
        s = np.load(files[str(fid)], allow_pickle=True)
        if "rgb" in s.files:                       # save_switch frame (native render at the switch)
            imgs.append(s["rgb"])
        else:                                      # step record: the policy view at the switch
            imgs.append(s["img"][int(np.argmax(s["sw"]))])
    out["img"] = np.stack(imgs)
    np.savez_compressed(a.out, **out)
    print(a.out, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
