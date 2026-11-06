"""Episode specs for the Wormhole v2 experiments (spatial pair, both directions).

usage: make_specs_w2.py OUT_DIR
Writes
  {bb}_w2smoke.jsonl  a handful of recorded episodes to check the traversal visually
  {bb}_d2.jsonl       development split: initial states 40-49 (never used for the paper's tables, nor for
                      the entry memory, which uses 20-39), early (t_s=2) and committed switches
  {bb}_t2.jsonl       test split: initial states 0-19, every t_s of Table 1, written once the
                      development choices are fixed (variants passed with --test)
"""
import argparse
import json
import os

BH = os.environ.get("BH", "/data/blackhole")
PAIRS = {
    "AB": dict(suite="libero_spatial", scene=7, old=7, pair="A->B",
               new_lang="pick up the black bowl on the wooden cabinet and place it on the plate",
               new_goal=[["on", "akita_black_bowl_2", "plate_1"]]),
    "BA": dict(suite="libero_spatial", scene=9, old=9, pair="B->A",
               new_lang="pick up the black bowl on the stove and place it on the plate",
               new_goal=[["on", "akita_black_bowl_2", "plate_1"]]),
}


MAIN = {}  # settings of the final Wormhole configuration, applied to every wormhole2 variant of the test split


def variants(bb):
    lib2 = lambda d: f"{BH}/lib/{bb}_sp{d}_v2.npz"  # noqa: E731
    lib1 = lambda d: f"{BH}/lib/{bb}_sp{d}.npz"  # noqa: E731
    return {
        "vanilla": lambda d: dict(method="vanilla"),
        "retreat": lambda d: dict(method="retreat"),
        "worm4": lambda d: dict(method="wormhole", worm_chunks=4, library=lib1(d)),
        "w2": lambda d: dict(method="wormhole2", library=lib2(d)),
        "w2m20": lambda d: dict(method="wormhole2", library=lib2(d), worm2_margin=20),
        "w2near": lambda d: dict(method="wormhole2", library=lib2(d), worm2_lam=0.0),
        "w2noot": lambda d: dict(method="wormhole2", library=lib2(d), worm2_ot=False),
        "w2nolift": lambda d: dict(method="wormhole2", library=lib2(d), worm2_lift=0.0, worm2_clear=0.0),
        "w2norate": lambda d: dict(method="wormhole2", library=lib2(d), worm2_rate=100.0),
        "w2nogate": lambda d: dict(method="wormhole2", library=lib2(d), worm2_gate=-1.0),
        "w2allph": lambda d: dict(method="wormhole2", library=lib2(d), worm2_margin=-1000),
        "w2m10": lambda d: dict(method="wormhole2", library=lib2(d), worm2_margin=10),
    }


def episodes(bb, tag, variant, inits, tss, video_inits=()):
    out = []
    V = variants(bb)[variant]
    for d, P in PAIRS.items():
        for ts in tss:
            for i in inits:
                s = dict(P)
                s.update(id=f"{bb}_{tag}_sp{d}_{variant}_ts{ts}_i{i}", init=i, seed=7, ts=ts, tag=tag, variant=variant)
                v = V(d)
                if v.get("method") == "wormhole2" and tag not in ("d2", "w2smoke"):
                    v = {**MAIN, **v}  # variant-specific settings override the main configuration
                s.update(v)
                if i in video_inits:
                    s["video"] = True
                out.append(s)
    return out


GOAL_PAIRS = [(0, 7), (1, 8), (2, 6), (6, 2), (7, 0), (8, 1), (8, 9), (9, 8)]  # as in the LIBERO-Goal sweep


def goal_episodes(bb, tag, variant, inits, tss, video_inits=()):
    """LIBERO-Goal switches (six to another object, two to another destination of the same object)."""
    out = []
    for old, new in GOAL_PAIRS:
        for ts in tss:
            for i in inits:
                s = dict(suite="libero_goal", scene=old, old=old, new=new, pair=f"goal:{old}->{new}",
                         id=f"{bb}_{tag}_goal_{old}to{new}_{variant}_ts{ts}_i{i}", init=i, seed=7, ts=ts, tag=tag,
                         variant=variant)
                if variant == "w2":
                    s.update(method="wormhole2", library=f"{BH}/lib/{bb}_goal_t{new}_v2.npz", **MAIN)
                else:
                    s.update(method=variant)
                if i in video_inits:
                    s["video"] = True
                out.append(s)
    return out


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(path, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--test", default="", help="comma list of variants for the test split")
    ap.add_argument("--goal", default="", help="comma list of variants for the LIBERO-Goal switches")
    ap.add_argument("--ablate", default="", help="comma list of variants for the ablation (pi0.5, inits 0-9)")
    ap.add_argument("--main_margin", type=int, default=10, help="phase margin of the final configuration")
    a = ap.parse_args()
    MAIN["worm2_margin"] = a.main_margin
    os.makedirs(a.out, exist_ok=True)
    for bb in ("pi05", "pi0"):
        smoke = episodes(bb, "w2smoke", "w2", [40, 41], [2, 32, 52], video_inits=(40, 41))
        write(os.path.join(a.out, f"{bb}_w2smoke.jsonl"), smoke)
        dev = []
        names = ["vanilla", "retreat", "worm4", "w2", "w2m20", "w2near"] if bb == "pi05" else ["vanilla", "retreat", "worm4", "w2"]
        for v in names:
            dev += episodes(bb, "d2", v, range(40, 50), [2, 12, 22, 32, 42], video_inits=(40,) if v in ("w2", "retreat") else ())
        write(os.path.join(a.out, f"{bb}_d2.jsonl"), dev)
        if a.test:
            test = []
            for v in a.test.split(","):
                test += episodes(bb, "t2", v, range(20), [2, 12, 22, 32, 42, 52, 62],
                                 video_inits=(0,) if v in ("w2", "retreat") else ())
            write(os.path.join(a.out, f"{bb}_t2.jsonl"), test)
        if a.ablate and bb == "pi05":
            ab = []
            for v in a.ablate.split(","):
                ab += episodes(bb, "a2", v, range(10), [2, 12, 22, 32, 42])
            write(os.path.join(a.out, f"{bb}_a2.jsonl"), ab)
        if a.goal:
            g = []
            for v in a.goal.split(","):
                g += goal_episodes(bb, "g2", v, range(8), [20, 40], video_inits=(0,) if v == "w2" else ())
            write(os.path.join(a.out, f"{bb}_g2.jsonl"), g)


if __name__ == "__main__":
    main()
