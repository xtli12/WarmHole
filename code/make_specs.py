"""Generate episode specs (JSONL) for the black-hole experiments."""
import os
import json
import sys

SPATIAL_LANG = {
    7: "pick up the black bowl on the stove and place it on the plate",
    9: "pick up the black bowl on the wooden cabinet and place it on the plate",
}
# In the scene of task 7 the bowl on the wooden cabinet is akita_black_bowl_2; in the scene of task 9
# the bowl on the stove is akita_black_bowl_2. Tasks 7 and 9 share the same layout.
BOWL2_ON_PLATE = [["on", "akita_black_bowl_2", "plate_1"]]


def spatial_pair(direction):
    """direction 'AB': scene 7, stove bowl -> cabinet bowl; 'BA': scene 9, cabinet bowl -> stove bowl."""
    if direction == "AB":
        return dict(suite="libero_spatial", scene=7, old=7, new_lang=SPATIAL_LANG[9], new_goal=BOWL2_ON_PLATE, pair="A->B")
    return dict(suite="libero_spatial", scene=9, old=9, new_lang=SPATIAL_LANG[7], new_goal=BOWL2_ON_PLATE, pair="B->A")


def spatial_noswitch(direction, which):
    base = spatial_pair(direction)
    if which == "old":
        return dict(suite=base["suite"], scene=base["scene"], old=base["old"], pair=base["pair"] + ":old")
    return dict(suite=base["suite"], scene=base["scene"], old_lang=base["new_lang"], old_goal=base["new_goal"],
                pair=base["pair"] + ":new")


GOAL_PAIRS = [(8, 9), (9, 8), (6, 2), (2, 6), (7, 0), (0, 7), (1, 8), (8, 1)]


def write(path, specs):
    with open(path, "w") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")
    print(path, len(specs))


def main(backbone, out_dir):
    tag = backbone
    # E0: no-switch references (recorded observations for the offline analyses)
    e0 = []
    for d in ("AB", "BA"):
        for which in ("old", "new"):
            for i in range(20):
                s = spatial_noswitch(d, which)
                s.update(id=f"{tag}_e0_sp{d}_{which}_i{i}", init=i, seed=7, method="vanilla", record=True,
                         video=(i < 2), tag="e0")
                e0.append(s)
    for tid in range(10):
        for i in range(10):
            e0.append(dict(id=f"{tag}_e0_goal_t{tid}_i{i}", suite="libero_goal", scene=tid, old=tid, init=i, seed=7,
                           method="vanilla", record=True, video=(i < 1), tag="e0", pair=f"goal:{tid}"))
    write(f"{out_dir}/{tag}_e0.jsonl", e0)

    # E0-lib: no-switch rollouts of the *new* instruction from held-out initial states (20-39);
    # they form the entry library of Mouth B and are never used for evaluation
    lib = []
    for d in ("AB", "BA"):
        for i in range(20, 40):
            s = spatial_noswitch(d, "new")
            s.update(id=f"{tag}_lib_sp{d}_i{i}", init=i, seed=7, method="vanilla", tag="lib")
            lib.append(s)
    for tid in range(10):
        for i in range(20, 30):
            lib.append(dict(id=f"{tag}_lib_goal_t{tid}_i{i}", suite="libero_goal", scene=tid, old=tid, init=i, seed=7,
                            method="vanilla", tag="lib", pair=f"goal:{tid}"))
    write(f"{out_dir}/{tag}_lib.jsonl", lib)

    # E1: switch-step sweep, vanilla
    e1 = []
    for d in ("AB", "BA"):
        for ts in range(0, 85, 5):
            for i in range(20):
                s = spatial_pair(d)
                s.update(id=f"{tag}_e1_sp{d}_ts{ts}_i{i}", init=i, seed=7, ts=ts, method="vanilla", tag="e1",
                         video=(i == 0 and ts % 20 == 0))
                e1.append(s)
    for (a, b) in GOAL_PAIRS:
        for ts in range(0, 130, 10):
            for i in range(10):
                e1.append(dict(id=f"{tag}_e1_goal_{a}to{b}_ts{ts}_i{i}", suite="libero_goal", scene=a, old=a, new=b,
                               init=i, seed=7, ts=ts, method="vanilla", tag="e1", pair=f"goal:{a}->{b}",
                               video=(i == 0 and ts % 40 == 0)))
    # LIBERO-Object probe from the introduction: salad dressing <-> alphabet soup (same scene)
    for (a, b) in [(2, 0), (0, 2)]:
        for ts in (0, 4, 8, 12, 16, 20, 25, 30, 40, 50, 60):
            for i in range(20):
                e1.append(dict(id=f"{tag}_e1_obj_{a}to{b}_ts{ts}_i{i}", suite="libero_object", scene=a, old=a, new=b,
                               init=i, seed=7, ts=ts, method="vanilla", tag="e1", pair=f"object:{a}->{b}",
                               video=(i == 0 and ts in (8, 16))))
    write(f"{out_dir}/{tag}_e1.jsonl", e1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])


def remedies(backbone, out_dir, ts_list, inits=range(20), methods=None, lib_dir=(os.environ.get("BH", "/data/blackhole") + "/lib")):
    """E2: training-free remedies and the wormhole on the spatial pair (both directions)."""
    tag = backbone
    methods = methods or [
        ("vanilla", {}), ("flush", {}), ("cig1", {"method": "cig", "w": 1.0}), ("cig3", {"method": "cig", "w": 3.0}),
        ("bias2", {"method": "bias", "beta": 2.0}), ("bias5", {"method": "bias", "beta": 5.0}),
        ("retreat", {"method": "retreat"}),
        ("worm", {"method": "wormhole"}),
    ]
    specs = []
    for d in ("AB", "BA"):
        for ts in ts_list:
            for i in inits:
                for name, extra in methods:
                    s = spatial_pair(d)
                    s.update(id=f"{tag}_e2_sp{d}_{name}_ts{ts}_i{i}", init=i, seed=7, ts=ts, tag="e2", variant=name,
                             method=extra.get("method", name))
                    s.update({k: v for k, v in extra.items() if k != "method"})
                    if s["method"] == "wormhole":
                        s["library"] = f"{lib_dir}/{tag}_sp{d}.npz"
                    if i == 0 and ts in (ts_list[len(ts_list) // 2],):
                        s["video"] = True
                    specs.append(s)
    write(f"{out_dir}/{tag}_e2.jsonl", specs)


def ablations(backbone, out_dir, ts_list, inits=range(20), lib_dir=(os.environ.get("BH", "/data/blackhole") + "/lib")):
    """E3: wormhole variants (learned bridge head and component ablations) on the spatial pair."""
    tag = backbone
    variants = [
        ("worm_head", {"head": True}),
        ("worm_noot", {"worm_noot": True}),
        ("worm_noanchor", {"worm_noanchor": True}),
        ("worm_naive", {"worm_naive": True}),
        ("worm_ramp1", {"worm_ramp": 1}),
        ("worm_1chunk", {"worm_chunks": 1}),
    ]
    specs = []
    for d in ("AB", "BA"):
        for ts in ts_list:
            for i in inits:
                for name, extra in variants:
                    s = spatial_pair(d)
                    s.update(id=f"{tag}_e3_sp{d}_{name}_ts{ts}_i{i}", init=i, seed=7, ts=ts, tag="e3",
                             variant=name, method="wormhole", library=f"{lib_dir}/{tag}_sp{d}.npz")
                    for k, v in extra.items():
                        if k == "head":
                            s["head"] = f"{lib_dir}/{tag}_head_{d}.npz"
                        else:
                            s[k] = v
                    specs.append(s)
    write(f"{out_dir}/{tag}_e3.jsonl", specs)
