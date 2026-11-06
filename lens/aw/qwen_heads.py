"""Mechanism analysis: localization heads of Qwen2.5-VL on the robot's own frames.

For frames at the instruction switch (steps/<id>.npz of recorded episodes: the 224 px policy view, upright and mirrored,
with the true positions of both bowls), Qwen2.5-VL-7B gets the frame (upsampled to 448 px, so that its 16 x 16 grid of
merged 28 px tokens coincides with pi0.5's 16 x 16 grid of 14 px SigLIP patches) and the grounding request for the
object the instruction refers to (the referring expression copied from the instruction).  Qwen answers with a box
(greedy); the answer is then fed back (teacher forcing) and, for every layer and head, the attention over the image
tokens is read at the positions that PRODUCE the box coordinates (the query positions right before the first digit of
x1, y1, x2, y2) -- where the model has to "say where" the object is.  --query last uses the last prompt position instead.
Per head: pointing accuracy (argmax token within 1.5 cells of the referred bowl), mass on the referred bowl vs the
other bowl, and the switch score (the argmax follows the instruction: near bowl A under A and near bowl B under B).
Also Qwen's own answer: does the box centre lie nearer to the referred bowl than to the other one?
Writes OUT.npz (maps[frame, instr, layer, head, 256], float16) and OUT.json.
usage: qwen_heads.py OUT --frames 'GLOB' [--model DIR] [--max 200] [--query coord|last]"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch
from PIL import Image

GRID, CELL, RES = 16, 14, 224
PROMPT = 'Locate {ref} in the image and output its bbox coordinates in JSON format.'


def referring_expression(instr):
    """'pick up the black bowl on the stove and place it on the plate' -> 'the black bowl on the stove'."""
    t = instr.strip()
    if t.startswith("pick up "):
        t = t[len("pick up "):]
    return t.split(" and place")[0].strip()


def project(points, cam_pos, cam_rot, fovy_deg):
    pc = (np.asarray(points, np.float64) - np.asarray(cam_pos, np.float64)) @ np.asarray(cam_rot, np.float64)
    z = -pc[:, 2]
    f = 0.5 * RES / math.tan(math.radians(float(fovy_deg)) / 2)
    u = 0.5 * RES + f * pc[:, 0] / z
    v = 0.5 * RES - f * pc[:, 1] / z
    return np.stack([RES - u, v], axis=1)


def cell_centers():
    c = (np.arange(GRID) + 0.5) * CELL
    cx, cy = np.meshgrid(c, c)
    return np.stack([cx.ravel(), cy.ravel()], axis=1)


def frames_from_steps(paths, max_n, when):
    out = []
    for p in sorted(paths):
        z = np.load(p)
        ts = int(z["ts"])
        if ts < 0:
            continue
        sw = z["sw"].astype(bool)
        t = int(np.argmax(sw)) if when == "switch" else 0
        old_lang, new_lang = str(z["old_lang"]), str(z["new_lang"])
        uv = project(np.stack([z["old_obj"][t], z["new_obj"][t]]), z["bcam_pos"], z["bcam_rot"], z["bfovy"])
        out.append(dict(id=os.path.basename(p)[:-4], t=t, img=z["img"][t], instr=(old_lang, new_lang), uv=uv))
        if len(out) >= max_n:
            break
    return out


def frames_from_switch(paths, max_n):
    """save_switch frames (<run>/switch/<id>.npz): 512 px upright agent view rendered at the switch, true objects."""
    out = []
    for p in sorted(paths):
        z = np.load(p, allow_pickle=True)
        res = z["rgb"].shape[0]
        pc = (np.stack([z["old_obj"], z["new_obj"]]).astype(np.float64) - z["cam_pos"]) @ z["cam_rot"]
        zc = -pc[:, 2]
        f = 0.5 * res / math.tan(math.radians(float(z["fovy"])) / 2)
        uv = np.stack([0.5 * res + f * pc[:, 0] / zc, 0.5 * res - f * pc[:, 1] / zc], axis=1)   # upright, not mirrored
        out.append(dict(id=os.path.basename(p)[:-4], t=int(z["t"]), img=z["rgb"],
                        instr=(str(z["old_lang"]), str(z["new_lang"])), uv=uv))
        if len(out) >= max_n:
            break
    return out


def scores(maps, uv):
    """maps (L, H, 256) for one instruction; uv (2, 2): [referred bowl, other bowl] pixel positions."""
    c = cell_centers()
    am = c[maps.argmax(-1)]                                     # (L, H, 2)
    d_ref = np.linalg.norm(am - uv[0], axis=-1)
    d_oth = np.linalg.norm(am - uv[1], axis=-1)
    near = lambda u: (np.linalg.norm(c - u, axis=-1) < 1.5 * CELL).astype(np.float32)  # noqa: E731
    m_ref = (maps * near(uv[0])).sum(-1)
    m_oth = (maps * near(uv[1])).sum(-1)
    return dict(point=(d_ref < 1.5 * CELL).astype(np.float32), closer=(d_ref < d_oth).astype(np.float32),
                mass_ref=m_ref, mass_oth=m_oth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--model", default=(os.environ.get("BH", "/data/blackhole") + "/hf_models/Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--when", default="switch", choices=["switch", "start"])
    ap.add_argument("--query", default="coord", choices=["coord", "last"])
    ap.add_argument("--source", default="steps", choices=["steps", "switch"],
                    help="steps: 224 px policy views (upsampled to 448); switch: 512 px native renders (resized to 504)")
    a = ap.parse_args()
    global GRID, CELL
    import re
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                               attn_implementation="eager", device_map="cuda:0")
    model.eval()
    if a.source == "switch":
        frames = frames_from_switch(glob.glob(a.frames), a.max)
        src_res, q_res = frames[0]["img"].shape[0], 504
    else:
        frames = frames_from_steps(glob.glob(a.frames), a.max, a.when)
        src_res, q_res = RES, 448
    GRID, CELL = q_res // 28, src_res / (q_res // 28)        # Qwen's merged-token grid in source-image pixels
    proc = AutoProcessor.from_pretrained(a.model, min_pixels=q_res * q_res, max_pixels=q_res * q_res)
    img_tok = model.config.image_token_id
    print(f"{len(frames)} frames", flush=True)
    tokz = proc.tokenizer
    all_maps, rows, answers = [], [], []
    for k, fr in enumerate(frames):
        im = Image.fromarray(fr["img"]).resize((q_res, q_res), Image.BICUBIC)
        per_instr = []
        for j, instr in enumerate(fr["instr"]):
            ref = referring_expression(instr)
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(ref=ref)}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda:0")
            thw = inp["image_grid_thw"][0].tolist()
            assert thw[1] // 2 == GRID and thw[2] // 2 == GRID, thw
            n_prompt = inp["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
            ans_ids = gen[0, n_prompt:]
            ans = tokz.decode(ans_ids, skip_special_tokens=True)
            # positions (in the full sequence) that predict the first digit of each box number
            qpos = []
            if a.query == "coord":
                pieces = [tokz.decode([int(t)]) for t in ans_ids]
                in_num, armed, acc = False, False, ""
                for i, pc in enumerate(pieces):
                    if armed:            # only the numbers of the box (the "2" of "bbox_2d" is not one of them)
                        starts = pc.strip()[:1].isdigit()
                        if starts and not in_num:
                            qpos.append(n_prompt + i - 1)
                        in_num = bool(pc.strip()) and pc.strip()[-1].isdigit()
                        if len(qpos) == 4:
                            break
                    acc += pc
                    if not armed and "bbox_2d" in acc and "[" in acc[acc.find("bbox_2d"):]:
                        armed = True
            box = None
            mm = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", ans)
            if mm:
                box = [int(x) * src_res / q_res for x in mm.groups()]      # Qwen2.5-VL: absolute pixels of its input
            if a.query == "last" or len(qpos) < 4:
                qpos = [n_prompt - 1]
            full = {"input_ids": gen[:, : max(qpos) + 1], "attention_mask": torch.ones_like(gen[:, : max(qpos) + 1]),
                    "pixel_values": inp["pixel_values"], "image_grid_thw": inp["image_grid_thw"]}
            with torch.no_grad():
                o = model(**full, output_attentions=True, use_cache=False)
            ids = full["input_ids"][0]
            pos = torch.nonzero(ids == img_tok).flatten()
            assert len(pos) == GRID * GRID, len(pos)
            att = torch.stack([x[0][:, qpos, :].mean(1) for x in o.attentions])   # (L, H, T)
            m = att[:, :, pos].float()                                       # (L, H, 256)
            m = m / (m.sum(-1, keepdim=True) + 1e-12)
            per_instr.append(m.cpu().numpy())
            answers.append(dict(id=fr["id"], instr=j, ref=ref, answer=ans[:120], box=box, n_q=len(qpos)))
        maps = np.stack(per_instr)                                           # (2, L, H, 256)
        all_maps.append(maps.astype(np.float16))
        s_old = scores(maps[0], fr["uv"][[0, 1]])
        s_new = scores(maps[1], fr["uv"][[1, 0]])
        rows.append(dict(id=fr["id"], t=fr["t"], s_old=s_old, s_new=s_new))
        if k % 20 == 0:
            print(k, fr["id"], flush=True)
    M = np.stack(all_maps)
    P_old = np.stack([r["s_old"]["point"] for r in rows])
    P_new = np.stack([r["s_new"]["point"] for r in rows])
    C_old = np.stack([r["s_old"]["closer"] for r in rows])
    C_new = np.stack([r["s_new"]["closer"] for r in rows])
    point = (P_old + P_new) / 2                                              # (F, L, H)
    switch = P_old * P_new                                                   # both instructions answered by this head
    closer = (C_old + C_new) / 2
    mr = np.stack([(r["s_old"]["mass_ref"] + r["s_new"]["mass_ref"]) / 2 for r in rows])
    mo = np.stack([(r["s_old"]["mass_oth"] + r["s_new"]["mass_oth"]) / 2 for r in rows])
    L, H = point.shape[1:]
    head_rank = np.argsort(-point.mean(0).ravel())
    top = [dict(layer=int(i // H), head=int(i % H), point=float(point.mean(0).ravel()[i]),
                switch=float(switch.mean(0).ravel()[i]), closer=float(closer.mean(0).ravel()[i]),
                mass_ref=float(mr.mean(0).ravel()[i]), mass_other=float(mo.mean(0).ravel()[i])) for i in head_rank[:15]]
    # the top-k localization heads pooled (as in training-free grounding with a few heads)
    pooled = {}
    for kk in (1, 3, 5, 10):
        sel = head_rank[:kk]
        def pool(mm):  # noqa: E306
            x = mm.reshape(mm.shape[0], -1, mm.shape[-1])[:, sel].mean(1)
            return x
        po, pn = pool(M[:, 0].astype(np.float32)), pool(M[:, 1].astype(np.float32))
        c = cell_centers()
        uvo = np.stack([fr["uv"][0] for fr in frames]); uvn = np.stack([fr["uv"][1] for fr in frames])
        ho = np.linalg.norm(c[po.argmax(-1)] - uvo, axis=-1) < 1.5 * CELL
        hn = np.linalg.norm(c[pn.argmax(-1)] - uvn, axis=-1) < 1.5 * CELL
        pooled[f"top{kk}"] = dict(point_old=float(ho.mean()), point_new=float(hn.mean()), switch=float((ho & hn).mean()))
    # Qwen's own answer: is the box centre nearer to the referred bowl than to the other one?
    sel_ok, sel_n = 0, 0
    for r_ in answers:
        if r_["box"] is None:
            continue
        fr = next(f for f in frames if f["id"] == r_["id"])
        c_ = np.array([(r_["box"][0] + r_["box"][2]) / 2, (r_["box"][1] + r_["box"][3]) / 2])
        ref_uv, oth_uv = (fr["uv"][0], fr["uv"][1]) if r_["instr"] == 0 else (fr["uv"][1], fr["uv"][0])
        sel_n += 1
        sel_ok += int(np.linalg.norm(c_ - ref_uv) < np.linalg.norm(c_ - oth_uv))
    summary_answers = dict(boxes=sel_n, of=len(answers), referred_bowl_selected=sel_ok,
                           coord_queries=float(np.mean([r_["n_q"] == 4 for r_ in answers])), examples=answers[:4])
    summary = dict(n_frames=len(frames), layers=int(L), heads=int(H), top_heads=top, pooled=pooled,
                   qwen_answers=summary_answers, query=a.query,
                   best_layer_point=[float(x) for x in point.mean(0).max(1)],
                   chance_point=float(np.pi * 1.5 ** 2 / GRID ** 2))
    np.savez_compressed(a.out + ".npz", maps=M, uv=np.stack([fr["uv"] for fr in frames]),
                        ids=np.array([fr["id"] for fr in frames]))
    json.dump(summary, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
