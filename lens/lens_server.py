"""Policy server with the attention lens (openpi venv; PYTHONPATH=<overlay>:<code dir>:<lens dir>).

All modes of bh_server.BHPolicy (plain, multi, cig, bias, ...) are kept, so that queries before the switch are
identical to those of the earlier runs.  Two modes are added; both take "prompt_old" and use the request's prompt as
the new instruction:

  {"mode": "lens_map", "prompt_old": s, "nsamp_attn": 2, "noise_seed": n}
      attention of the action queries over the image patches under the new and the old instruction (paired flow
      noise) and their robust-scaled difference -> {"map": (512,), "dA_l1": float}
  {"mode": "lens", "prompt_old": s, "beta": b, "variant": v, "nsamp_attn": 2, "noise_seed": n, ["map": (512,)]}
      re-generates the chunk under the new instruction with an additive bias beta * map on the attention logits
      from the action queries to the image keys.  variant "cur": the map is the attention difference measured on
      this observation; "map": the map is given by the client (e.g. the difference measured at the first query of
      the episode, or an oracle object map); "learned": the map is predicted by the focus network (--focus).
  {"mode": "lens_target", "prompt_old": s, "noise_seed": n}
      backbone grounding of both instructions: attention from the language positions to the base / wrist patches,
      per layer -> {"G_new": (L, 512), "G_old": (L, 512), "actions": the plain chunk under the new instruction}
  {"mode": "lens2", "off": [camera, ...], "ae_cut": ["wrist" | "base" | "lang" | "state"], "map": (512,), "beta": b,
   "noise_seed": n}
      truncation by attention: cameras in "off" are masked out of the whole model (as the unused right wrist camera
      is), the action queries cannot attend to the key groups in "ae_cut", and an optional focus map adds beta * map
      to their image-key logits; the chunk is generated under the request's prompt.  With "probe": true and
      "prompt_old", the reply also carries the engagement of the untruncated policy on this observation
      {"dA_l1", "dchunk"}: how much switching the instruction changes its image attention and its chunk.
      With "guide": {"dir": unit xyz, "lam": l, "steps": k} the flow is guided so that the first k actions of the
      chunk move by l * dir (robot action units): the policy itself generates the path, pulled toward a target.
"""
import argparse
import logging
import time

import numpy as np

import bh_server as BS
import lens_core as LC

CUT = -1.0e4


class LensPolicy(BS.BHPolicy):
    def __init__(self, config_name, ckpt_dir, focus=None, lens_w=None):
        super().__init__(config_name, ckpt_dir)
        self.lens = LC.Lens(policy=self.policy, cfg=self.cfg)
        self.learned = LC.LearnedLens(lens_w) if lens_w else None
        self.focus = None
        if focus:
            import lens_focus
            self.focus = lens_focus.FocusPredictor(focus, self.lens)

    @staticmethod
    def _elem(obs, prompt):
        e = {k: v for k, v in obs.items() if k.startswith("observation/")}
        e["prompt"] = prompt
        return e

    def attn_diff(self, obs, prompt_new, prompt_old, S, seed):
        seeds = [int(seed) % 100000 * 10 + 7 + s for s in range(S)]
        inps, ob = self.lens.prepare([self._elem(obs, prompt_old)] * S + [self._elem(obs, prompt_new)] * S)
        P = self.lens.n_prefix(ob)
        _, sink = self.lens.sample(ob, self.lens.noise(seeds * 2), np.zeros((2 * S, P), np.float32), capture=True)
        a_old = LC.image_attention(sink, rows=list(range(S)))
        a_new = LC.image_attention(sink, rows=list(range(S, 2 * S)))
        return a_new - a_old

    def lens_sample(self, obs, prompt, m, beta, noise):
        inps, ob = self.lens.prepare([self._elem(obs, prompt)])
        kb = LC.key_bias_from_map(m, self.lens.n_prefix(ob), beta)[None]
        x0, _ = self.lens.sample(ob, noise, kb, capture=False)
        return np.asarray(self.policy._output_transform({"state": np.asarray(inps[0]["state"]), "actions": x0[0]})
                          ["actions"])

    def grounding(self, obs, prompt_new, prompt_old):
        """Backbone attention from the language positions to the base and wrist patches, averaged over heads and the
        valid language tokens: (L, 2, 512) for (old, new)."""
        _, ob = self.lens.prepare([self._elem(obs, prompt_old), self._elem(obs, prompt_new)])
        PA = self.lens.prefix_attention(ob)
        tm = np.asarray(ob.tokenized_prompt_mask, np.float32)
        return (PA * tm[None, :, :, None]).sum(2) / tm.sum(1)[None, :, None]

    def lens_target(self, obs, ctl):
        t0 = time.monotonic()
        obs = {k: v for k, v in obs.items() if k != "bh"}
        G = self.grounding(obs, obs["prompt"], ctl["prompt_old"])
        M = None
        if self.learned is not None and ctl.get("learned", True):
            _, ob = self.lens.prepare([self._elem(obs, obs["prompt"]), self._elem(obs, ctl["prompt_old"])])
            M = self.learned.map(self.lens, ob)
        out = super().infer({**obs, "bh": {"noise_seed": ctl.get("noise_seed")}})
        out = dict(out)
        out["G_old"] = G[:, 0].astype(np.float32)
        out["G_new"] = G[:, 1].astype(np.float32)
        if M is not None:
            out["M_new"] = np.asarray(M, np.float32)
        out["policy_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
        return out

    def lens2(self, obs, ctl):
        t0 = time.monotonic()
        obs = {k: v for k, v in obs.items() if k != "bh"}
        off = list(ctl.get("off") or [])
        inps, ob = self.lens.prepare([self._elem(obs, obs["prompt"])], off=[off])
        P = self.lens.n_prefix(ob)
        kb = np.zeros(P + self.lens.n_suffix, np.float32)
        for c in ctl.get("ae_cut") or []:
            if c == "wrist":
                kb[LC.IMG_TOKENS:LC.N_IMG] = CUT
            elif c == "base":
                kb[:LC.IMG_TOKENS] = CUT
            elif c == "lang":
                kb[3 * LC.IMG_TOKENS:P] = CUT
            elif c == "state":
                if self.lens.pi05:
                    raise ValueError("pi0.5 has no state token")
                kb[P] = CUT
            else:
                raise ValueError(c)
        if ctl.get("map") is not None and float(ctl.get("beta", 0.0)):
            kb[:LC.N_IMG] += float(ctl["beta"]) * np.asarray(ctl["map"], np.float32)
        g = ctl.get("guide")
        if g:   # attention-guided flow: the chunk is pulled along the direction of the attended target
            vb = self.lens.guidance(inps[0], g["dir"], float(g.get("lam", 0.5)), int(g.get("steps", self.H)))
            x0 = self.lens.sample_guided(ob, self.noise(ctl.get("noise_seed")), kb[None], vb[None])
        else:
            x0, _ = self.lens.sample(ob, self.noise(ctl.get("noise_seed")), kb[None], capture=False)
        acts = np.asarray(self.policy._output_transform({"state": np.asarray(inps[0]["state"]), "actions": x0[0]})
                          ["actions"])
        out = {"actions": acts}
        if ctl.get("probe"):
            out["probe"] = self.engagement(obs, obs["prompt"], ctl["prompt_old"], int(ctl.get("nsamp_attn", 2)),
                                           int(ctl.get("noise_seed") or 0))
        out["policy_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
        return out

    def engagement(self, obs, prompt_new, prompt_old, S, seed):
        """How much the instruction still steers the untruncated policy here: the change of the action queries'
        image attention (dA_l1) and of the chunk (dchunk) between the old and the new instruction (paired noise)."""
        seeds = [int(seed) % 100000 * 10 + 7 + s for s in range(S)]
        inps, ob = self.lens.prepare([self._elem(obs, prompt_old)] * S + [self._elem(obs, prompt_new)] * S)
        P = self.lens.n_prefix(ob)
        x0, sink = self.lens.sample(ob, self.lens.noise(seeds * 2), np.zeros((2 * S, P), np.float32), capture=True)
        acts = self.lens.to_actions(inps, x0)
        a_old = LC.image_attention(sink, rows=list(range(S)))
        a_new = LC.image_attention(sink, rows=list(range(S, 2 * S)))
        return {"dA_l1": float(np.abs(a_new - a_old).sum()),
                "dchunk": float(np.linalg.norm(acts[S:].mean(0) - acts[:S].mean(0)))}

    def infer(self, obs):
        ctl = obs.get("bh") or {}
        mode = ctl.get("mode", "plain")
        if mode == "lens_target":
            return self.lens_target(obs, ctl)
        if mode == "lens2":
            return self.lens2(obs, ctl)
        if mode not in ("lens", "lens_map"):
            return super().infer(obs)
        obs = {k: v for k, v in obs.items() if k != "bh"}
        t0 = time.monotonic()
        S = int(ctl.get("nsamp_attn", 2))
        seed = ctl.get("noise_seed")
        seed = int(seed) if seed is not None else 0
        if mode == "lens_map":
            dA = self.attn_diff(obs, obs["prompt"], ctl["prompt_old"], S, seed)
            out = {"actions": np.zeros((self.H, 7), np.float32), "map": LC.robust_unit(dA),
                   "dA_l1": float(np.abs(dA).sum())}
        else:
            variant = ctl.get("variant", "cur")
            info = {"variant": variant}
            if variant == "cur":
                dA = self.attn_diff(obs, obs["prompt"], ctl["prompt_old"], S, seed)
                m = LC.robust_unit(dA)
                info["dA_l1"] = float(np.abs(dA).sum())
            elif variant == "map":
                m = np.asarray(ctl["map"], np.float32)
            elif variant == "learned":
                if self.focus is None:
                    raise ValueError("variant 'learned' needs --focus")
                m, finfo = self.focus.contrast_map(self._elem(obs, obs["prompt"]), obs["prompt"], ctl["prompt_old"])
                info.update(finfo)
            else:
                raise ValueError(variant)
            acts = self.lens_sample(obs, obs["prompt"], m, float(ctl.get("beta", 1.0)), self.noise(seed))
            out = {"actions": acts, "lens": info}
        out["policy_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8810)
    ap.add_argument("--focus", default="", help="focus network parameters (npz) for variant 'learned'")
    ap.add_argument("--lens_w", default="", help="learned attention lens (npz from lens_train_heads.py)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    from openpi.serving import websocket_policy_server
    pol = LensPolicy(args.config, args.ckpt, args.focus or None, args.lens_w or None)
    # compile every mode before accepting clients (a JIT compile inside a request can outlast the keepalive)
    rng = np.random.default_rng(0)
    dummy = {"observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
             "observation/wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
             "observation/state": rng.standard_normal(8).astype(np.float32), "prompt": "put the bowl on the plate"}
    warm = [{"noise_seed": 1}, {"mode": "multi", "n": 16, "noise_seed": 1},
            {"mode": "lens_map", "prompt_old": "turn on the stove", "nsamp_attn": 2, "noise_seed": 1},
            {"mode": "lens", "variant": "cur", "prompt_old": "turn on the stove", "beta": 1.0, "noise_seed": 1},
            {"mode": "lens", "variant": "map", "map": np.zeros(LC.N_IMG, np.float32), "prompt_old": "x",
             "beta": 1.0, "noise_seed": 1}]
    warm += [{"mode": "lens_target", "prompt_old": "turn on the stove", "noise_seed": 1},
             {"mode": "lens2", "off": ["left_wrist_0_rgb"], "noise_seed": 1},
             {"mode": "lens2", "off": ["left_wrist_0_rgb"], "probe": True, "prompt_old": "turn on the stove",
              "noise_seed": 1},
             {"mode": "lens2", "off": ["left_wrist_0_rgb"], "guide": {"dir": [1.0, 0.0, 0.0], "lam": 0.5, "steps": 5},
              "noise_seed": 1},
             {"mode": "lens2", "ae_cut": ["wrist"], "map": np.zeros(LC.N_IMG, np.float32), "beta": 1.0, "noise_seed": 1}]
    if pol.focus is not None:
        warm.append({"mode": "lens", "variant": "learned", "prompt_old": "turn on the stove", "beta": 1.0,
                     "noise_seed": 1})
    for ctl in warm:
        t0 = time.time()
        out = pol.infer({**dummy, "bh": dict(ctl)})
        logging.info("warm-up %s/%s: %.1fs", ctl.get("mode", "plain"), ctl.get("variant", ""), time.time() - t0)
    # self-check: a lens query with beta = 0 reproduces the custom sampler without bias
    a = pol.infer({**dummy, "bh": {"mode": "lens", "variant": "map", "map": np.zeros(LC.N_IMG, np.float32),
                                   "prompt_old": "x", "beta": 0.0, "noise_seed": 3}})["actions"]
    b = pol.infer({**dummy, "bh": {"mode": "custom_plain", "noise_seed": 3}})["actions"]
    logging.info("self-check |lens(beta=0) - custom_plain| = %.6f", float(np.abs(np.asarray(a) - np.asarray(b)).max()))
    c = pol.infer({**dummy, "bh": {"mode": "lens2", "noise_seed": 3}})["actions"]
    logging.info("self-check |lens2(no cut) - custom_plain| = %.6f", float(np.abs(np.asarray(c) - np.asarray(b)).max()))
    d = pol.infer({**dummy, "bh": {"mode": "lens2", "guide": {"dir": [0.0, 0.0, 1.0], "lam": 0.0, "steps": 5},
                                   "noise_seed": 3}})["actions"]
    e = pol.infer({**dummy, "bh": {"mode": "lens2", "guide": {"dir": [0.0, 0.0, 1.0], "lam": 0.5, "steps": 5},
                                   "noise_seed": 3}})["actions"]
    logging.info("self-check guided flow: |lam=0 - plain| = %.6f, mean dz shift of the first 5 actions at lam=0.5: %.3f",
                 float(np.abs(np.asarray(d) - np.asarray(b)).max()),
                 float(np.mean(np.asarray(e)[:5, 2] - np.asarray(b)[:5, 2])))
    server = websocket_policy_server.WebsocketPolicyServer(policy=pol, host="127.0.0.1", port=args.port,
                                                           metadata=pol.metadata)
    logging.info("serving %s with the attention lens on port %d", args.config, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
