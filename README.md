# WarmHole

Code for the paper **Escaping the Instruction Black Hole of Vision-Language-Action Policies via Optimal-Transport Wormholes** (manuscript under review).

When the instruction of a vision-language-action (VLA) policy changes while the robot is acting, π0 and π0.5 stop listening once they have committed to the first task: the new instruction is absorbed without effect and the robot completes the old task or freezes. We call this the *instruction black hole*. This repository contains

- **an instruction-switching benchmark on LIBERO** that measures the black hole (switch-step sweeps, transmission ratio and event horizon, offline mechanistic analyses) together with the training-free baselines: flush, attention bias, contrastive instruction guidance, scripted retreat-and-restart, and the action-space bridge of our earlier design;
- **Wormhole**, the training-free plug-in proposed in the paper. Mouth A samples N = 16 continuations of the old task, Mouth B holds approach-phase states of successful executions of the new task, entropic optimal transport between the two selects the far mouth y*, a rate-limited lift–transit–settle traversal carries the robot there, and the unmodified policy takes over under the new instruction;
- **the self-trained attention wormhole (AW-π0.5)**, which distils the relocation into the policy. An oracle relocation teacher generates switching data without human demonstrations, and π0.5 is fine-tuned with a learned grounding token [LOC] through which the instruction reaches the action expert. This part also covers the behavior-cloning controls, the diagnostics, the kinematically safe exit of the teacher, and the replication on π0.

## Repository layout

All scripts address files relative to one project root `$BH` (`$BH/code`, `$BH/lens`, `$BH/lens/aw`, `$BH/specs`, ...), and the layout of this repository is that root, so a clone can be used as `$BH` directly.

```
code/                  benchmark harness, training-free remedies and Wormhole
  bh_server.py           openpi policy server: deterministic flow noise, batched samples (Mouth A), CIG and attention bias
  bh_client.py           LIBERO instruction-switch harness; methods vanilla | flush | cig | bias | retreat |
                         wormhole (action-space bridge) | wormhole2 (Wormhole); Tunnel = the rate-limited traversal
  wormhole.py            OT throat (log-domain Sinkhorn), far-mouth selection (select_mouth), traversal geometry
  build_library2.py      entry memory (Mouth B) from held-out no-switch executions of the new task
  build_library.py, train_bridge.py, embed_instr.py, bridge_np.py   action-space bridge and its learned head
  make_specs.py, make_specs_w2.py                                   episode specifications
  bh_analyze.py, compute_anatomy.py, compute_tables.py, make_measured.py, summarize.py, status.py,
  relocation_stats.py, w2_summary.py, bench_overhead.py, viz_runtime.py                analyses
  setup_env.sh, patch_overlay.py, dl_fast.py, verify_ckpt.py        environment, checkpoints, patched openpi overlay
  run_exp.sh, driver.sh, w2_launch.sh, w2_chain.sh, w2_status.sh, viz.sh                orchestration
specs/                 specifications of the no-switch recordings, switch sweeps, remedies and bridge variants
lens/                  policy server with attention access and the LIBERO client of the self-trained part
  lens_server.py, lens_core.py, lens_geom.py   released policy with attention capture; camera geometry
  bh_client_lens.py      client: teacher episodes (oracle relocation, safe exit), DAgger, evaluation of trained policies
  make_specs_lens.py     specifications: teacher data, development / test splits, diagnostics, DAgger, ...
  lens_launch.sh, lens_run.sh, lens_grasp_offset.py, lens_clean_errors.py, patch_overlay_prefix.py, patch_overlay_heads.py
lens/aw/               self-trained attention wormhole
  aw_gemma.py, aw_model.py   [LOC] token and wormhole gates inside π0.5 (adapted from openpi's Gemma, Apache-2.0)
  aw_build_data.py, aw_train.py   training set from teacher episodes; training of the kinds bc | aw | awx | awa | loc
  aw_server.py, aw_eval.sh        serving (with diagnostic switches) and evaluation splits; aw_dagger.sh, dag_check.py
  aw_summary.py, aw_paired.py, aw_dev2.py, aw_dev3.py, aw_pool.py, paper_rows.py, aw_tables_local.py   statistics
  qwen_heads.py, pi_heads.py, heads_small.py, posture.py, h0_trap.py, ...                   mechanism analyses
  aw_leak.py, aw_equiv.py, aw_equiv_p0.py, p0_actconv.py, p0_datacheck.py                   correctness checks
  *_check.py, chain/       checks and the detached pipelines of the experiments (table below)
docs/                  evaluation protocol and the provenance of every file
```

## Setup

We ran everything on one Linux server with 8 NVIDIA L20 GPUs (46 GB each). The software stack is openpi at commit `215abfb` (JAX 0.5.3, Flax 0.10.2, Python 3.11) for the policies, and the LIBERO client of openpi (Python 3.8, robosuite 1.4.1, MuJoCo 3.2.3, EGL rendering). The policies are the released LIBERO checkpoints `pi05_libero` and `pi0_libero`.

```bash
git clone https://github.com/xtli12/WarmHole.git
cd WarmHole
export BH=$PWD              # project root; the scripts default to /data/blackhole
mkdir -p logs && setsid nohup bash code/setup_env.sh > logs/setup.log 2>&1 < /dev/null &
# done when logs/setup.log ends with SETUP DONE; markers logs/*.complete (checkpoints) and logs/client.ok (LIBERO)
```

`setup_env.sh` does the following:

- clones openpi into `$BH/openpi` and creates its virtual environment from the frozen lock file;
- creates the LIBERO client environment `$BH/venvs/libero`;
- downloads and verifies both checkpoints into `$BH/openpi_data`;
- copies openpi into `$BH/code/overlay` and patches it with `code/patch_overlay.py`, which adds an additive float attention mask and an attention-capture hook (off by default).

The analyses of the attention of the language positions also need `lens/patch_overlay_prefix.py` and then `lens/patch_overlay_heads.py`, applied to the same overlay. The script uses download mirrors that were fast from our server (Tsinghua PyPI, npmmirror, gh-proxy, hf-mirror); replace them if they are slow for you. `lens/aw/qwen_heads.py` needs a separate environment with PyTorch, `transformers` and Qwen2.5-VL-7B-Instruct (default location `$BH/hf_models`).

Shell scripts read `$BH`, and the Python defaults read the environment variable `BH` (fallback `/data/blackhole`). Everything the experiments write stays under `$BH` (`runs/`, `lib/`, `logs/`, `specs/w2/`, `specs/lens/`, `lens/aw/data_*`, `lens/aw/run_*`, `lens/aw/pack/`) and is ignored by git. The `library` paths inside `specs/*.jsonl` point to `/data/blackhole/lib`; edit them if your root differs.

## Evaluation protocol

See [docs/protocol.md](docs/protocol.md) for details.

- **Switching pair.** LIBERO-Spatial tasks 7 (bowl on the stove, A) and 9 (bowl on the wooden cabinet, B) share a layout with two identical bowls. A→B runs in the scene of task 7 and B→A in the scene of task 9.
- **LIBERO-Goal.** Eight ordered task pairs in the shared scene.
- **Switch and seeding.** The instruction changes at control step t_s, counted after 10 settle steps. The policy is queried every 5 steps with 10 flow steps. Flow noise is seeded per (initial state, query), so rollouts from the same initial state are identical up to t_s for every method.
- **Outcomes.** From the BDDL predicates: success = the new goal is reached first; capture = the old goal is reached first; freeze (timeout) = neither within 220 steps (spatial) or 300 steps (goal).
- **Splits.**
  - Initial states 0–19 form the test split; each configuration is run on it once.
  - States 40–49 form the development split, where every choice was made.
  - States 20–39 provide the entry memory of Wormhole and the teacher data of the self-trained policies; states 38–39 are held out of training for the validation curves.

## Reproducing the experiments

Long jobs run detached (`setsid nohup ...`). Policy servers are started and checked for health before the LIBERO workers. `PY=$BH/openpi/.venv/bin/python` below.

### 1. The black hole and the training-free remedies

```bash
$PY code/make_specs.py pi05 specs        # E0 recorded no-switch runs, entry-memory runs, E1 switch sweeps
# E2 (remedies) and E3 (bridge variants) come from make_specs.remedies(...) / make_specs.ablations(...);
# the files that were run are in specs/.
# driver.sh BB CONFIG PORT STAGES SERVER_GPU RENDER_GPU ANALYSIS_GPU NWORKERS
code/driver.sh pi05 pi05_libero 8810 e0,ana,e12,heads,e3,e1b 0 1 2 16
code/driver.sh pi0  pi0_libero  8811 e0,ana,e12,e1b 1 0 3 16
```

The driver stages are:

- `e0`: the no-switch recordings and the entry-memory rollouts;
- `ana`: the offline analysis `bh_analyze.py` (instruction response, attention to the instruction, language-free probe);
- `e12`: the switch sweeps and the remedies;
- `heads`: the learned bridge heads;
- `e3`: the bridge variants;
- `e1b`: LIBERO-Goal and LIBERO-Object sweeps.

Tables and figure data come from `compute_anatomy.py`, `compute_tables.py` and `make_measured.py`.

### 2. Wormhole

```bash
for bb in pi05 pi0; do                    # entry memories (Mouth B) from the held-out runs of stage e0
  $PY code/build_library2.py runs/${bb}_e0 "A->B:new" lib/${bb}_spAB_v2.npz 10
  $PY code/build_library2.py runs/${bb}_e0 "B->A:new" lib/${bb}_spBA_v2.npz 10
  for k in 0 1 2 6 7 8 9; do $PY code/build_library2.py runs/${bb}_e0 "goal:$k" lib/${bb}_goal_t${k}_v2.npz 10; done
done
$PY code/make_specs_w2.py specs/w2                         # smoke test and development split (d2)
code/w2_launch.sh d2 pi05 "8301:0 8302:1" 14              # choose the configuration here
code/w2_launch.sh d2 pi0  "8311:2 8312:3" 14
# after the configuration is fixed: test split (run once), LIBERO-Goal and the ablations
$PY code/make_specs_w2.py specs/w2 --test w2,retreat --goal w2,retreat \
    --ablate w2noot,w2near,w2m20,w2nolift,w2norate,w2allph --main_margin 10
code/w2_chain.sh pi05 "8301:0 8302:1" 14 t2 g2 a2
code/w2_chain.sh pi0  "8311:2 8312:3" 14 t2 g2
$PY code/w2_summary.py runs/pi05_t2_p0 runs/pi05_t2_p1
```

`relocation_stats.py` relates the hand-over state to success. `bench_overhead.py` measures the wall-clock cost of a switch against an ordinary query on an idle server. `viz.sh` / `viz_runtime.py` record what the policy attends to and what Wormhole computes at a switch.

### 3. The self-trained attention wormhole

```bash
L=$BH/lens; A=$L/aw
# teacher: the released pi0.5 (lens_server.py) plus the oracle relocation wo_off_h15, training states 20-39 only
$PY $L/make_specs_lens.py $BH/specs/lens --rec --bbs pi05        # no-switch recordings -> own grasp offset
$L/lens_launch.sh lrec pi05 "8401:0 8402:1" 8
$PY $L/lens_grasp_offset.py $BH/runs/lens/pi05_lrec_p* --json $L/grasp_offsets.json
$PY $L/make_specs_lens.py $BH/specs/lens --teach --bbs pi05 --vlm_offset '{"pi05": [-0.017, 0.044], "pi0": [-0.012, 0.044]}'
$L/lens_launch.sh teach pi05 "8401:0 8402:1" 8
$PY $A/aw_build_data.py $A/data_pi05 $BH/runs/lens/pi05_teach_p0 $BH/runs/lens/pi05_teach_p1
# training (4 GPUs, about 1.5 h); --kind bc gives the behavior-cloning control on the same data
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$A $PY $A/aw_train.py --data $A/data_pi05 --kind awx --geo_aug \
  --bs 32 --steps 6000 --lr 5e-5 --lr_new 5e-4 --lam_loc 0.1 --lam_coord 0.2 --log_every 50 --val_every 500 \
  --save_every 1000 --seed 0 --skip_gn 1000 --host_place --out $A/run_awx3
# evaluation: start the servers, then the workers of each split
PHASE=servers $A/aw_eval.sh awx $A/run_awx3/params_6000.npz awx3 "8631:0 8632:1" 6
PHASE=workers $A/aw_eval.sh awx $A/run_awx3/params_6000.npz awx3 "8631:0 8632:1" 6 dev
PHASE=workers $A/aw_eval.sh awx $A/run_awx3/params_6000.npz awx3 "8631:0 8632:1" 6 dev2ext
DIAGNAME=awxawx3 PHASE=workers $A/aw_eval.sh awx $A/run_awx3/params_6000.npz awx3 "8631:0 8632:1" 6 diag aw_off,aw_bias0,aw_hl
PHASE=workers $A/aw_eval.sh awx $A/run_awx3/params_6000.npz awx3 "8631:0 8632:1" 6 test     # once per model
$PY $A/aw_summary.py awdawx3 awpawx3 awtawx3
$PY $A/aw_dev2.py awx3 bc3 --pairs awx3:bc3
(cd $A && $PY aw_paired.py awtawx3 awtbc3)                  # exact McNemar test on paired episodes
```

The model kinds are:

| kind | What the action expert sees | Path of the instruction into the actions |
|---|---|---|
| `bc` | π0.5 with LoRA | instruction tokens and instruction-conditioned image tokens |
| `aw` | as `bc`, plus [LOC] and the attention bias | as `bc`, plus the wormhole |
| `awx` | image tokens that cannot read the instruction, plus the K/V of [LOC] and the bias | only [LOC] |
| `awa` | image tokens that cannot read the instruction, plus the bias | only the attention bias |

`aw_leak.py` checks that no other path exists, including the position offset that depends on the instruction length. The diagnostic switches of `aw_server.py` are set by the client (spec key `aw_ctl`) only after the instruction changes. Their names are `aw_off`, `aw_bias0`, `aw_hl` (hide [LOC]), `aw_tn` / `aw_to` (old words with the new [LOC], and the reverse), and `aw_ht` / `aw_int` (hide the words, or keep the image tokens from reading them).

The experiments after the first models ran as detached pipelines in `lens/aw/chain/`. Each pipeline states its plan and its decision rules in its header; these rules were fixed before any of its results. The lettered rounds:

| Pipeline | Content | Role in the paper |
|---|---|---|
| `a1_dev2_select.sh`, `a2_select_test.sh` | dev2 selection between geometric augmentation and DAgger data; the selected model and its BC control once on the test split | Distillation: configuration selection |
| `b1_start_awc.sh`, `b2_awc_awac.sh` | AW with visible instruction words (AW-words) and the bias-only variant, retrained and evaluated | Distillation: diagnostics |
| `c_a_bc3s1.sh`, `c_b_heads_awx3s1.sh` | second training run of AW and BC; localization heads | Distillation: two runs |
| `d_a_bc3p0.sh`, `d_b_awx3p0.sh` | the same configuration on π0 | Second backbone |
| `e_master.sh` | switch self-play with group-relative self-imitation | Further attempts (negative) |
| `f_master.sh` | contrastive instruction guidance on the trained policies | Further attempts (negative) |
| `g_master.sh`, `h_master.sh`, `i_master.sh` | three relabeling schemes after the switch | Further attempts (negative) |
| `j_master.sh` | kinematically safe exit of the teacher (elbow margin 1.2 rad); AW and BC, two runs | Safe exit |
| `k_master.sh` | diagnostics of the safe-exit models: hidden [LOC], localization heads, elbow replay | Safe exit |
| `l_master.sh` | third training run of the safe-exit AW and BC (confirmation) | Safe exit |
| `m_master.sh` | the safe exit on π0 | Second backbone |

Every pipeline runs as follows: `cd $A && setsid nohup bash chain/<x>.sh > chain/<x>.log 2>&1 < /dev/null &`; `FROM=<phase>` restarts it at a later phase. The π0 lanes `d_a` / `d_b` evaluate only after `aw_equiv_p0.py` has confirmed that the π0 serving path reproduces the released policy (marker `chain/p0_equiv.ok`).

Mechanism analyses:

- `qwen_heads.py` and `pi_heads.py`: localization heads of Qwen2.5-VL-7B and of π0.5 / the trained [LOC] at the switch moments;
- `heads_small.py`: compact copies of these results for plotting;
- `posture.py` and `h0_trap.py`: elbow margin at the switch and the kinematic trap;
- `freeze_anatomy.py`, `si_diag.py`, `switch_onset.py`: anatomy of the remaining failures.

## Practical notes

- **Memory.** A training process needs about 39 GiB of host memory, a policy server about 6.5 GiB, and a LIBERO worker about 1.2 GiB. Budget the total; the pipelines wait for memory and take an evaluation lock.
- **Multi-GPU training.** Use `--host_place`. On our server, direct GPU-to-GPU copies corrupted the replicated state silently, although NCCL reductions were correct. Host placement avoids such copies. Before every multi-GPU run, the pipelines also compare the step-0 validation loss with a single-GPU fingerprint (`aw_train.py --val_only`).
- **Backbone conventions.** `pi0_libero` predicts actions whose first six dimensions are relative to the state, while `pi05_libero` does not. `aw_train.py` and `aw_server.py` follow openpi's convention for each backbone; `p0_actconv.py` and `aw_equiv_p0.py` check it.
- **Development history.** Comments refer to development versions (`cN`, `rNN`, `lensNN`) and to sections of our internal project notes (`handover 7.x`), where each plan was written down before its results. Each released file is the final version of its lineage, and the earlier versions are special cases of it. [docs/PROVENANCE.md](docs/PROVENANCE.md) maps every file to the version that ran on the server.
- **Exploratory code paths.** `bh_client_lens.py`, `lens_server.py` and `make_specs_lens.py` still contain an exploratory attention-lens variant with external VLM grounding, which is not part of the paper. Its helper scripts are not included.
- **Not included.** Rollout results, trained parameters and logs are not part of this repository.

## Citation

```bibtex
@article{warmhole2026,
  title  = {Escaping the Instruction Black Hole of Vision-Language-Action Policies via Optimal-Transport Wormholes},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## Acknowledgements

The policies, the training utilities and the LIBERO client build on [openpi](https://github.com/Physical-Intelligence/openpi) (Apache-2.0), and the simulator is [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). `lens/aw/aw_gemma.py` is a modified copy of openpi's Gemma module and keeps its license header.
