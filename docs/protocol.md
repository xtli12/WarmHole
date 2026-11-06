# Evaluation protocol

## Simulator and policies

- **Simulator.** LIBERO (robosuite 1.4.1, MuJoCo 3.2.3, EGL rendering). The agent-view and wrist images are 256×256 and are resized with padding to 224, as in the openpi LIBERO client. Every episode starts with 10 settle steps.
- **Policies.** The openpi checkpoints `pi05_libero` and `pi0_libero` (JAX), with 10 flow-integration steps, queried every 5 control steps (the official evaluation setting).

## Instruction switch

- **Switch step.** The instruction changes at control step t_s, counted after the settle steps.
- **vanilla.** The prompt is replaced at t_s; the queued actions of the current chunk (at most 4) are still executed.
- **flush.** The queue is cleared at t_s and the policy is queried again.
- **Noise.** Flow noise is seeded per (initial state, query index). Rollouts from the same initial state are therefore identical up to t_s for every switch step and method (paired counterfactuals).

## Tasks

- **Spatial pair.** LIBERO-Spatial tasks 7 (bowl on the stove, A) and 9 (bowl on the wooden cabinet, B) share one layout with two identical black bowls.
  - A→B runs in the scene of task 7 and B→A in the scene of task 9.
  - In both directions the new goal is On(akita_black_bowl_2, plate_1).
- **LIBERO-Goal.** All 10 tasks share one scene. The ordered pairs are 8→9, 9→8, 6→2, 2→6, 7→0, 0→7, 1→8 and 8→1.
  - The switch-step sweep uses initial states 0–9.
  - The Wormhole runs use initial states 0–7 at t_s = 20 and 40.

## Outcomes

Outcomes are read from the BDDL predicates:

- **success:** the new goal is reached before the old one;
- **capture:** the old goal is reached first;
- **timeout (freeze):** neither goal is reached within 220 steps (spatial) or 300 steps (goal).

Episodes in which the old task was already finished before t_s are excluded from the escape probability (`done_before_switch`).

## Data splits

| Initial states | Use |
|---|---|
| 0–19 | test split; each configuration is run once, after all choices were fixed |
| 40–49 | development split; every choice of configuration was made here |
| 20–39 | entry memory of Wormhole (no-switch executions of the new task); teacher data and DAgger data of the self-trained policies |
| 38–39 | held out of training; used only for the validation curves |

Switch steps by split:

- **Development split.** Early t_s = 2 and committed t_s ∈ {12, 22, 32, 42}.
- **dev2.** The development split extended with the late switches t_s ∈ {52, 62} (n = 120 per model); `aw_eval.sh dev2ext`.
- **devR.** The same switch steps on procedurally generated layouts (seeds r501–r520).
- **Test split.** t_s ∈ {2, 12, 22, 32, 42, 52, 62}.

Episodes that end because of an infrastructure failure, such as an unreachable policy server, are moved out of the result files (`lens_clean_errors.py`, `e_check.py clean`) and run again. Each episode therefore has exactly one valid result.

## Statistics

- **Success rates.** Reported with Wilson 95% intervals.
- **Comparisons between models.** Paired over identical episodes (same initial state, switch step and seed) and tested with the exact McNemar test (`aw_paired.py`, `aw_dev2.py`, `aw_pool.py`).
- **Several training runs.** Pairs are formed within each run and then pooled.
