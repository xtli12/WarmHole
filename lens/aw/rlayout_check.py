"""Follow-up of rlayout_check_c1 (read-only): which object moves during the settle steps, for official initial states
and for seeded env.reset() layouts; robot end-effector pose at the start; cabinet joint positions.
usage: rlayout_check.py"""
import os

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DUMMY = [0.0] * 6 + [-1.0]
bd = benchmark.get_benchmark_dict()
suite = bd["libero_spatial"]()


def positions(env):
    out = {}
    for name, st in env.env.object_states_dict.items():
        try:
            out[name] = np.array(st.get_geom_state()["pos"], dtype=np.float64)
        except Exception:  # noqa: BLE001
            pass
    return out


def report(env, obs, tag):
    p0 = positions(env)
    sim = env.env.sim
    jn = [sim.model.joint_id2name(j) for j in range(sim.model.njnt)]
    qp = {n: float(sim.data.qpos[sim.model.jnt_qposadr[j]]) for j, n in enumerate(jn) if n and ("cabinet" in n or "drawer" in n)}
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY)
    p1 = positions(env)
    mv = sorted(((float(np.linalg.norm(p1[n] - p0[n])), n) for n in p0), reverse=True)[:3]
    print(f"  {tag}: ee {np.round(obs['robot0_eef_pos'], 3)} grip {np.round(obs['robot0_gripper_qpos'], 3)} | settle top movers "
          + ", ".join(f"{n} {d:.4f}" for d, n in mv) + f" | cabinet joints {qp}", flush=True)


for scene in (7, 9):
    t = suite.get_task(scene)
    f = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=f, camera_heights=128, camera_widths=128)
    inits = suite.get_task_init_states(scene)
    print(f"== scene {scene}", flush=True)
    for i in (20, 40):
        env.seed(7)
        env.reset()
        obs = env.set_init_state(inits[i])
        report(env, obs, f"official {i}")
    for s in (0, 3):
        np.random.seed(s)
        env.seed(s)
        obs = env.reset()
        report(env, obs, f"seed {s}")
    env.close()
print("== END")
