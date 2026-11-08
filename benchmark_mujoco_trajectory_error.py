"""Does the achieved integrator order change simulated trajectories?  (conservative free-flight setup, no policy.)

For each environment: the shipped timestep h0 and refinements h0/2, h0/4, ...; reference = h0/32 with the same integrator.
Error = max over bodies of the world-frame distance between body centres of mass (xipos) after T seconds, median over starts.
Global convergence slope from successive errors.  A quaternion free joint should converge at slope 2, ordinary joints at 4.
This measures a trajectory-level consequence in free flight; it does NOT measure any effect on a trained policy.
Paper: Section 6.3.   Output: Results/mujoco_trajectory_error.csv
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import mujoco
import gymnasium as gym
from results_io import save_csv, n_trials

ENVS = ["Hopper-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5"]
T = 0.5
LEVELS = (1, 2, 4, 8)        # timestep = h0 / level
REF = 32


BAD = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL)
N_RESET = [0]


def prepare(env_id):
    env = gym.make(env_id); env.reset(seed=0); u = env.unwrapped
    m = u.model
    m.dof_damping[:] = 0; m.dof_frictionloss[:] = 0; m.opt.viscosity = 0; m.opt.density = 0; m.jnt_limited[:] = 0
    m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_ACTUATION
                           | mujoco.mjtDisableBit.mjDSBL_EQUALITY | mujoco.mjtDisableBit.mjDSBL_LIMIT)
    return u, m, mujoco.MjData(m), float(m.opt.timestep)


def final_positions(m, d, qpos0, qvel0, dt, h0_, NSTEPS0):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos0; d.qvel[:] = qvel0; d.ctrl[:] = 0; d.time = 0
    m.opt.timestep = dt
    for w in BAD: d.warning[w].number = 0
    for _ in range(NSTEPS0 * int(round(h0_ / dt))):    # same final time for every step size (NSTEPS0 shipped steps)
        mujoco.mj_step(m, d)
    if any(d.warning[w].number for w in BAD):          # MuJoCo silently resets the state on an unstable step
        N_RESET[0] += 1
        return np.full((m.nbody, 3), np.nan)
    mujoco.mj_forward(m, d)
    return d.xipos.copy()


def main():
    ntr = n_trials(8, 3)
    rows = []
    for env_id in ENVS:
        u, m, d, h0 = prepare(env_id)
        integ = {0: "Euler", 1: "RK4"}.get(int(m.opt.integrator), "other")
        nst0 = int(round(T / h0))            # horizon = nst0 shipped steps; T/h0 is not an integer for Humanoid (0.5/0.003)
        rng = np.random.default_rng(0)
        errs = {lv: [] for lv in LEVELS}
        for _ in range(ntr):
            qp = np.array(u.init_qpos, float)
            qv = rng.uniform(-1, 1, m.nv)
            ref = final_positions(m, d, qp, qv, h0 / REF, h0, nst0)
            for lv in LEVELS:
                p = final_positions(m, d, qp, qv, h0 / lv, h0, nst0)
                errs[lv].append(float(np.max(np.linalg.norm(p - ref, axis=1))))
        errs = {lv: [e for e in v if np.isfinite(e)] for lv, v in errs.items()}
        print(f'  {env_id}: finite runs per level {[len(errs[lv]) for lv in LEVELS]}, resets so far {N_RESET[0]}', flush=True)
        med = {lv: float(np.median(errs[lv])) if errs[lv] else float('nan') for lv in LEVELS}
        slopes = {lv: float(np.log2(med[lv] / med[lv * 2])) for lv in LEVELS[:-1] if med[lv * 2] > 0}
        print(f"{env_id:<14}{integ:<6} h0={h0:g}  median error at h0: {med[1]:.2e}  slopes {slopes}", flush=True)
        for lv in LEVELS:
            rows.append(dict(environment=env_id, declared_integrator=integ, shipped_timestep=h0, level=lv, timestep=h0 / lv,
                             horizon_s=nst0*h0, requested_horizon_s=T, n_starts=ntr,
                             n_valid=len(errs[lv]), median_max_body_position_error_m=med[lv],
                             convergence_slope_to_next=slopes.get(lv)))
    save_csv("mujoco_trajectory_error.csv", rows)


if __name__ == "__main__":
    main()
