"""Per-start order estimates: energy-refinement exponents versus trajectory convergence (the reference-free advantage).
Paper: Section 6.3.   Output: Results/order_estimator_comparison.csv

The energy fingerprint needs no reference solution.  The conventional way to measure achieved order is a trajectory
convergence study against a fine-step reference.  For each start state of Ant and Humanoid (declared RK4) we compute both
estimates at the shipped step in the conservative free-flight setup and report their spread over starts:
  energy:     mean of (a,b,c) from m = 1,2,4,8 sub-steps (one macro step per state, averaged over 20 steps)
  trajectory: slope log2(err(h0)/err(h0/2)) and log2(err(h0/2)/err(h0/4)) of the max body-centre position error after T seconds
              against a reference at h0/32 with the same integrator (T = 0.5 s and a short horizon 0.1 s).
A good order estimator has a tight distribution around the true order of 2 (Ant, Humanoid here: quaternion base)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, mujoco, gymnasium as gym
import fingerprint as F
from results_io import save_csv, n_trials
MS = (1, 2, 4, 8)
BAD = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL)


def setup(env_id):
    env = gym.make(env_id); env.reset(seed=0); u = env.unwrapped; m = u.model; d = mujoco.MjData(m)
    m.dof_damping[:] = 0; m.dof_frictionloss[:] = 0; m.opt.viscosity = 0; m.opt.density = 0; m.jnt_limited[:] = 0
    m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_ACTUATION
                           | mujoco.mjtDisableBit.mjDSBL_EQUALITY | mujoco.mjtDisableBit.mjDSBL_LIMIT)
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    return u, m, d, float(m.opt.timestep), float(m.opt.timestep)


def energy_exps(u, m, d, h, qv, nsteps):
    def E(qp, qv):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qp; d.qvel[:] = qv; d.ctrl[:] = 0; mujoco.mj_forward(m, d); return float(d.energy[0] + d.energy[1])
    def adv(qp, qv, mm):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qp; d.qvel[:] = qv; d.ctrl[:] = 0; d.time = 0; m.opt.timestep = h / mm
        for w in BAD: d.warning[w].number = 0
        for _ in range(mm): mujoco.mj_step(m, d)
        if any(d.warning[w].number for w in BAD): return None
        return d.qpos.copy(), d.qvel.copy()
    qp = np.array(u.init_qpos, float); E0 = np.empty(nsteps); Ee = {mm: np.empty(nsteps) for mm in MS}
    for k in range(nsteps):
        E0[k] = E(qp, qv)
        for mm in MS:
            r = adv(qp, qv, mm)
            if r is None: return None
            Ee[mm][k] = E(*r)
            if mm == 1: nq, nv = r
        qp, qv = nq, nv
    r = {mm: float(np.mean(np.abs((Ee[mm] - E0) / h))) for mm in MS}
    if r[8] <= F.roundoff_floor(np.mean(np.abs(E0)), h): return None
    return np.array(F.exponents(r[1], r[2], r[4], r[8]))


def traj_slopes(u, m, d, h0, qv, T):
    nst0 = int(round(T / h0))     # same final time for all step sizes
    def final(dt):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = np.array(u.init_qpos, float); d.qvel[:] = qv; d.ctrl[:] = 0; d.time = 0; m.opt.timestep = dt
        for w in BAD: d.warning[w].number = 0
        for _ in range(nst0 * int(round(h0 / dt))): mujoco.mj_step(m, d)
        if any(d.warning[w].number for w in BAD): return None
        mujoco.mj_forward(m, d); return d.xipos.copy()
    ref = final(h0 / 32); p = [final(h0), final(h0 / 2), final(h0 / 4)]
    if ref is None or any(x is None for x in p): return None
    e = [float(np.max(np.linalg.norm(x - ref, axis=1))) for x in p]
    if min(e) <= 0: return None
    return np.array([np.log2(e[0] / e[1]), np.log2(e[1] / e[2])])


def main():
    ntr = n_trials(24, 4); rows = []
    for env_id in ("Ant-v5", "Humanoid-v5"):
        u, m, d, h, dt0 = setup(env_id); rng = np.random.default_rng(0)
        qvs = [rng.uniform(-1, 1, m.nv) for _ in range(ntr)]
        en = [energy_exps(u, m, d, h, qv, n_trials(20, 8)) for qv in qvs]
        est = {"energy exponents (a,b,c mean)": [x.mean() for x in en if x is not None]}
        for T in (0.5, 0.1):
            # Use the native engine timestep and a common integer-step horizon.
            sl = [traj_slopes(u, m, d, dt0, qv, T) for qv in qvs]
            est[f"trajectory slope, T={T}s"] = [x.mean() for x in sl if x is not None]
        for k, v in est.items():
            v = np.array(v)
            requested = float(k.split("T=")[1].split("s")[0]) if "T=" in k else None
            rows.append(dict(environment=env_id, estimator=k, n_valid=len(v), n_starts=ntr,
                             requested_horizon_s=requested,
                             horizon_s=round(requested/dt0)*dt0 if requested is not None else None,
                             median=float(np.median(v)),
                             p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                             frac_within_0p5_of_2=float(np.mean(np.abs(v - 2) < 0.5))))
            print(f"{env_id:<12}{k:<32} valid {len(v):>2}/{ntr}  median {np.median(v):5.2f}  p10-p90 [{np.percentile(v,10):5.2f}, {np.percentile(v,90):5.2f}]  within 2+-0.5: {np.mean(np.abs(v-2)<0.5):.2f}", flush=True)
    save_csv("order_estimator_comparison.csv", rows)


if __name__ == "__main__":
    main()
