"""Declared-versus-achieved integrator order of Gymnasium MuJoCo environments (v5).  Paper: Section 6.2.
Output: Results/mujoco_integrator_order_audit.csv

Conservative setup (declared by us, applied identically to every env): damping, friction loss,
fluid forces, joint limits, contacts, actuation and equality-free passive forces off; gravity and
springs kept.  Invariant: MuJoCo's own mechanical energy (mj_energy).  Random joint velocities,
free flight. Macro step h = native engine timestep, scaled by K in (1,4,16,64,256) until the finest
residual clears the float64 floor.  Declared order from model.opt.integrator."""
import warnings, sys; warnings.filterwarnings("ignore")
import numpy as np, mujoco, gymnasium as gym
import fingerprint as F
from results_io import save_csv, n_trials
MS = (1, 2, 4, 8)
BAD = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL)
nan_state = lambda m: (np.full(m.nq, np.nan), np.full(m.nv, np.nan))
INT = {0: ("Euler", 1), 1: ("RK4", 4), 2: ("implicit", 1), 3: ("implicitfast", 1)}

def audit(env_id, ntr=None, nsteps=None, seed=0, Ks=(1, 4, 16, 64, 256), trial_rows=None):
    ntr = ntr or n_trials(12, 4); nsteps = nsteps or n_trials(20, 8)
    env = gym.make(env_id); env.reset(seed=seed); u = env.unwrapped
    m = u.model; d = mujoco.MjData(m)
    iname, q = INT[int(m.opt.integrator)]
    dt0 = float(m.opt.timestep); fs = int(u.frame_skip)
    m.dof_damping[:] = 0; m.dof_frictionloss[:] = 0; m.opt.viscosity = 0; m.opt.density = 0
    m.jnt_limited[:] = 0
    m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_ACTUATION
                           | mujoco.mjtDisableBit.mjDSBL_EQUALITY | mujoco.mjtDisableBit.mjDSBL_LIMIT)
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    q0 = np.array(u.init_qpos, float)
    starts = np.random.default_rng(seed).uniform(-1, 1, (ntr, m.nv))
    def energy(qp, qv):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qp; d.qvel[:] = qv; d.ctrl[:] = 0; mujoco.mj_forward(m, d)
        return float(d.energy[0] + d.energy[1])
    def advance(qp, qv, h, mm):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qp; d.qvel[:] = qv; d.ctrl[:] = 0; d.time = 0
        m.opt.timestep = h/mm
        for w in BAD: d.warning[w].number = 0
        for _ in range(mm): mujoco.mj_step(m, d)
        if any(d.warning[w].number for w in BAD):      # MuJoCo silently resets the state on an unstable step
            return nan_state(m)
        return d.qpos.copy(), d.qvel.copy()
    out = {}
    for K in Ks:
        h = dt0*K
        cls = {}; r8s = []; ex = []
        for t in range(ntr):
            qv = starts[t].copy()
            qp = q0.copy()
            E0 = np.empty(nsteps); Ee = {mm: np.empty(nsteps) for mm in MS}; ok = True
            for k in range(nsteps):
                E0[k] = energy(qp, qv)
                for mm in MS:
                    a, b = advance(qp, qv, h, mm); Ee[mm][k] = energy(a, b)
                    if mm == 1: nq, nv = a, b
                qp, qv = nq, nv
                if not np.isfinite(E0[k]): ok = False; break
            r = {mm: float(np.mean(np.abs((Ee[mm]-E0)/h))) if ok else float("nan") for mm in MS}
            Em = float(np.mean(np.abs(E0))) if ok else float("nan")
            floor = F.roundoff_floor(Em, h)
            c = F.guarded_classify(r[1], r[2], r[4], r[8], q, Em, h)
            cls[c] = cls.get(c, 0)+1
            if trial_rows is not None:
                trial_rows.append(dict(environment=env_id, seed=seed, start=t, K=K, macro_h=h,
                                       native_timestep=dt0, frame_skip=fs, cluster_id=f"{env_id}:{seed}:{t}",
                                       initial_velocity=repr(starts[t].tolist()), cls=c, E_abs_mean=Em,
                                       floor=floor, guard_margin=r[8]/floor,
                                       **{f"r{mm}": r[mm] for mm in MS}))
            if np.isfinite(r[8]): r8s.append(r[8])
            if c not in ("undetermined", "blowup", "unstable"):
                ex.append(F.exponents(r[1], r[2], r[4], r[8]))
        det = ntr - sum(cls.get(k, 0) for k in ("undetermined", "blowup", "unstable"))
        out[K] = (h, cls, np.median(np.array(ex), 0).round(2) if ex else None, float(np.median(r8s)) if r8s else None)
        if det >= ntr//2: break
    env.close()
    return iname, q, dt0, fs, out

if __name__ == "__main__":
    ids = sys.argv[1:] or ["Reacher-v5", "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Swimmer-v5",
                           "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5", "HumanoidStandup-v5", "Pusher-v5"]
    rows, trials = [], []
    for e in ids:
        try:
            iname, q, dt0, fs, out = audit(e, trial_rows=trials)
            K = max(out); h, cls, ex, r8 = out[K]
            print(f"{e:<28} declared {iname:<8}(q={q}) dt={dt0:g}*{fs}  K={K:<3} h={h:<8.3g} classes={cls}  median(a,b,c)={ex}", flush=True)
            rows.append(dict(environment=e, declared_integrator=iname, declared_order=q, timestep=dt0, frame_skip=fs,
                             K=K, macro_h=h, classes=str(cls), median_a=None if ex is None else ex[0],
                             median_b=None if ex is None else ex[1], median_c=None if ex is None else ex[2]))
        except Exception as ex_:
            rows.append(dict(environment=e, error=type(ex_).__name__))
            print(f"{e:<28} ERROR {type(ex_).__name__}: {str(ex_)[:100]}", flush=True)
    save_csv("mujoco_integrator_order_audit.csv", rows)
    save_csv("mujoco_integrator_order_trials.csv", trials)
    if any("error" in row for row in rows):
        raise SystemExit(1)
