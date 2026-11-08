"""Regime-gated (lexicographic) fingerprint on contact-rich multi-joint MuJoCo models.  Paper: Appendix C.
Output: Results/regime_gated_fingerprint.csv

Level 0 (regime): a macro step is admissible only if NO constraint is active (d.nefc==0: contacts, joint
limits) at the start state and at every sub-step of every refinement m in (1,2,4,8).
Level 1 (conservation): the refinement fingerprint on admissible steps only.
Contacts and joint limits stay ON in the model.  Damping/frictionloss/actuation off so that
free-flight steps are conservative.  Compare ungated (all steps) vs gated."""
import warnings, sys; warnings.filterwarnings("ignore")
import numpy as np, mujoco, gymnasium as gym, fingerprint as F
from results_io import save_csv, n_trials
MS = (1, 2, 4, 8)
BAD = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL)


def root_height_indices(m):
    """Position and velocity coordinates for a root translation along world z."""
    for j in range(m.njnt):
        if m.body_parentid[m.jnt_bodyid[j]] != 0:
            continue
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(m.jnt_qposadr[j])+2, int(m.jnt_dofadr[j])+2
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE and np.allclose(m.jnt_axis[j], [0, 0, 1]):
            return int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
    raise ValueError("No supported vertical root coordinate")


def audit(env_id, ntr=None, nst=60, seed=0, K=1, hoff=(-0.45, 0.05)):
    ntr = ntr or n_trials(24, 6)
    env = gym.make(env_id); env.reset(seed=seed); u = env.unwrapped
    m = u.model; d = mujoco.MjData(m)
    q = {0: 1, 1: 4}.get(int(m.opt.integrator), 1)
    h = float(m.opt.timestep)*int(u.frame_skip)*K
    m.dof_damping[:] = 0; m.dof_frictionloss[:] = 0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    q0 = np.array(u.init_qpos, float); rng = np.random.default_rng(seed)
    height_qpos, height_qvel = root_height_indices(m)
    dt0 = float(m.opt.timestep)
    def energy(qp, qv):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qp; d.qvel[:] = qv; d.ctrl[:] = 0; mujoco.mj_forward(m, d)
        return float(d.energy[0]+d.energy[1]), int(d.nefc)
    def adv(qp, qv, mm):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qp; d.qvel[:] = qv; d.ctrl[:] = 0; m.opt.timestep = h/mm; mx = 0
        for w in BAD: d.warning[w].number = 0
        for _ in range(mm):
            mujoco.mj_step(m, d); mx = max(mx, int(d.nefc))
        if any(d.warning[w].number for w in BAD):      # MuJoCo silently resets the state on an unstable step
            return np.full(m.nq, np.nan), np.full(m.nv, np.nan), 1
        return d.qpos.copy(), d.qvel.copy(), mx
    res = {"ungated": {}, "gated": {}}; frac = []
    for _ in range(ntr):
        qp = q0.copy(); qp[height_qpos] += rng.uniform(*hoff)
        for jj in range(m.njnt):
            if m.jnt_limited[jj] and m.jnt_type[jj] in (2, 3):          # slide / hinge: start inside the range
                lo, hi = m.jnt_range[jj]; qp[m.jnt_qposadr[jj]] = lo + rng.uniform(0.3, 0.7)*(hi-lo)           # drop height; contact occurs mid-run
        qv = rng.uniform(-1.5, 1.5, m.nv); qv[height_qvel] += rng.uniform(-1.0, 0.0)
        R = {mm: [] for mm in MS}; E0l = []; ok = []
        for k in range(nst):
            e0, n0 = energy(qp, qv); gate = (n0 == 0); E0l.append(e0)
            for mm in MS:
                a, b, mx = adv(qp, qv, mm); e, n1 = energy(a, b); R[mm].append(e); gate &= (mx == 0 and n1 == 0)
                if mm == 1: nq, nv = a, b
            ok.append(gate); qp, qv = nq, nv
        E0 = np.array(E0l); ok = np.array(ok); frac.append(ok.mean())
        r = {mm: np.abs((np.array(R[mm])-E0)/h) for mm in MS}
        for name, sel in (("ungated", np.ones(nst, bool)), ("gated", ok)):
            if sel.sum() < 5: c = "no-admissible-steps"
            else:
                rr = {mm: float(r[mm][sel].mean()) for mm in MS}
                c = F.guarded_classify(rr[1], rr[2], rr[4], rr[8], q, np.mean(np.abs(E0[sel])), h)
            res[name][c] = res[name].get(c, 0)+1
    return q, h, res, float(np.mean(frac))
if __name__ == "__main__":
    cfg = [("Hopper-v5", 4, (0.2, 0.8), 24), ("Hopper-v5", 8, (0.2, 0.8), 12), ("Walker2d-v5", 4, (0.2, 0.8), 24),
           ("Walker2d-v5", 8, (0.2, 0.8), 12), ("HalfCheetah-v5", 1, (0.2, 0.8), 20), ("Ant-v5", 1, (0.3, 0.9), 14)]
    rows = []
    for e, K, ho, ns in cfg:
        q, h, res, fr = audit(e, K=K, hoff=ho, nst=ns)
        rows.append(dict(environment=e, declared_order=q, K=K, macro_h=h, steps_per_run=ns, admissible_fraction=fr,
                         ungated=str(res["ungated"]), gated=str(res["gated"])))
        print(f"{e:<16} declared q={q} h={h:g} (K={K}) admissible {fr:.0%}\n   ungated {res['ungated']}\n   gated   {res['gated']}", flush=True)
    save_csv("regime_gated_fingerprint.csv", rows)
