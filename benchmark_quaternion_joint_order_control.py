"""Where does the order-2 behaviour of Ant/Humanoid come from?  Paper: Section 6.3.
Output: Results/quaternion_joint_order_control.csv

Base + hinged arm under gravity, energy depends on base orientation.
 A: base on a free joint (quaternion)            B: base on 3 slides + 3 hinges (one body, no quaternion)
 C: torque-free asymmetric body on a free joint, no gravity, no arm: the energy depends only on the body-frame angular
    velocity and is blind to orientation errors, so RK4 should keep order 4 even with the quaternion.
The guard is the frozen float64 round-off floor of Definition 1 (50 eps (|E|+1)/h), as in every other MuJoCo benchmark."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, mujoco, fingerprint as F
from results_io import save_csv, n_trials
MS = (1, 2, 4, 8)
ARM = ('<body pos=".3 0 0"><joint type="hinge" axis="0 1 0"/>'
       '<geom type="capsule" fromto="0 0 0 .5 0 0" size=".03" mass=".5"/></body>')
def xml(kind, integ):
    base = '<geom type="box" size=".2 .1 .05" mass="1"/>'
    grav, arm = ("0 0 0", "") if kind == "free_tf" else ("0 0 -9.8", ARM)
    j = '<freejoint/>' if kind in ("free", "free_tf") else ''.join(
        f'<joint type="slide" axis="{a}"/>' for a in ("1 0 0", "0 1 0", "0 0 1")) + ''.join(
        f'<joint type="hinge" axis="{a}"/>' for a in ("0 0 1", "1 0 0", "0 1 0"))
    return (f'<mujoco><option gravity="{grav}" integrator="{integ}" timestep="0.01"/><worldbody>'
            f'<body pos="0 0 2">{j}{base}{arm}</body></worldbody></mujoco>')
def run(kind, integ, q, h=0.05, nst=20, ntr=None, seed=0, rot_only=False):
    ntr = ntr or n_trials(24, 6)
    m = mujoco.MjModel.from_xml_string(xml(kind, integ)); d = mujoco.MjData(m)
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    rng = np.random.default_rng(seed); cls = {}; ex = []
    def E(qp, qv):
        d.qpos[:] = qp; d.qvel[:] = qv; mujoco.mj_forward(m, d); return float(d.energy[0]+d.energy[1])
    BAD = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL)
    def adv(qp, qv, mm):
        # MuJoCo silently resets the state on an unstable step; detect it through the warning counters
        d.qpos[:] = qp; d.qvel[:] = qv; m.opt.timestep = h/mm
        for w in BAD: d.warning[w].number = 0
        for _ in range(mm): mujoco.mj_step(m, d)
        if any(d.warning[w].number for w in BAD): raise RuntimeError("unstable")
        return d.qpos.copy(), d.qvel.copy()
    for _ in range(ntr):
        qp = np.array(m.qpos0, float); qp[:] = m.qpos0
        if kind == "joints": qp[3:6] = rng.uniform(-.6, .6, 3)
        qv = rng.uniform(-3, 3, m.nv)
        E0 = np.empty(nst); Ee = {mm: np.empty(nst) for mm in MS}
        try:
            for k in range(nst):
                E0[k] = E(qp, qv)
                for mm in MS:
                    a, b = adv(qp, qv, mm); Ee[mm][k] = E(a, b)
                    if mm == 1: nq, nv = a, b
                qp, qv = nq, nv
        except RuntimeError:
            cls["blowup"] = cls.get("blowup", 0)+1; continue
        r = {mm: float(np.mean(np.abs((Ee[mm]-E0)/h))) for mm in MS}
        c = F.guarded_classify(r[1], r[2], r[4], r[8], q, np.mean(np.abs(E0)), h); cls[c] = cls.get(c, 0)+1
        if c != "undetermined": ex.append(F.exponents(r[1], r[2], r[4], r[8]))
    return cls, (np.median(np.array(ex), 0).round(2) if ex else None)
ROWS = []
NAMES = {"free": "free (quaternion) + arm, gravity", "joints": "3 slides + 3 hinges + arm, gravity",
         "free_tf": "torque-free asymmetric body, free joint, no gravity"}
for kind in ("free", "joints", "free_tf"):
    for integ, q in ((("RK4", 4),) if kind == "free_tf" else (("RK4", 4), ("Euler", 1))):
        cls, ex = run(kind, integ, q)
        print(f"{kind:<8} declared {integ:<6}(q={q}) ->", cls, ex, flush=True)
        ROWS.append(dict(base_joints=NAMES[kind], declared_integrator=integ, declared_order=q, classes=str(cls),
                         median_a=None if ex is None else ex[0], median_b=None if ex is None else ex[1],
                         median_c=None if ex is None else ex[2]))
save_csv("quaternion_joint_order_control.csv", ROWS)
