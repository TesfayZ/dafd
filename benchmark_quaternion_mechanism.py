"""Mechanism of the MuJoCo order loss with quaternion joints.  Paper: Section 6.3 and Appendix E.
Outputs: Results/quaternion_mechanism_order.csv, Results/quaternion_mechanism_replication.csv,
         Results/quaternion_mechanism_convergence.csv, Results/quaternion_mechanism_local_error.csv

(A) Minimal MuJoCo models (heavy top = ball joint at the pivot, centre of mass offset), RK4:
      ball joint | three hinges | free joint with COM at the joint | free joint with COM offset.
    Orientation enters the energy only when the COM is offset from the joint origin (heavy top, offset free body).
(B) Independent NumPy re-implementation of MuJoCo's RK4 update read from engine_forward.c / engine_support.c /
    engine_util_spatial.c: stage positions are formed by ONE quaternion exponential of the weighted stage velocities
    (mju_quatIntegrate), velocities by plain RK4.  It is compared with MuJoCo's own step on the same model and states
    (max abs difference of one macro step), and with a standard embedded RK4 on (q, omega) with renormalisation.
(C) Trajectory convergence on the heavy top against a DOP853 reference (rtol 1e-13): MuJoCo, the NumPy replica, embedded RK4,
    and a commutator-free Lie-group RK4 (CF4, Celledoni-Marthinsen-Owren 2003) that also advances orientation by
    exponentials but keeps the commutator structure (two exponentials in the final update, in the order that matches
    MuJoCo's body-frame, right-multiplied convention).
(D) Local error of the single-exponential update.  Magnus expansion of R' = R hat(w): the exact one-step rotation is
    exp(h wbar + (h^3/12) w x wdot + O(h^4)) while the update applies exp(h sum_j b_j w_j); if the velocity quadrature is
    O(h^4) the one-step orientation error is -(h^3/12) w x wdot + O(h^4).  Checked against a 1e-13 DOP853 step: fitted
    coefficient, cosine with w x wdot, the omega error, and the energy-defect exponent with gravity (orientation enters
    the energy) and without (torque-free; the energy is blind to orientation)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, mujoco
from scipy.integrate import solve_ivp
import fingerprint as F
from results_io import save_csv, n_trials

MS = (1, 2, 4, 8)
BOX = '<geom type="box" size=".2 .1 .05" {pos} mass="1"/>'
XMLS = {
    "heavy top, ball joint": '<body pos="0 0 0"><joint type="ball"/>' + BOX.format(pos='pos="0 0 .3"') + '</body>',
    "heavy top, three hinges": '<body pos="0 0 0"><joint type="hinge" axis="1 0 0"/><joint type="hinge" axis="0 1 0"/>'
                               '<joint type="hinge" axis="0 0 1"/>' + BOX.format(pos='pos="0 0 .3"') + '</body>',
    "free body, COM at joint": '<body pos="0 0 2"><freejoint/>' + BOX.format(pos="") + '</body>',
    "free body, COM offset": '<body pos="0 0 2"><freejoint/>' + BOX.format(pos='pos="0 0 .3"') + '</body>',
}


def build(body, integ="RK4"):
    xml = f'<mujoco><option gravity="0 0 -9.8" integrator="{integ}" timestep="0.01"/><worldbody>{body}</worldbody></mujoco>'
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    return m


def random_state(m, rng):
    qp = np.zeros(m.nq); mujoco.mj_resetData(m, mujoco.MjData(m))
    d = mujoco.MjData(m); qp = d.qpos.copy()
    for j in range(m.njnt):
        a = m.jnt_qposadr[j]; t = m.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_FREE:
            qp[a:a + 3] = rng.uniform(-1, 1, 3); qq = rng.normal(size=4); qp[a + 3:a + 7] = qq / np.linalg.norm(qq)
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            qq = rng.normal(size=4); qp[a:a + 4] = qq / np.linalg.norm(qq)
        else:
            qp[a] = rng.uniform(-1, 1)
    return qp, rng.uniform(-2, 2, m.nv)


BAD = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL)


class Unstable(Exception):
    pass


def mj_advance(m, d, qp, qv, h, mm):
    """Advance one macro step.  MuJoCo silently RESETS the state (qpos -> qpos0) when it detects an unstable step, so the
    warning counters are checked and a reset raises Unstable (labelled blowup), never read as data."""
    d.qpos[:] = qp; d.qvel[:] = qv; d.ctrl[:] = 0; d.time = 0; m.opt.timestep = h / mm
    for w in BAD:
        d.warning[w].number = 0
    for _ in range(mm):
        mujoco.mj_step(m, d)
    if any(d.warning[w].number for w in BAD):
        raise Unstable()
    return d.qpos.copy(), d.qvel.copy()


def mj_energy(m, d, qp, qv):
    d.qpos[:] = qp; d.qvel[:] = qv; mujoco.mj_forward(m, d)
    return float(d.energy[0] + d.energy[1])


def order_audit(m, q, h, ntr, nsteps, seed=0, factor=50.0):
    d = mujoco.MjData(m); rng = np.random.default_rng(seed); cls = {}; ex = []
    for _ in range(ntr):
        qp, qv = random_state(m, rng)
        E0 = np.empty(nsteps); Ee = {mm: np.empty(nsteps) for mm in MS}
        try:
            for k in range(nsteps):
                E0[k] = mj_energy(m, d, qp, qv)
                for mm in MS:
                    a, b = mj_advance(m, d, qp, qv, h, mm); Ee[mm][k] = mj_energy(m, d, a, b)
                    if mm == 1:
                        nq_, nv_ = a, b
                qp, qv = nq_, nv_
        except Unstable:
            cls["blowup"] = cls.get("blowup", 0) + 1; continue
        r = {mm: float(np.mean(np.abs((Ee[mm] - E0) / h))) for mm in MS}
        c = F.guarded_classify(r[1], r[2], r[4], r[8], q, np.mean(np.abs(E0)), h, roundoff_factor=factor)
        cls[c] = cls.get(c, 0) + 1
        if c != "undetermined":
            ex.append(F.exponents(r[1], r[2], r[4], r[8]))
    return cls, (np.median(np.array(ex), 0).round(2) if ex else None)


# ---------- NumPy re-implementation (heavy top: ball joint, COM offset c, pivot inertia J) ----------
def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def rot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def quat_integrate(q, vel, scale):                      # mju_quatIntegrate: q <- q * exp(scale*vel) (local frame)
    q = q / np.linalg.norm(q); n = np.linalg.norm(vel); ang = scale * n
    if ang == 0:
        return q
    ax = vel / n; s = np.sin(ang / 2)
    return qmul(q, np.array([np.cos(ang / 2), ax[0]*s, ax[1]*s, ax[2]*s]))


class Top:
    def __init__(self, m):
        mass = m.body_mass[1]; c = m.body_ipos[1].copy()
        # MuJoCo stores principal moments in the inertial frame, not the body frame.
        inertial_rotation = rot(m.body_iquat[1])
        I = inertial_rotation @ np.diag(m.body_inertia[1]) @ inertial_rotation.T
        self.J = I + mass * (c @ c * np.eye(3) - np.outer(c, c)); self.c = c; self.mass = mass
        self.g = np.array([0, 0, -9.8]); self.Ji = np.linalg.inv(self.J)

    def wdot(self, q, w):
        tau = np.cross(self.c, self.mass * rot(q).T @ self.g)
        return self.Ji @ (np.cross(self.J @ w, w) + tau)

    def energy(self, q, w):
        return 0.5 * w @ self.J @ w - self.mass * self.g @ (rot(q) @ self.c)

    def qdot(self, q, w):
        return 0.5 * qmul(q, np.array([0, *w]))


RK4_A = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1.0]]); RK4_B = np.array([1/6, 1/3, 1/3, 1/6])


def step_mujoco_style(T, q0, w0, h):
    Xq, Xw, Fw = [q0], [w0], [T.wdot(q0, w0)]
    for i in range(1, 4):
        dq = sum(RK4_A[i-1, j] * Xw[j] for j in range(i)); dw = sum(RK4_A[i-1, j] * Fw[j] for j in range(i))
        q = quat_integrate(q0, dq, h); w = w0 + h * dw
        Xq.append(q); Xw.append(w); Fw.append(T.wdot(q, w))
    dq = sum(RK4_B[j] * Xw[j] for j in range(4)); dw = sum(RK4_B[j] * Fw[j] for j in range(4))
    return quat_integrate(q0, dq, h), w0 + h * dw


def step_embedded_rk4(T, q0, w0, h):
    f = lambda q, w: (T.qdot(q / np.linalg.norm(q), w), T.wdot(q / np.linalg.norm(q), w))
    k1 = f(q0, w0); k2 = f(q0 + .5*h*k1[0], w0 + .5*h*k1[1]); k3 = f(q0 + .5*h*k2[0], w0 + .5*h*k2[1])
    k4 = f(q0 + h*k3[0], w0 + h*k3[1])
    q = q0 + h/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]); w = w0 + h/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    return q / np.linalg.norm(q), w


def quat_exp(v):
    n = np.linalg.norm(v)
    return np.array([1.0, 0, 0, 0]) if n == 0 else np.concatenate([[np.cos(n / 2)], np.sin(n / 2) * v / n])


def step_cf4(T, q0, w0, h, reverse=False):
    """Commutator-free Lie-group RK4 (CF4), transcribed to right multiplication R' = R hat(w) (body-frame velocity):
    the products of exponentials are reversed relative to the left-action form of the original paper.  The additive
    (velocity) part reduces to classical RK4.  reverse=True multiplies the exponentials in the other order (a control:
    the order of the product matters, and the wrong order loses the fourth order)."""
    a = T.wdot
    K1w, K1r = h * a(q0, w0), h * w0
    q2 = qmul(q0, quat_exp(K1r / 2)); w2 = w0 + K1w / 2; K2w, K2r = h * a(q2, w2), h * w2
    q3 = qmul(q0, quat_exp(K2r / 2)); w3 = w0 + K2w / 2; K3w, K3r = h * a(q3, w3), h * w3
    e1, e3 = quat_exp(K1r / 2), quat_exp(K3r - K1r / 2)
    q4 = qmul(qmul(q0, e3), e1) if reverse else qmul(qmul(q0, e1), e3)
    w4 = w0 + K3w; K4w, K4r = h * a(q4, w4), h * w4
    A = (3 * K1r + 2 * K2r + 2 * K3r - K4r) / 12; B = (-K1r + 2 * K2r + 2 * K3r + 3 * K4r) / 12
    qn = qmul(qmul(q0, quat_exp(B)), quat_exp(A)) if reverse else qmul(qmul(q0, quat_exp(A)), quat_exp(B))
    return qn / np.linalg.norm(qn), w0 + (K1w + 2 * K2w + 2 * K3w + K4w) / 6


def rotvec(qrel):
    """Rotation vector of a unit quaternion (shortest representative)."""
    if qrel[0] < 0:
        qrel = -qrel
    v = qrel[1:]; n = np.linalg.norm(v)
    return 2 * np.arctan2(n, qrel[0]) * v / n if n > 0 else np.zeros(3)


def exact_step(T, q0, w0, h):
    rhs = lambda t, y: np.concatenate([T.qdot(y[:4] / np.linalg.norm(y[:4]), y[4:]), T.wdot(y[:4] / np.linalg.norm(y[:4]), y[4:])])
    s = solve_ivp(rhs, (0, h), np.concatenate([q0, w0]), method="DOP853", rtol=1e-13, atol=1e-15)
    return s.y[:4, -1] / np.linalg.norm(s.y[:4, -1]), s.y[4:, -1]


def local_error_rows(m, ntr):
    """(D): one-step orientation error of the single-exponential update against the Magnus prediction."""
    rows = []; rng = np.random.default_rng(3)
    for label, grav in (("heavy top (gravity)", True), ("torque-free asymmetric body", False)):
        T = Top(m)
        if not grav:
            T.g = np.zeros(3)
        starts = []
        for _ in range(ntr):
            qq = rng.normal(size=4); starts.append((qq / np.linalg.norm(qq), rng.uniform(-2, 2, 3)))
        for h in (0.05, 0.025, 0.0125, 0.00625):
            coef, cosv, werr, dE = [], [], [], []
            for q0, w0 in starts:
                qe, we = exact_step(T, q0, w0, h); q1, w1 = step_mujoco_style(T, q0, w0, h)
                e = rotvec(qmul(np.array([qe[0], -qe[1], -qe[2], -qe[3]]), q1))
                pred = np.cross(w0, T.wdot(q0, w0))               # w x wdot
                coef.append(float(e @ pred / (pred @ pred)) / h ** 3)   # e = coef * h^3 * (w x wdot); Magnus predicts -1/12
                cosv.append(float(e @ pred / (np.linalg.norm(e) * np.linalg.norm(pred))))
                werr.append(float(np.abs(w1 - we).max())); dE.append(abs(T.energy(q1, w1) - T.energy(qe, we)))
            rows.append(dict(model=label, h=h, n=len(starts), coef_median=float(np.median(coef)), coef_min=float(np.min(coef)),
                             coef_max=float(np.max(coef)), cos_median=float(np.median(cosv)), omega_err_median=float(np.median(werr)),
                             energy_defect_median=float(np.median(dE)), magnus_prediction=-1 / 12))
            print(f"(D) {label:<28} h={h:<8} coef/h^3 median {np.median(coef):+.4f} [{np.min(coef):+.4f},{np.max(coef):+.4f}] (Magnus -1/12 = {-1/12:+.4f})"
                  f"  cos {np.median(cosv):+.4f}  omega err {np.median(werr):.1e}  |dE| {np.median(dE):.1e}", flush=True)
    return rows


def np_order(T, step, h, ntr, nsteps, seed=0):
    rng = np.random.default_rng(seed); cls = {}; ex = []
    for _ in range(ntr):
        qq = rng.normal(size=4); q = qq / np.linalg.norm(qq); w = rng.uniform(-2, 2, 3)
        E0 = np.empty(nsteps); Ee = {mm: np.empty(nsteps) for mm in MS}
        for k in range(nsteps):
            E0[k] = T.energy(q, w)
            for mm in MS:
                a, b = q, w
                for _ in range(mm):
                    a, b = step(T, a, b, h / mm)
                Ee[mm][k] = T.energy(a, b)
                if mm == 1:
                    nq_, nw_ = a, b
            q, w = nq_, nw_
        r = {mm: float(np.mean(np.abs((Ee[mm] - E0) / h))) for mm in MS}
        c = F.guarded_classify(r[1], r[2], r[4], r[8], 4, np.mean(np.abs(E0)), h); cls[c] = cls.get(c, 0) + 1
        if c != "undetermined":
            ex.append(F.exponents(r[1], r[2], r[4], r[8]))
    return cls, (np.median(np.array(ex), 0).round(2) if ex else None)


def quat_err(a, b):
    """Rotation angle between unit quaternions via atan2 (arccos loses half the digits near 0)."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    conj = np.array([a[0], -a[1], -a[2], -a[3]]); rel = qmul(conj, b)
    return 2 * float(np.arctan2(np.linalg.norm(rel[1:]), abs(rel[0])))


def main():
    ntr, nst, h = n_trials(24, 4), n_trials(20, 8), 0.05
    rows = []
    for name, body in XMLS.items():
        for fac in (50.0, 500.0):          # 50 = frozen guard; 500 = ten times more conservative round-off margin
            integ, q = "RK4", 4
            m = build(body, integ); cls, ex = order_audit(m, q, h, ntr, nst, factor=fac)
            print(f"(A) {name:<26} {integ} declared {q} floor factor {fac:g}: {cls} median(a,b,c)={ex}", flush=True)
            rows.append(dict(model=name, integrator=integ, declared_order=q, macro_h=h, floor_factor=fac, classes=str(cls),
                             median_a=None if ex is None else ex[0], median_b=None if ex is None else ex[1],
                             median_c=None if ex is None else ex[2]))
    save_csv("quaternion_mechanism_order.csv", rows)

    m = build(XMLS["heavy top, ball joint"]); d = mujoco.MjData(m); T = Top(m); rng = np.random.default_rng(1)
    rep = []
    for hh in (0.05, 0.01):
        dq, dw, dqe = [], [], []
        for _ in range(n_trials(50, 8)):
            qp, qv = random_state(m, rng)
            a, b = mj_advance(m, d, qp, qv, hh, 1)
            q1, w1 = step_mujoco_style(T, qp, qv, hh); q2, w2 = step_embedded_rk4(T, qp, qv, hh)
            dq.append(quat_err(a, q1)); dw.append(np.abs(b - w1).max()); dqe.append(quat_err(a, q2))
        print(f"(B) h={hh}: |MuJoCo - replica|  rotation {max(dq):.1e} rad, omega {max(dw):.1e};   |MuJoCo - embedded RK4| rotation {np.median(dqe):.1e}", flush=True)
        rep.append(dict(h=hh, replica_rotation_max=max(dq), replica_omega_max=max(dw), embedded_rotation_median=float(np.median(dqe))))
    for label, step in (("replica of MuJoCo RK4 update", step_mujoco_style), ("embedded RK4 + renormalisation", step_embedded_rk4),
                        ("commutator-free Lie RK4 (CF4)", step_cf4)):
        cls, ex = np_order(T, step, h, ntr, nst)
        print(f"(B) {label:<32} {cls} median(a,b,c)={ex}", flush=True)
        rep.append(dict(h=h, scheme=label, classes=str(cls), median_a=None if ex is None else ex[0],
                        median_b=None if ex is None else ex[1], median_c=None if ex is None else ex[2]))
    save_csv("quaternion_mechanism_replication.csv", rep)

    conv = []
    rng = np.random.default_rng(2); Tend = 1.0
    ics = []
    for _ in range(n_trials(10, 3)):
        qp, qv = random_state(m, rng); ics.append((qp, qv))
    def ref(q0, w0):
        rhs = lambda t, y: np.concatenate([T.qdot(y[:4] / np.linalg.norm(y[:4]), y[4:]), T.wdot(y[:4] / np.linalg.norm(y[:4]), y[4:])])
        s = solve_ivp(rhs, (0, Tend), np.concatenate([q0, w0]), method="DOP853", rtol=1e-13, atol=1e-14)
        return s.y[:4, -1] / np.linalg.norm(s.y[:4, -1]), s.y[4:, -1]
    refs = [ref(*ic) for ic in ics]
    hs = [0.05, 0.025, 0.0125, 0.00625]; errs = {k: [] for k in ("MuJoCo", "replica", "embedded RK4", "CF4", "CF4 reversed")}
    for hh in hs:
        n = int(round(Tend / hh)); e = {k: [] for k in errs}
        for (qp, qv), (qr, wr) in zip(ics, refs):
            m.opt.timestep = hh; d.qpos[:] = qp; d.qvel[:] = qv; d.time = 0
            for _ in range(n):
                mujoco.mj_step(m, d)
            e["MuJoCo"].append(max(quat_err(d.qpos, qr), np.abs(d.qvel - wr).max()))
            for k, st in (("replica", step_mujoco_style), ("embedded RK4", step_embedded_rk4), ("CF4", step_cf4),
                          ("CF4 reversed", lambda T_, q_, w_, h_: step_cf4(T_, q_, w_, h_, reverse=True))):
                q, w = qp.copy(), qv.copy()
                for _ in range(n):
                    q, w = st(T, q, w, hh)
                e[k].append(max(quat_err(q, qr), np.abs(w - wr).max()))
        for k in errs:
            errs[k].append(float(np.median(e[k])))
    for k, v in errs.items():
        sl = [float(np.log2(v[i] / v[i + 1])) for i in range(len(hs) - 1)]
        print(f"(C) {k:<14} median error at h={hs}: {['%.1e' % x for x in v]} slopes {np.round(sl, 2)}", flush=True)
        conv.append(dict(scheme=k, hs=str(hs), errors=str(v), slopes=str(np.round(sl, 2).tolist())))
    save_csv("quaternion_mechanism_convergence.csv", conv)
    save_csv("quaternion_mechanism_local_error.csv", local_error_rows(m, n_trials(12, 4)))


if __name__ == "__main__":
    main()
