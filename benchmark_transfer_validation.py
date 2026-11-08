"""Transfer of the FROZEN classifier to systems not used in development, and energy-preserving integrators.
Paper: Section 5.4.   Outputs: Results/transfer_validation.csv (per cell), Results/transfer_trials.csv (per-trial
residuals, re-scored by benchmark_classifier_sensitivity.py), Results/transfer_horizon.csv,
Results/energy_preserving_integrator.csv

Classifier (fingerprint.classify), its constants and the round-off floor 50*eps*(|E|+1)/h are exactly those of the
Acrobot experiments; nothing is tuned here.  Three systems not used in development:
  * Henon-Heiles  H = (px^2+py^2+x^2+y^2)/2 + lam*(x^2 y - y^3/3);  fault lam = 1+eps (E_ref uses lam=1)
  * planar Kepler  E = |v|^2/2 - mu/r;                              fault mu  = 1+eps (E_ref uses mu=1)
  * free rigid body (Euler top), m' = m x I^-1 m, E = m.I^-1 m/2;  fault I_3 = 3(1+eps) (E_ref uses I_3 = 3).
    This system was added after the first two and its protocol was fixed BEFORE it was run: I = (1, 2, 3), starts
    m uniform on the unit sphere times U(0.5, 2), h in (0.1, 0.2, 0.4), T = 12, orders 1-4, eps as below, 40 starts per
    cell.  It is a Lie-Poisson (non-canonical) system, so it differs in structure from the other two.
Generic RK integrators of order q = 1..4 with declared order = true order; eps = 0 is healthy.
Horizon check (chaotic dynamics): the residual is a one-step defect, so the horizon enters only
through which states are visited; Henon-Heiles (mixed regular/chaotic at E < 1/6) is re-run at T = 20 and T = 200.
Energy-preserving case: implicit midpoint on the harmonic oscillator conserves the quadratic energy exactly, so beta = 0 for
every k; the classifier should abstain (undetermined), not report a model fault."""
import numpy as np
import fingerprint as F
from benchmark_finite_step_classifier_validation import TABLEAUX, wilson
from results_io import save_csv, n_trials

MS = (1, 2, 4, 8)
EPS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
NTR = n_trials(40, 6)


def hh_f(s, lam):
    x, y, px, py = s
    return np.array([px, py, -x - 2 * lam * x * y, -y - lam * (x * x - y * y)])


def hh_E(s):
    x, y, px, py = s
    return 0.5 * (px * px + py * py + x * x + y * y) + x * x * y - y ** 3 / 3


def hh_starts(rng, n):
    out = []
    while len(out) < n:
        s = rng.uniform(-0.4, 0.4, 4)
        if hh_E(s) < 0.15:
            out.append(s)
    return out


def kep_f(s, mu):
    x, y, vx, vy = s
    r3 = (x * x + y * y) ** 1.5
    return np.array([vx, vy, -mu * x / r3, -mu * y / r3])


def kep_E(s):
    x, y, vx, vy = s
    return 0.5 * (vx * vx + vy * vy) - 1.0 / np.hypot(x, y)


def kep_starts(rng, n):
    out = []
    for _ in range(n):
        e = rng.uniform(0.0, 0.5); th = rng.uniform(0, 2 * np.pi)       # unit semi-major axis, eccentricity e, random phase
        r = 1 - e * np.cos(th); v = np.sqrt(2 / r - 1)
        # flight-path angle from angular momentum sqrt(1-e^2)
        L = np.sqrt(1 - e * e); vt = L / r; vr = np.sqrt(max(v * v - vt * vt, 0.0)) * np.sign(np.sin(th))
        c, s_ = np.cos(th), np.sin(th)
        out.append(np.array([r * c, r * s_, vr * c - vt * s_, vr * s_ + vt * c]))
    return out


I_NOM = np.array([1.0, 2.0, 3.0])


def top_f(s, par):
    I = I_NOM.copy(); I[2] *= par                      # fault: I_3 -> I_3 (1 + eps)
    return np.cross(s, s / I)


def top_E(s):
    return 0.5 * float(np.sum(s * s / I_NOM))


def top_starts(rng, n):
    out = []
    for _ in range(n):
        v = rng.normal(size=3); out.append(v / np.linalg.norm(v) * rng.uniform(0.5, 2.0))
    return out


SYSTEMS = {"henon_heiles": (hh_f, hh_E, hh_starts, (0.1, 0.2, 0.4), 20.0),
           "kepler": (kep_f, kep_E, kep_starts, (0.05, 0.1, 0.2), 6.0),
           "euler_top": (top_f, top_E, top_starts, (0.1, 0.2, 0.4), 12.0)}


def rk_step(f, s, h, par, q):
    A, b, _ = TABLEAUX[q]
    k = []
    for i in range(len(b)):
        y = s + h * sum(A[i][j] * k[j] for j in range(i)) if i else s
        k.append(f(y, par))
    return s + h * sum(bi * ki for bi, ki in zip(b, k))


def residuals(f, E, s0, h, T, par, q):
    """Mean |r^(m)| over the trajectory (m = 1, 2, 4, 8) and mean |E|; None on a non-finite state."""
    n = max(2, int(round(T / h))); s = s0.copy()
    E0 = np.empty(n); Ee = {m: np.empty(n) for m in MS}
    for k in range(n):
        E0[k] = E(s)
        for m in MS:
            x = s
            for _ in range(m):
                x = rk_step(f, x, h / m, par, q)
            Ee[m][k] = E(x)
            if m == 1:
                nxt = x
        s = nxt
    if not (np.all(np.isfinite(E0)) and all(np.all(np.isfinite(v)) for v in Ee.values())):
        return None
    return {m: float(np.mean(np.abs((Ee[m] - E0) / h))) for m in MS}, float(np.mean(np.abs(E0)))


def label(res, h, q):
    if res is None:
        return "blowup"
    r, Em = res
    return F.guarded_classify(r[1], r[2], r[4], r[8], q, Em, h)


def systems():
    rows, trials = [], []
    print(f"{'system':<13}{'q':>2}{'h':>6}{'eps':>8}  classes", flush=True)
    for name, (f, E, st, hs, T) in SYSTEMS.items():
        for q in (1, 2, 3, 4):
            for h in hs:
                for eps in EPS:
                    rng = np.random.default_rng(11)
                    cls = {}
                    for k, s0 in enumerate(st(rng, NTR)):
                        res = residuals(f, E, s0, h, T, 1.0 + eps, q)
                        c = label(res, h, q); cls[c] = cls.get(c, 0) + 1
                        trials.append(dict(system=name, true_order=q, h=h, eps=eps, start=k,
                                           cluster_id=f"{name}:{k}", initial_state=repr(s0.tolist()),
                                           r1=None if res is None else res[0][1], r2=None if res is None else res[0][2],
                                           r4=None if res is None else res[0][4], r8=None if res is None else res[0][8],
                                           E_abs_mean=None if res is None else res[1], cls=c))
                    n = sum(cls.values()); nm = cls.get("model", 0); lo, hi = wilson(nm, n)
                    fl = n - cls.get("consistent", 0) - cls.get("undetermined", 0) - cls.get("unstable", 0) - cls.get("blowup", 0)
                    rows.append(dict(system=name, true_order=q, h=h, eps=eps, n=n, classes=str(cls), model_rate=nm / n,
                                     model_lo=lo, model_hi=hi, flagged=fl))
                    print(f"{name:<13}{q:>2}{h:>6}{eps:>8.0e}  {cls}", flush=True)
    save_csv("transfer_validation.csv", rows)
    save_csv("transfer_trials.csv", trials)


def horizon():
    """Henon-Heiles at h = 0.1: horizon T = 20 (as above) versus T = 200; one-step defect, so only the visited states change."""
    rows = []
    f, E, st, _, _ = SYSTEMS["henon_heiles"]
    for T in (20.0, 200.0):
        for q in (2, 4):
            for eps in (0.0, 1e-3):
                rng = np.random.default_rng(11); cls = {}
                for s0 in st(rng, n_trials(20, 4)):
                    c = label(residuals(f, E, s0, 0.1, T, 1.0 + eps, q), 0.1, q); cls[c] = cls.get(c, 0) + 1
                rows.append(dict(system="henon_heiles", h=0.1, T=T, true_order=q, eps=eps, n=sum(cls.values()), classes=str(cls)))
                print(f"horizon T={T:g} q={q} eps={eps:g}: {cls}", flush=True)
    save_csv("transfer_horizon.csv", rows)


def energy_preserving():
    """Implicit midpoint on x'' = -x (E = (x^2+p^2)/2 conserved exactly by the scheme; order 2)."""
    rows = []
    Mid = lambda h: np.linalg.solve(np.eye(2) - 0.5 * h * np.array([[0, 1], [-1, 0]]),
                                    (np.eye(2) + 0.5 * h * np.array([[0, 1], [-1, 0]])))
    Eq = lambda s: 0.5 * (s[0] ** 2 + s[1] ** 2)
    for h in (0.1, 0.4, 1.0):
        rng = np.random.default_rng(5); cls = {}
        for _ in range(NTR):
            s0 = rng.uniform(-1, 1, 2); n = 100; s = s0.copy()
            E0 = np.empty(n); Ee = {m: np.empty(n) for m in MS}
            for k in range(n):
                E0[k] = Eq(s)
                for m in MS:
                    x = s
                    Mm = Mid(h / m)
                    for _ in range(m):
                        x = Mm @ x
                    Ee[m][k] = Eq(x)
                    if m == 1:
                        nxt = x
                s = nxt
            r = {m: float(np.mean(np.abs((Ee[m] - E0) / h))) for m in MS}
            c = F.guarded_classify(r[1], r[2], r[4], r[8], 2, np.mean(np.abs(E0)), h)
            cls[c] = cls.get(c, 0) + 1
        rows.append(dict(integrator="implicit midpoint (E quadratic, conserved exactly)", declared_order=2, h=h, classes=str(cls)))
        print("implicit midpoint h=%g:" % h, cls, flush=True)
    save_csv("energy_preserving_integrator.csv", rows)


def rescore():
    """Reapply the complete rule to archived residuals; do not regenerate trajectories."""
    import pandas as pd
    from results_io import RESULTS
    trials = pd.read_csv(RESULTS / "transfer_trials.csv")
    rows = []
    trials["cluster_id"] = trials.system + ":" + trials.start.astype(str)
    starts_by_system = {name: sampler(np.random.default_rng(11), int(trials[trials.system == name].start.max()) + 1)
                        for name, (_, _, sampler, _, _) in SYSTEMS.items()}
    trials["initial_state"] = [repr(starts_by_system[x.system][x.start].tolist()) for x in trials.itertuples()]
    trials["cls"] = [F.guarded_classify(x.r1, x.r2, x.r4, x.r8, x.true_order, x.E_abs_mean, x.h)
                     for x in trials.itertuples()]
    for (name, q, h, eps), group in trials.groupby(["system", "true_order", "h", "eps"], sort=False):
        classes = group.cls.value_counts().to_dict(); n = len(group); nm = classes.get("model", 0)
        lo, hi = wilson(nm, n)
        fl = n - sum(classes.get(k, 0) for k in ("consistent", "undetermined", "unstable", "blowup"))
        rows.append(dict(system=name, true_order=q, h=h, eps=eps, n=n, classes=str(classes),
                         model_rate=nm/n, model_lo=lo, model_hi=hi, flagged=fl))
    save_csv("transfer_trials.csv", trials.to_dict("records"))
    save_csv("transfer_validation.csv", rows)


if __name__ == "__main__":
    import sys
    if "--rescore" in sys.argv:
        rescore()
    else:
        energy_preserving()
        systems()
        horizon()
