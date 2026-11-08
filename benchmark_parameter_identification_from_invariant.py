"""Identify an unknown simulator parameter from the invariant alone (the no-nominal-parameters setting).
Paper: Appendix F.   Output: Results/parameter_identification.csv

The simulator is treated as opaque: its second-link mass m2_true is unknown (randomised away from nominal).  Only the FORM of the
energy E(x; m2) is specified, not the dynamics.  The finest refinement residual |r^(8)|(m2') is minimised over the candidate m2'
in E; the minimiser is the identified parameter.  The frozen classifier is then run with E(x; m2_hat) (class should be consistent)
and with the nominal E (class should be model).  Compared with identification by matching the step map, which needs the dynamics
f(x; m2) written out.  Several start states are pooled in the objective."""
import numpy as np
from scipy.optimize import minimize_scalar
import acrobot_parametric as P
import fingerprint as F
from results_io import save_csv, n_trials

MS = (1, 2, 4, 8)
H, NSTEP = 0.2, 20


def trajectory(s0, m2, n=NSTEP):
    xs = [s0]
    for _ in range(n):
        xs.append(P.rk4_step(xs[-1], 0.0, H, m2))
    return xs


def subs(m2_true, xs):
    """Sub-stepped end states of the opaque simulator for each macro-step start."""
    out = {mm: [] for mm in MS}
    for s in xs[:-1]:
        for mm in MS:
            x = s
            for _ in range(mm):
                x = P.rk4_step(x, 0.0, H / mm, m2_true)
            out[mm].append(x)
    return out


def resid(xs, sub, m2_cand):
    E0 = np.array([P.energy(s, m2_cand) for s in xs[:-1]])
    return {mm: float(np.mean(np.abs((np.array([P.energy(x, m2_cand) for x in sub[mm]]) - E0) / H))) for mm in MS}


def main():
    rng = np.random.default_rng(7); rows = []
    ns = n_trials(6, 2)
    starts = [rng.uniform(-1.5, 1.5, 4) for _ in range(ns)]
    for m2_true in (0.7, 0.9, 1.1, 1.3, 1.0 + 1e-3):
        data = []
        for s0 in starts:
            xs = trajectory(s0, m2_true); data.append((xs, subs(m2_true, xs)))
        obj = lambda c: float(np.mean([resid(xs, sb, c)[8] for xs, sb in data]))
        res = minimize_scalar(obj, bounds=(0.3, 2.5), method='bounded', options=dict(xatol=1e-13))
        m2_hat = float(res.x)
        def cls(c):
            out = {}
            for xs, sb in data:
                r = resid(xs, sb, c)
                E0 = np.mean([abs(P.energy(s, c)) for s in xs[:-1]])
                k = F.guarded_classify(r[1], r[2], r[4], r[8], 4, E0, H)
                out[k] = out.get(k, 0) + 1
            return out
        rows.append(dict(m2_true=m2_true, m2_hat=m2_hat, rel_error=abs(m2_hat - m2_true) / m2_true,
                         classes_with_fitted_E=str(cls(m2_hat)), classes_with_nominal_E=str(cls(1.0)), n_starts=ns))
        print(f"m2_true={m2_true:.4f}  m2_hat={m2_hat:.6f}  rel.err={rows[-1]['rel_error']:.1e}  fitted E: {rows[-1]['classes_with_fitted_E']}  nominal E: {rows[-1]['classes_with_nominal_E']}", flush=True)
    save_csv("parameter_identification.csv", rows)


if __name__ == "__main__":
    main()
