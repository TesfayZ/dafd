"""Matched-step and direct-reference controls for the head-to-head comparison.  Paper: Section 7.1.
Outputs: Results/matched_step_control.csv

(1) MATCHED STEP.  The refinement statistics call the simulator at h/8, the screen of Gebrekidan and Gebrerufael (2026) sees logs at dt.  Here the
    screen of Gebrekidan and Gebrerufael (2026) (same permutation test, same 8 paired initial conditions, alpha=0.01, same fault m2 -> m2(1+eps),
    noise-free) is run at dt/2, dt/4, dt/8 as well, so its simulator step equals the finest step our statistics use.
(2) DIRECT REFERENCE.  With a reference implementation available, compare one step of the faulty simulator with one step of the
    nominal one from the same state.  Detected at eps if the largest state difference over the 8 initial conditions exceeds
    DIRECT_TOL (a float64-level tolerance); no integrator argument is needed.
Floor = smallest eps on the decade grid such that it and all larger eps are detected."""
import numpy as np
import predecessor
bugs, ideal_diff = predecessor.require()
import acrobot as A
import acrobot_parametric as P
from results_io import save_csv, QUICK

ALPHA = 0.01
EPS = [1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
MACRO = [(0.2, 10.0), (0.1, 10.0), (0.05, 10.0)]
DIRECT_TOL = 1e-14
if QUICK:
    MACRO = MACRO[:1]


def states_from_obs(traj):
    o = traj[0]
    return np.array([np.arctan2(o[1], o[0]), np.arctan2(o[3], o[2]), o[4], o[5]])


def monotone_floor(flags):
    fl = None
    for e, f in zip(EPS, flags):
        if f:
            fl = e if fl is None else fl
        else:
            fl = None
    return fl


def screen_flags(G, names, kinds, V, dt, T, floor):
    n = int(round(T / dt)); kw = dict(dt=dt, n_traj=8, n_steps=n)
    ref = bugs.acrobot_trajectories(seed=0, ic_seed=0, noise=0.0, **kw)
    out = []
    for e in EPS:
        test = bugs.acrobot_trajectories(params={"m2": 1.0 + e}, seed=1, ic_seed=0, noise=0.0, **kw)
        df = ideal_diff.screen(G, ref, test, V, names=names, kinds=kinds, alpha=ALPHA,
                               unit=ideal_diff.TRAJECTORY, floor=floor)
        out.append(bool(df.detected))
    return out, [states_from_obs(t) for t in ref]


def direct_flags(ics, dt):
    out = []
    for e in EPS:
        d = max(np.abs(P.rk4_step(s, 0.0, dt, 1.0 + e) - P.rk4_step(s, 0.0, dt, 1.0)).max() for s in ics)
        out.append(d > DIRECT_TOL)
    return out


def main():
    G, names, kinds = bugs.acrobot_reference(); V = list(bugs.ACRO_SYMS)
    rows = []
    for dt, T in MACRO:
        res = {}
        for div in (1, 2, 4, 8):
            fl, ics = screen_flags(G, names, kinds, V, dt / div, T, 1e-14)
            res[f"screen_at_dt/{div}"] = monotone_floor(fl)
            print(f"dt={dt} screen at dt/{div}={dt/div:g}: floor {res[f'screen_at_dt/{div}']}", flush=True)
        res["direct_reference_step_compare"] = monotone_floor(direct_flags(ics, dt))
        print(f"dt={dt} direct reference comparison: floor {res['direct_reference_step_compare']}", flush=True)
        rows.append(dict(dt=dt, T=T, **res))
    save_csv("matched_step_control.csv", rows)


if __name__ == "__main__":
    main()
