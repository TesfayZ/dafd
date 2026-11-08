"""
Matched head-to-head with the screen of Gebrekidan and Gebrerufael (2026) (needs its repository: see predecessor.py).
Paper: Section 7.1.  Output: Results/head_to_head_predecessor_screen.csv

Same analytic Acrobot (the comparison implementation's envs.py == Gymnasium 'book'), same 8
paired initial conditions (ic_seed=0), same exact permutation test
(ideal_diff._permutation_greater, alpha=0.01), same fault (m2 -> m2(1+eps)),
noise-free.

Methods
  screen(1e-8)   : screen of Gebrekidan and Gebrerufael (2026), its stated absolute floor 1e-8
  screen(1e-14)  : same screen, float64-level floor
  raw / sub8 / R2: per-trajectory mean |statistic| (refinement layer) fed to the
                   SAME permutation test; absolute floor FLOOR_MINE
Floor reported = smallest eps in the grid such that it and all larger eps are
detected (monotone floor).  Grid is by decades.
"""
import numpy as np
import predecessor
bugs, ideal_diff = predecessor.require()
import acrobot as A
import acrobot_parametric as P
from results_io import save_csv, QUICK

ALPHA = 0.01
FLOOR_MINE = 1e-11
EPS = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
CONFIGS = [(0.2, 10.0), (0.1, 10.0), (0.05, 10.0), (0.01, 4.0), (0.005, 2.0)]
if QUICK:
    CONFIGS = CONFIGS[:2]
MS = (1, 2, 4, 8)


def states_from_obs(traj):
    o = traj[0]
    return np.array([np.arctan2(o[1], o[0]), np.arctan2(o[3], o[2]), o[4], o[5]])


def my_stats(s0, m2, h, n):
    s = s0.copy()
    E0 = np.empty(n)
    Ee = {m: np.empty(n) for m in MS}
    for k in range(n):
        E0[k] = A.energy(s)
        nxt = None
        for m in MS:
            x = s
            for _ in range(m):
                x = P.rk4_step(x, 0.0, h / m, m2)
            Ee[m][k] = A.energy(x)
            if m == 1:
                nxt = x
        s = nxt
    r = {m: (Ee[m] - E0) / h for m in MS}
    R2 = (-r[1] + 48 * r[2] - 512 * r[4]) / 31.0
    return dict(raw=np.mean(np.abs(r[1])), sub8=np.mean(np.abs(r[8])),
                R2=np.mean(np.abs(R2)))


def monotone_floor(flags):
    fl = None
    for e, f in zip(EPS, flags):
        if f:
            if fl is None:
                fl = e
        else:
            fl = None
    return fl


def main():
    (G, names, kinds) = bugs.acrobot_reference()
    V = list(bugs.ACRO_SYMS)
    rows = []
    for dt, T in CONFIGS:
        n = int(round(T / dt))
        kw = dict(dt=dt, n_traj=8, n_steps=n)
        ref = bugs.acrobot_trajectories(seed=0, ic_seed=0, noise=0.0, **kw)
        ics = [states_from_obs(t) for t in ref]
        # sanity: the local RK4 step reproduces the comparison implementation's first step
        o1 = ref[0][1]
        mine = A.rk4_step(ics[0], 0.0, dt)
        th = np.array([np.arctan2(o1[1], o1[0]), np.arctan2(o1[3], o1[2]), o1[4], o1[5]])
        d = np.abs(((mine - th) + np.pi) % (2 * np.pi) - np.pi)[:2].max()
        d = max(d, np.abs(mine[2:] - th[2:]).max())
        print(f"\n=== dt={dt}, T={T}s ({n} steps), 8 paired ICs; step agreement with comparison model: {d:.1e}",
              flush=True)
        base = [my_stats(s, 1.0, dt, n) for s in ics]
        flags = {k: [] for k in ["screen(1e-8)", "screen(1e-14)", "raw", "sub8", "R2"]}
        for e in EPS:
            test = bugs.acrobot_trajectories(params={"m2": 1.0 + e}, seed=1, ic_seed=0,
                                             noise=0.0, **kw)
            for fl, key in ((1e-8, "screen(1e-8)"), (1e-14, "screen(1e-14)")):
                df = ideal_diff.screen(G, ref, test, V, names=names, kinds=kinds,
                                       alpha=ALPHA, unit=ideal_diff.TRAJECTORY, floor=fl)
                flags[key].append(bool(df.detected))
            mt = [my_stats(s, 1.0 + e, dt, n) for s in ics]
            for k in ("raw", "sub8", "R2"):
                a = np.array([x[k] for x in mt])
                b = np.array([x[k] for x in base])
                p = ideal_diff._permutation_greater(a, b, a, b, floor=FLOOR_MINE)
                flags[k].append(bool(p < ALPHA))
            print(f"  eps={e:.0e}  " + "  ".join(f"{k}:{'Y' if flags[k][-1] else '.'}" for k in flags),
                  flush=True)
        out = {k: monotone_floor(v) for k, v in flags.items()}
        print("  floors:", out, flush=True)
        rows.append(dict(dt=dt, T=T, **{f"floor_{k}": v for k, v in out.items()},
                         flags="|".join(f"{k}={''.join('1' if x else '0' for x in v)}"
                                        for k, v in flags.items())))
    save_csv("head_to_head_predecessor_screen.csv", rows)


if __name__ == "__main__":
    main()
