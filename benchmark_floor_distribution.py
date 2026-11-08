"""Detection floors over many start states.  Paper: Appendix D.
Output: Results/floor_distribution.csv

benchmark_parameter_fault_sweep.py reports floors from ONE start state.  Here the same fault (m2 -> m2*(1+eps), 10 s,
detection when mean|statistic| exceeds 10x that start state's own healthy baseline, as in the sweep) is run from
N_START random starts (U(-1.5,1.5)^4) on the same eps grid, for the raw residual, 8 sub-steps and R2, at h = 0.2 and 0.1.
Reported: per-start monotone floor, and its median / min / max / 10th / 90th percentile over starts."""
import numpy as np
import acrobot as A
import acrobot_parametric as P
from results_io import save_csv, n_trials

MS = (1, 2, 4, 8)
EPS = [10.0 ** k for k in range(-12, 0)] + [3 * 10.0 ** k for k in range(-12, -1)]
EPS = sorted(EPS)
T, RATIO, NS = 10.0, 10.0, n_trials(30, 4)


def stats(s0, m2, h):
    n = int(round(T / h)); s = s0.copy()
    E0 = np.empty(n); Ee = {m: np.empty(n) for m in MS}
    for k in range(n):
        E0[k] = A.energy(s)
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
    return dict(raw=np.mean(np.abs(r[1])), sub8=np.mean(np.abs(r[8])), R2=np.mean(np.abs(R2)))


def floor(flags):
    fl = None
    for e, f in zip(EPS, flags):
        if f:
            fl = e if fl is None else fl
        else:
            fl = None
    return fl


def main():
    rng = np.random.default_rng(2026); starts = [rng.uniform(-1.5, 1.5, 4) for _ in range(NS)]
    rows = []
    for h in (0.2, 0.1):
        fl = {k: [] for k in ("raw", "sub8", "R2")}
        for s0 in starts:
            base = stats(s0, 1.0, h)
            flags = {k: [] for k in fl}
            for e in EPS:
                st = stats(s0, 1.0 + e, h)
                for k in fl:
                    flags[k].append(st[k] > RATIO * base[k])
            for k in fl:
                fl[k].append(floor(flags[k]))
        for k in fl:
            v = np.array([np.nan if x is None else x for x in fl[k]], float)
            lg = np.log10(v[~np.isnan(v)])
            rows.append(dict(h=h, statistic=k, n_starts=NS, n_never=int(np.isnan(v).sum()), median=10 ** np.median(lg),
                             p10=10 ** np.percentile(lg, 10), p90=10 ** np.percentile(lg, 90), min=v[~np.isnan(v)].min(),
                             max=v[~np.isnan(v)].max()))
            print(f"h={h} {k:<5} floor median {rows[-1]['median']:.1e}  [p10 {rows[-1]['p10']:.1e}, p90 {rows[-1]['p90']:.1e}]  "
                  f"range {rows[-1]['min']:.0e}..{rows[-1]['max']:.0e}  never-detected {rows[-1]['n_never']}", flush=True)
    save_csv("floor_distribution.csv", rows)


if __name__ == "__main__":
    main()
