"""
Matched-compute sub-stepping vs Richardson (H1), energy-measurement noise (H2), and shipped Gymnasium agreement and
fault detection (H4).  Paper: Appendix D (extrapolation) and Section 5.2 (shipped step agrees with the analytic step).
Output: Results/matched_compute_noise_fingerprint.csv

H1  Matched-compute baselines.  Is the Richardson statistic R_2 better than
    simply re-integrating the same step with m sub-steps (raw residual r^(m))?
H2  Measurement noise on the energy evaluations (not process noise).
H4  Shipped Gymnasium AcrobotEnv: agreement with the analytic model and
    detection of an injected LINK_MASS_2 fault through env.step only.
(The refinement-order fingerprint itself is benchmarked by the frozen-classifier scripts, not here.)

Definitions (all residuals divided by the macro-step h):
    r^(m)(x) = [E_ref(Phi_{h/m}^m x) - E_ref(x)] / h
    R_2      = (-r^(1) + 48 r^(2) - 512 r^(4)) / 31      (identical to exp_c)
Cost = number of RK4 steps of size <= h per macro step (R_2 costs 7).
"""
import numpy as np
import acrobot as A
import acrobot_parametric as P
from results_io import save_csv, n_trials

STATE0 = np.array([0.1, -0.2, 0.3, -0.15])
GATE = 10.0
EPS = [0.0, 1e-12, 3e-12, 1e-11, 3e-11, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8,
       1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2,
       3e-2, 0.1, 0.3]
MS = (1, 2, 4, 8, 16)


def E_ref(s):
    return P.energy(s, m2=1.0)


# ---------------------------------------------------------------- H1 / H2
def trajectory_energies(m2, h, T=10.0):
    n = int(round(T / h))
    s = STATE0.copy()
    E0 = np.empty(n)
    Ee = {m: np.empty(n) for m in MS}
    for k in range(n):
        E0[k] = E_ref(s)
        for m in MS:
            x = s
            for _ in range(m):
                x = P.rk4_step(x, 0.0, h / m, m2)
            Ee[m][k] = E_ref(x)
        s = P.rk4_step(s, 0.0, h, m2)
    return E0, Ee


def stats_from_energies(E0, Ee, h, sigma=0.0, rng=None):
    if sigma > 0:
        E0 = E0 + sigma * rng.standard_normal(E0.shape)
        Ee = {m: v + sigma * rng.standard_normal(v.shape) for m, v in Ee.items()}
    r = {m: (Ee[m] - E0) / h for m in MS}
    R2 = (-r[1] + 48 * r[2] - 512 * r[4]) / 31.0
    out = {"raw(cost1)": np.mean(np.abs(r[1])),
           "sub4(cost4)": np.mean(np.abs(r[4])),
           "sub8(cost8)": np.mean(np.abs(r[8])),
           "sub16(cost16)": np.mean(np.abs(r[16])),
           "R2(cost7)": np.mean(np.abs(R2))}
    return out


def thresholds(table, names):
    """Smallest tested eps detected together with every larger grid value."""
    res = {}
    for nm in names:
        base = table[0.0][nm]
        thr = None
        for e in sorted(e for e in table if e > 0):
            if table[e][nm] > GATE * base:
                thr = e if thr is None else thr
            else:
                thr = None
        res[nm] = (thr, base)
    return res


def run_h1h2(h, sigmas, ntr=None):
    ntr = ntr or n_trials(20, 4)
    cache = {e: trajectory_energies(P.M2_REF + e, h) for e in EPS}
    names = ["raw(cost1)", "sub4(cost4)", "sub8(cost8)", "sub16(cost16)", "R2(cost7)"]
    rows = []
    for sg in sigmas:
        rng = np.random.default_rng(7)
        table = {}
        for e in EPS:
            E0, Ee = cache[e]
            if sg == 0:
                table[e] = stats_from_energies(E0, Ee, h)
            else:
                trials = [stats_from_energies(E0, Ee, h, sg, rng) for _ in range(ntr)]
                table[e] = {nm: float(np.median([t[nm] for t in trials])) for nm in names}
        th = thresholds(table, names)
        rows.append((sg, th))
    return rows, names


def rk4_fault(eps):
    return lambda s, tau, h: P.rk4_step(s, tau, h, 1.0 + eps)


def composite_resid(step, states, h, m):
    out = []
    for s in states:
        x = s
        for _ in range(m):
            x = step(x, 0.0, h / m)
        out.append((E_ref(x) - E_ref(s)) / h)
    return np.mean(np.abs(out))


# ---------------------------------------------------------------- H4
def run_h4():
    import gymnasium as gym
    env = gym.make("Acrobot-v1")
    env.reset(seed=0)
    u = env.unwrapped

    def ship_step(s, h):
        u.state = np.array(s, dtype=float)
        u.dt = h
        u.step(1)                       # action 1 -> torque 0
        return np.array(u.state, dtype=float)

    d = np.max(np.abs(ship_step(STATE0, 0.2) - A.rk4_step(STATE0, 0.0, 0.2)))
    print(f"  max |shipped step - analytic RK4 step| at dt=0.2: {d:.2e}")
    H4 = [dict(exp="H4", h=0.2, sigma=0, stat="shipped_vs_analytic_step", baseline=float(d), threshold=None)]

    def run(eps, h=0.2, T=10.0):
        u.LINK_MASS_2 = 1.0 + eps
        n = int(round(T / h))
        s = STATE0.copy()
        raw, sub8, R2 = [], [], []
        for _ in range(n):
            E0 = E_ref(s)
            r = {}
            for m in (1, 2, 4, 8):
                x = s
                for _ in range(m):
                    x = ship_step(x, h / m)
                r[m] = (E_ref(x) - E0) / h
            raw.append(abs(r[1]))
            sub8.append(abs(r[8]))
            R2.append(abs((-r[1] + 48 * r[2] - 512 * r[4]) / 31.0))
            s = ship_step(s, h)
        u.LINK_MASS_2 = 1.0
        return np.mean(raw), np.mean(sub8), np.mean(R2)

    base = run(0.0)
    print(f"  healthy (shipped): raw={base[0]:.3e}  sub8={base[1]:.3e}  R2={base[2]:.3e}")
    print(f"  {'eps':>8} {'raw/bl':>10} {'sub8/bl':>10} {'R2/bl':>10}")
    rows = []
    for e in [1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        v = run(e)
        rat = [v[i] / base[i] for i in range(3)]
        rows.append((e, *rat))
        H4.append(dict(exp="H4", h=0.2, sigma=0, stat=f"ratio_raw_sub8_R2_eps={e:g}", baseline=None, threshold=" ".join(f"{x:.3e}" for x in rat)))
        print(f"  {e:>8.0e} {rat[0]:>10.2e} {rat[1]:>10.2e} {rat[2]:>10.2e}")
    return d, base, rows, H4


def main():
    print("=" * 96)
    print("H1: matched-compute baselines (noise-free), mean|stat| over a 10 s trajectory")
    print("=" * 96)
    allrows = []
    for h in (0.2, 0.1):
        rows, names = run_h1h2(h, [0.0])
        th = rows[0][1]
        print(f"h = {h}")
        print(f"  {'statistic':>14} {'healthy baseline':>18} {'threshold eps':>14}")
        for nm in names:
            print(f"  {nm:>14} {th[nm][1]:>18.3e} {str(th[nm][0]):>14}")
            allrows.append(dict(exp="H1", h=h, sigma=0, stat=nm,
                                baseline=th[nm][1], threshold=th[nm][0]))
    print()
    print("=" * 96)
    print("H2: measurement noise sigma_E on every energy evaluation, h = 0.2 (20 trials, median)")
    print("=" * 96)
    sigmas = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
    rows, names = run_h1h2(0.2, sigmas)
    print(f"  {'sigma_E':>9} " + " ".join(f"{n:>14}" for n in names))
    for sg, th in rows:
        print(f"  {sg:>9.0e} " + " ".join(f"{str(th[n][0]):>14}" for n in names))
        for n in names:
            allrows.append(dict(exp="H2", h=0.2, sigma=sg, stat=n,
                                baseline=th[n][1], threshold=th[n][0]))
    print()
    print("=" * 96)
    print("H4: shipped Gymnasium AcrobotEnv (env.step only), h = 0.2")
    print("=" * 96)
    _, _, _, H4rows = run_h4()
    allrows.extend(H4rows)
    save_csv("matched_compute_noise_fingerprint.csv", allrows)


if __name__ == "__main__":
    main()
