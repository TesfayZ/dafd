"""Finite-step validation of the FROZEN classifier (fingerprint.classify), separate from the asymptotic theorem.

Analytic Acrobot, generic Runge-Kutta integrators of order q = 1..4 (Euler, midpoint, Kutta-3, RK4), declared order = true order.
 * development starts (U(-1.5,1.5)^4) vs HELD-OUT starts (theta in U(-pi,pi), omega in U(-3,3): a shifted distribution);
 * model faults m2 -> 1+eps, eps in {0, 1e-4, 1e-3, 1e-2, 1e-1}  (eps > 0 on top of an order-q integrator = mixed fault);
 * macro steps h in {0.05, 0.1, 0.2};
 * degenerate cases: near-equilibrium starts (energy defect ~ 0) and the h-scan of one start where the RK4 defect changes sign.
Reported per cell: counts of every class, model rate with Wilson 95% interval, healthy false-alarm rate with interval.
Constants of the classifier were frozen BEFORE this run; nothing is tuned here.  The guard floor is the float64 round-off
floor of one residual, 50*eps*(|E|+1)/h.
Paper: Section 5.3.   Outputs: Results/finite_step_validation.csv, Results/finite_step_sign_change_scan.csv
"""
import argparse
import glob
import numpy as np
import pandas as pd
import acrobot_parametric as P
import fingerprint as F
from results_io import save_csv, n_trials, RESULTS

MS = (1, 2, 4, 8)
HS = (0.05, 0.1, 0.2)
EPS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
NTR = n_trials(40, 6)
T = 2.0

TABLEAUX = {
    1: ([[0]], [1], [0]),
    2: ([[0, 0], [0.5, 0]], [0, 1], [0, 0.5]),
    3: ([[0, 0, 0], [0.5, 0, 0], [-1, 2, 0]], [1 / 6, 2 / 3, 1 / 6], [0, 0.5, 1]),
    4: ([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]], [1 / 6, 1 / 3, 1 / 3, 1 / 6], [0, 0.5, 0.5, 1]),
}


def rk_step(s, h, m2, q):
    A, b, _ = TABLEAUX[q]
    k = []
    for i in range(len(b)):
        y = s + h * sum(A[i][j] * k[j] for j in range(i)) if i else s
        k.append(P.dynamics(y, 0.0, m2))
    return s + h * sum(bi * ki for bi, ki in zip(b, k))


def energy(s):
    return P.energy(s, m2=1.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - w), min(1.0, c + w))


def trial(s0, h, m2, q):
    n = max(2, int(round(T / h)))
    s = s0.copy(); E0 = np.empty(n); Ee = {m: np.empty(n) for m in MS}
    for k in range(n):
        E0[k] = energy(s)
        for m in MS:
            x = s
            for _ in range(m):
                x = rk_step(x, h / m, m2, q)
            Ee[m][k] = energy(x)
            if m == 1:
                nxt = x
        s = nxt
    if not (np.all(np.isfinite(E0)) and all(np.all(np.isfinite(v)) for v in Ee.values())):
        return "blowup"
    r = {m: float(np.mean(np.abs((Ee[m] - E0) / h))) for m in MS}
    return F.guarded_classify(r[1], r[2], r[4], r[8], q, np.mean(np.abs(E0)), h)


def starts(kind, rng, n):
    if kind == "development":
        return [rng.uniform(-1.5, 1.5, 4) for _ in range(n)]
    if kind == "held_out":
        return [np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi), rng.uniform(-3, 3), rng.uniform(-3, 3)]) for _ in range(n)]
    if kind == "near_equilibrium":
        return [rng.uniform(-1e-3, 1e-3, 4) for _ in range(n)]


def run_cells(sets, orders, tag):
    rows, trials = [], []
    print(f"{'set':<17}{'q':>2}{'h':>6}{'eps':>8}  {'model rate [95% CI]':<24} classes", flush=True)
    for kind in sets:
        for q in orders:
            for h in HS:
                for eps in (EPS if kind != "near_equilibrium" else (0.0, 1e-2)):
                    rng = np.random.default_rng({"development": 1, "held_out": 2, "near_equilibrium": 3}[kind])
                    cls = {}
                    for start_id, s0 in enumerate(starts(kind, rng, NTR)):
                        c = trial(s0, h, 1.0 + eps, q); cls[c] = cls.get(c, 0) + 1
                        trials.append(dict(start_set=kind, true_order=q, h=h, eps=eps, start=start_id,
                                           cluster_id=f"{kind}:{start_id}", initial_state=repr(s0.tolist()), cls=c))
                    n = sum(cls.values()); nm = cls.get("model", 0); lo, hi = wilson(nm, n)
                    flagged = n - sum(cls.get(k, 0) for k in ("consistent", "undetermined", "unstable", "blowup"))
                    flo, fhi = wilson(flagged, n)
                    rows.append(dict(start_set=kind, true_order=q, h=h, eps=eps, n=n, classes=str(cls), model_rate=nm / n,
                                     model_rate_lo=lo, model_rate_hi=hi, undetermined_rate=cls.get("undetermined", 0) / n,
                                     flagged_rate=flagged / n, flagged_lo=flo, flagged_hi=fhi))
                    print(f"{kind:<17}{q:>2}{h:>6}{eps:>8.0e}  {nm / n:.2f} [{lo:.2f},{hi:.2f}]          {cls}", flush=True)
    save_csv(f"finite_step_validation__{tag}.csv", rows)
    save_csv("finite_step_trials.csv" if tag == "all" else f"finite_step_trials__{tag}.csv", trials)


def sign_change_scan():
    s0 = np.array([0.1, -0.2, 0.3, -0.15]); scan = []
    for h in np.concatenate([np.geomspace(0.2, 0.02, 10), np.linspace(0.016, 0.006, 11)]):
        E0 = energy(s0); r = {}
        for m in MS:
            x = s0
            for _ in range(m):
                x = rk_step(x, h / m, 1.0, 4)
            r[m] = (energy(x) - E0) / h
        a = [abs(r[m]) for m in MS]
        fl = F.roundoff_floor(abs(E0), h)
        scan.append(dict(h=float(h), r1=r[1], r2=r[2], r4=r[4], r8=r[8], float64_floor=fl,
                         cls_no_floor=F.classify(a[0], a[1], a[2], a[3], 4, 0.0), cls_with_floor=F.guarded_classify(*a, 4, abs(E0), h)))
        print(f"  h={h:.4f}  r1={r[1]:+.2e}  no-floor={scan[-1]['cls_no_floor']}  with-floor={scan[-1]['cls_with_floor']}")
    save_csv("finite_step_sign_change_scan.csv", scan)


def merge(pattern="finite_step_validation__*.csv"):
    """Concatenate part files (written by --tag runs) into finite_step_validation.csv."""
    parts = sorted(glob.glob(str(RESULTS / pattern)))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    key = ["start_set", "true_order", "h", "eps"]
    dup = int(df.duplicated(subset=key).sum())
    if dup:
        print(f"  merge: {dup} duplicate cells across parts; keeping first")
        df = df.drop_duplicates(subset=key, keep="first")
    save_csv("finite_step_validation.csv", df.to_dict("records"))
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["development", "held_out", "near_equilibrium"])
    ap.add_argument("--orders", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--tag", default="all")
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--merge", action="store_true", help="concatenate Results/finite_step_validation__*.csv")
    a = ap.parse_args()
    if a.merge:
        merge(); return
    if a.scan_only:
        sign_change_scan(); return
    run_cells(a.sets, a.orders, a.tag)
    if a.tag == "all":                       # a default run: merge only its own output, then drop the temporary part
        for p in merge("finite_step_validation__all.csv"):
            import os; os.remove(p)
        sign_change_scan()


if __name__ == "__main__":
    main()
