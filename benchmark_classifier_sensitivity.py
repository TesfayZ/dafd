"""Sensitivity of the Definition-1 classifier to its constants, its round-off factor and a misdeclared order.

Paper: Section 5.6.  Two parts.
 (1) Development system (shipped Acrobot, RK4): fresh healthy runs (own seed) give the false-alarm side; the fault side
     re-scores the stored per-trial residuals of benchmark_shipped_acrobot_fingerprint.py (run that first).
 (2) analysis == "unified": the SAME variants re-scored on the stored per-trial residuals of the three transfer systems
     (benchmark_transfer_validation.py, run that first) where the constants were never tuned.  Healthy = eps 0, all
     orders and steps (n = 1440); faults = eps 1e-2 for q = 2,3,4 (n = 1080), strong enough that no integrator masks them.
     The round-off factor (default 50) is varied here as well. Weak Hénon–Heiles faults
     at eps=1e-4, q=2,3,4 are also re-scored (360 descriptive evaluations).
All variants are post hoc: they are reported as sensitivity, not adopted.

Output: Results/classifier_sensitivity.csv
"""
import warnings
warnings.filterwarnings("ignore")
import collections
import itertools

import numpy as np
import pandas as pd

import fingerprint as F
import shipped_acrobot as S
from results_io import RESULTS, save_csv, n_trials

SEED = 777                      # independent of the seeds used by the fingerprint benchmark
FAULT_ATTRS = ["LINK_MASS_1", "LINK_MASS_2", "LINK_LENGTH_1",
               "LINK_COM_POS_1", "LINK_COM_POS_2", "LINK_MOI"]
ROWS = []


def variant(r, q=4, c_thr=0.5, delta=0.4, theta=None, label="mean"):
    """Definition 1 with the constants exposed. Defaults equal fingerprint.classify."""
    # Deliberately exponent-only controls; the unified analysis uses full_label.
    return F.classify(*r, q, c_thr=c_thr, delta=delta, theta=theta, order_rule=label)


def healthy_residuals(n):
    rng = np.random.default_rng(SEED)
    out, Em = [], []
    with S.fault():
        for s0 in S.sample_states(rng, n):
            E0, Ee = S.run_trial(s0)
            d = S.stats(E0, Ee)
            out.append((d["r1"], d["r2"], d["r4"], d["r8"])); Em.append(float(np.mean(np.abs(E0))))
    return np.array(out), np.array(Em)


def fault_residuals(T, pattern):
    return T[T.case.str.contains(pattern, regex=False)][["r1", "r2", "r4", "r8"]].values


def tally(rs, **kw):
    return collections.Counter(variant(r, **kw) for r in rs)


def full_label(r, Em, h, q, c_thr=0.5, delta=0.4, theta_rule="default", factor=50.0):
    """Definition 1 including the guard, with every constant exposed (defaults reproduce fingerprint.classify)."""
    theta = {"default": None, "q-0.5": q - 0.5, "q-1": q - 1.0, "q-1.5": q - 1.5}[theta_rule]
    return F.guarded_classify(*r, q, Em, h, roundoff_factor=factor,
                              c_thr=c_thr, delta=delta, theta=theta)


VARIANTS = [("default (as in the paper)", {}),
            ("c_thr = 0.25", dict(c_thr=0.25)), ("c_thr = 1.0", dict(c_thr=1.0)),
            ("delta = 0.1", dict(delta=0.1)), ("delta = 0.8", dict(delta=0.8)),
            ("theta = q - 0.5 (all q)", dict(theta_rule="q-0.5")), ("theta = q - 1 (all q)", dict(theta_rule="q-1")),
            ("theta = q - 1.5 (all q)", dict(theta_rule="q-1.5")),
            ("round-off factor 10", dict(factor=10.0)), ("round-off factor 500", dict(factor=500.0))]


def unified(T, H, HE):
    """Every variant scored on the development system (fresh healthy RK4 runs; stored 1e-5 faults) and on the transfer systems."""
    path = RESULTS / "transfer_trials.csv"
    if not path.exists():
        print("  (transfer_trials.csv missing: run benchmark_transfer_validation.py for the unified table)")
        return
    X = pd.read_csv(path)
    hlt = X[X.eps == 0.0]
    flt = X[(X.eps == 1e-2) & (X.true_order >= 2)]
    weak = X[(X.system == "henon_heiles") & (X.eps == 1e-4) & (X.true_order >= 2)]
    dev_f = T[T.case.str.startswith("LINK_") & T.case.str.endswith("+1e-05")]
    sets = [("dev_healthy", [(r, e, 0.2, 4) for r, e in zip(H, HE)]),
            ("dev_fault_1e-5", [((x.r1, x.r2, x.r4, x.r8), x.E_abs_mean, 0.2, 4) for x in dev_f.itertuples()]),
            ("transfer_healthy", [((x.r1, x.r2, x.r4, x.r8), x.E_abs_mean, x.h, x.true_order) for x in hlt.itertuples()]),
            ("transfer_fault_1e-2", [((x.r1, x.r2, x.r4, x.r8), x.E_abs_mean, x.h, x.true_order) for x in flt.itertuples()]),
            ("transfer_hh_fault_1e-4", [((x.r1, x.r2, x.r4, x.r8), x.E_abs_mean, x.h, x.true_order) for x in weak.itertuples()])]
    for name, kw in VARIANTS:
        rec = dict(analysis="unified", setting=name)
        for tag, items in sets:
            cnt = collections.Counter()
            for r, E, h, q in items:
                cnt["blowup" if not np.all(np.isfinite(r)) else full_label(r, E, h, q, **kw)] += 1
            n = sum(cnt.values())
            rec[f"{tag}_n"] = n
            rec[f"{tag}_model"] = cnt["model"]
            rec[f"{tag}_other_flag"] = n - sum(cnt[k] for k in ("model", "consistent", "undetermined", "unstable", "blowup"))
            rec[f"{tag}_abstain"] = cnt["undetermined"] + cnt["unstable"] + cnt["blowup"]
        ROWS.append(rec)
        print(name, {k: v for k, v in rec.items() if k not in ("analysis", "setting")}, flush=True)
    # self-check: the default variant equals fingerprint.classify on every healthy transfer trial
    for r, E, h, q in sets[2][1]:
        if np.all(np.isfinite(r)):
            assert full_label(r, E, h, q) == F.classify(*r, q, F.roundoff_floor(E, h), e_scale=E + 1), "full_label drifted"


def main():
    H, HE = healthy_residuals(n_trials(1000, 100))
    nH = len(H)
    # self-check: the exposed-constant variant equals the shipped classifier at default settings
    assert all(variant(r) == F.classify(*r, 4) for r in H), "variant() drifted from fingerprint.classify"

    ex = np.array([F.exponents(*r) for r in H])
    print(f"healthy n={nH}: min exponent {ex.min():.3f}; minima (a,b,c) {ex.min(0).round(3)}")
    ROWS.append(dict(analysis="healthy_margin", setting="min over (a,b,c)", n_healthy=nH,
                     value=round(float(ex.min()), 3)))

    T = pd.read_csv(RESULTS / "shipped_acrobot_fingerprint_trials.csv")
    faults = np.vstack([fault_residuals(T, f"{a} +1e-05") for a in FAULT_ATTRS])
    euler = fault_residuals(T, "CAT euler")
    rk2 = T[T.case == "rk2"][["r1", "r2", "r4", "r8"]].values
    nF = len(faults)

    # 1. (c_thr, delta)
    for c_thr, delta in itertools.product([0.25, 0.5, 1.0, 1.5], [0.1, 0.2, 0.4, 0.8, 1.2]):
        h = tally(H, c_thr=c_thr, delta=delta)
        ROWS.append(dict(analysis="c_thr_delta", setting=f"c_thr={c_thr}, delta={delta}",
                         n_healthy=nH, healthy_not_consistent=nH - h["consistent"],
                         healthy_model=h["model"],
                         faults_detected=sum(variant(r, c_thr=c_thr, delta=delta) == "model" for r in faults),
                         n_faults=nF,
                         euler_labelled_order1=sum(variant(r, c_thr=c_thr, delta=delta) == "order-1" for r in euler),
                         n_euler=len(euler),
                         rk2_labelled_order2=sum(variant(r, c_thr=c_thr, delta=delta) == "order-2" for r in rk2),
                         n_rk2=len(rk2)))

    # 2. consistency threshold theta (paper: q-1 = 3)
    for th in [2.0, 2.5, 3.0, 3.5, 3.8, 4.0]:
        h = tally(H, theta=th)
        ROWS.append(dict(analysis="theta", setting=f"theta={th}", n_healthy=nH,
                         healthy_not_consistent=nH - h["consistent"], healthy_model=h["model"],
                         faults_detected=sum(variant(r, theta=th) == "model" for r in faults), n_faults=nF))

    # 3. misdeclared order q
    for q in [2, 3, 4, 5, 6]:
        h = tally(H, q=q)
        ROWS.append(dict(analysis="declared_q", setting=f"q={q}", n_healthy=nH,
                         healthy_not_consistent=nH - h["consistent"], healthy_model=h["model"],
                         healthy_classes=str(dict(h)),
                         euler_classes=str(dict(tally(euler, q=q))),
                         rk2_classes=str(dict(tally(rk2, q=q)))))

    # 4. order-k label: mean of (a,b,c) (paper) versus the finest exponent c
    for rule in ("mean", "c"):
        k_h = [int(round(np.mean(e) if rule == "mean" else e[2])) for e in ex]
        ROWS.append(dict(analysis="order_label_rule", setting=rule, n_healthy=nH,
                         healthy_label_equals_4=sum(k == 4 for k in k_h),
                         rk2_labelled_order2=sum(variant(r, label=rule) == "order-2" for r in rk2),
                         n_rk2=len(rk2)))

    unified(T, H, HE)

    for r in ROWS:
        print({k: v for k, v in r.items() if k not in ("euler_classes", "rk2_classes", "healthy_classes")})
    save_csv("classifier_sensitivity.csv", ROWS)


if __name__ == "__main__":
    main()
