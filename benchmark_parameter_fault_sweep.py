"""
Second-link-mass fault sweep (m2 -> m2 + eps) with step-size study: raw residual, two-level R2, static baseline.
Paper: Appendix D.  Outputs: Results/parameter_fault_sweep.csv, Results/parameter_fault_sweep_summary.csv

Static-baseline threshold added so that the paper's claim
"corrected threshold is >1e5 below static baseline at h=0.2"
is script-verifiable.

Derivation (used in the paper, Appendix D):
    r_h = d(eps) + c4 h^4 + c5 h^5 + O(h^6),   d(0) = 0.
    R_1(h) = r_h - 16 r_{h/2} = -15 d + (c5/2) h^5 + O(h^6).
    R_2(h) = (32 R_1(h/2) - R_1(h)) / 31
           = -15 d(eps) + O(h^6).
The amplification column |R_2|/|r_h| should converge to 15 at large
eps (fault-dominated) and drop below 1 at small eps (truncation-
dominated). R_2 is a detection statistic, not an unbiased estimator
of d(eps).
"""

import numpy as np
from results_io import save_csv

from acrobot_parametric import (
    energy, rk4_step, M2_REF, consistency_check,
)

STATE0 = np.array([0.1, -0.2, 0.3, -0.15])
T = 10.0
DETECTION_RATIO = 10.0

H_VALUES = [0.2, 0.1, 0.05, 0.025]

# Added 0.15 and 0.2 so the h=0.2 static threshold is bracketed at 0.2
# instead of reported as the overshot value 0.3.
EPS_VALUES = [
    0.0,
    1e-12, 3e-12, 1e-11, 3e-11, 1e-10, 3e-10,
    1e-9,  3e-9,  1e-8,  3e-8,  1e-7,  3e-7,
    1e-6,  3e-6,  1e-5,  3e-5,  1e-4,  3e-4,
    1e-3,  3e-3,  1e-2,  3e-2,  0.1,   0.15,  0.2,   0.3,
]


def residual_local(state, h, m2_model, m2_ref=M2_REF):
    s1 = rk4_step(state, 0.0, h, m2_model)
    if not np.all(np.isfinite(s1)):
        return np.nan, None
    E0 = energy(state, m2=m2_ref)
    E1 = energy(s1,   m2=m2_ref)
    return (E1 - E0) / h, s1


def richardson_2lvl(state, h, m2_model, m2_ref=M2_REF):
    r_h, _ = residual_local(state, h, m2_model, m2_ref)
    if not np.isfinite(r_h):
        return np.nan
    s1 = rk4_step(state, 0.0, h / 2, m2_model)
    s2 = rk4_step(s1,    0.0, h / 2, m2_model) if s1 is not None else None
    if s2 is None or not (np.all(np.isfinite(s1)) and np.all(np.isfinite(s2))):
        return np.nan
    E0 = energy(state, m2=m2_ref)
    E2 = energy(s2,   m2=m2_ref)
    r_h2 = (E2 - E0) / h
    hs4 = h / 4
    s = state
    for _ in range(4):
        s = rk4_step(s, 0.0, hs4, m2_model)
        if not np.all(np.isfinite(s)):
            return np.nan
    E4 = energy(s, m2=m2_ref)
    r_h4 = (E4 - E0) / h
    p = 4
    R1_h  = r_h  - (2.0 ** p) * r_h2
    R1_h2 = r_h2 - (2.0 ** p) * r_h4
    return ((2.0 ** (p + 1)) * R1_h2 - R1_h) / (2.0 ** (p + 1) - 1.0)


def run(m2_model, h, n_steps):
    state = STATE0.copy()
    r_unc, r_cor, Es = [], [], []
    for _ in range(n_steps):
        r1, _ = residual_local(state, h, m2_model)
        r2    = richardson_2lvl(state, h, m2_model)
        r_unc.append(r1)
        r_cor.append(r2)
        state = rk4_step(state, 0.0, h, m2_model)
        if not np.all(np.isfinite(state)):
            raise RuntimeError("Simulation became non-finite.")
        Es.append(energy(state, m2=M2_REF))
    return {
        "mean_abs_raw":       float(np.nanmean(np.abs(r_unc))),
        "mean_abs_corrected": float(np.nanmean(np.abs(r_cor))),
        "energy_std":         float(np.std(Es)),
        "final_energy_drift": float(abs(Es[-1] - energy(STATE0, m2=M2_REF))),
    }


def threshold_from_rows(rows, key, thr=DETECTION_RATIO):
    """Smallest tested epsilon such that it and every larger tested epsilon is detected
    (the monotone-floor rule used throughout the paper)."""
    eps_pos = sorted(r["epsilon"] for r in rows if r["epsilon"] > 0)
    if not eps_pos:
        return None, None
    floor = None
    for e in eps_pos:
        r = next(r for r in rows if r["epsilon"] == e)
        if r[key] > thr:
            if floor is None:
                floor = e
        else:
            floor = None
    if floor is None:
        return None, None
    status = "at_floor" if floor == eps_pos[0] else "bracketed"
    return floor, status


def sweep(h):
    n_steps = int(round(T / h))
    healthy = run(M2_REF, h, n_steps)

    raw_bl    = healthy["mean_abs_raw"]
    cor_bl    = healthy["mean_abs_corrected"]
    static_bl = healthy["energy_std"]

    for name, bl in (("raw", raw_bl), ("corrected", cor_bl), ("static", static_bl)):
        if not np.isfinite(bl) or bl <= 0:
            raise RuntimeError(f"degenerate {name} baseline at h={h}: {bl!r}")

    rows = []
    for eps in EPS_VALUES:
        res = run(M2_REF + eps, h, n_steps)
        rows.append({
            "h": h,
            "epsilon": eps,
            "m2": M2_REF + eps,
            **res,
            "raw_ratio":       res["mean_abs_raw"]       / raw_bl,
            "corrected_ratio": res["mean_abs_corrected"] / cor_bl,
            "static_ratio":    res["energy_std"]         / static_bl,
            "amplification":   res["mean_abs_corrected"] /
                               res["mean_abs_raw"] if res["mean_abs_raw"] > 0
                               else float("nan"),
        })
    return healthy, raw_bl, cor_bl, static_bl, rows


def main():
    print("=" * 104)
    print("m2 fault sweep, step-size study, static baseline")
    print("=" * 104)
    try:
        err = consistency_check()
        print(f"Model consistency error: {err:.3e}")
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Consistency check skipped: {exc}")
    print()

    all_rows = []
    summary  = []

    for h in H_VALUES:
        print(f"--- h = {h:.4f}  ({int(round(T / h))} steps over {T:g} s) ---")
        healthy, raw_bl, cor_bl, static_bl, rows = sweep(h)
        print(f"  baselines:  raw = {raw_bl:.4e}   "
              f"corrected = {cor_bl:.4e}   static = {static_bl:.4e}")
        print()
        print(f"  {'eps':>9}  {'raw/bl':>10}  {'cor/bl':>10}  "
              f"{'static/bl':>11}  {'amp':>9}")
        for r in rows:
            marker = "  *" if (r["epsilon"] == 0.0) else ""
            print(f"  {r['epsilon']:>9.1e}  "
                  f"{r['raw_ratio']:>10.2e}  "
                  f"{r['corrected_ratio']:>10.2e}  "
                  f"{r['static_ratio']:>11.2e}  "
                  f"{r['amplification']:>9.2e}{marker}")
            all_rows.append(r)
        print()

        t_raw, s_raw = threshold_from_rows(rows, "raw_ratio")
        t_cor, s_cor = threshold_from_rows(rows, "corrected_ratio")
        t_st,  s_st  = threshold_from_rows(rows, "static_ratio")
        print(f"  thresholds (ratio > {DETECTION_RATIO:g}):")
        print(f"    raw        eps = {t_raw}   [{s_raw}]")
        print(f"    corrected  eps = {t_cor}   [{s_cor}]")
        print(f"    static     eps = {t_st}    [{s_st}]")
        print()
        summary.append({
            "h":             h,
            "healthy_raw":   raw_bl,
            "healthy_corr":  cor_bl,
            "healthy_static": static_bl,
            "thr_raw":       t_raw, "thr_raw_st":  s_raw,
            "thr_corr":      t_cor, "thr_corr_st": s_cor,
            "thr_static":    t_st,  "thr_static_st": s_st,
        })

    print("=" * 104)
    print("Summary: thresholds vs h  (None = not reached in sweep)")
    print("=" * 104)
    print(f"  {'h':>6}  {'healthy_raw':>13}  {'healthy_corr':>13}  "
          f"{'healthy_static':>15}  "
          f"{'thr_raw':>10}  {'thr_corr':>10}  {'thr_static':>11}")
    for s in summary:
        print(f"  {s['h']:>6.4f}  "
              f"{s['healthy_raw']:>13.3e}  "
              f"{s['healthy_corr']:>13.3e}  "
              f"{s['healthy_static']:>15.3e}  "
              f"{str(s['thr_raw']):>10}  "
              f"{str(s['thr_corr']):>10}  "
              f"{str(s['thr_static']):>11}")
    print()

    print("Scaling of the healthy baselines (successive ratios):")
    print(f"  {'h pair':>14}  {'raw ratio':>11}  {'raw p':>7}  "
          f"{'corr ratio':>12}  {'corr p':>8}")
    for i in range(1, len(summary)):
        h_prev, h_curr = summary[i - 1]["h"], summary[i]["h"]
        r_raw = summary[i - 1]["healthy_raw"]  / summary[i]["healthy_raw"]
        r_cor = summary[i - 1]["healthy_corr"] / summary[i]["healthy_corr"]
        p_raw = np.log(r_raw) / np.log(h_prev / h_curr)
        p_cor = np.log(r_cor) / np.log(h_prev / h_curr)
        print(f"  {h_prev:.4f}->{h_curr:.4f}  "
              f"{r_raw:>11.2e}  {p_raw:>7.2f}  "
              f"{r_cor:>12.2e}  {p_cor:>8.2f}")
    print()

    print("Amplification convergence (large eps):")
    print(f"  {'h':>6}  {'amp at eps=3e-2':>16}  {'amp at eps=3e-1':>16}")
    for h in H_VALUES:
        amps = {r["epsilon"]: r["amplification"]
                for r in all_rows if r["h"] == h}
        print(f"  {h:>6.4f}  {amps.get(3e-2, float('nan')):>16.4f}  "
              f"{amps.get(0.3, float('nan')):>16.4f}")
    print()

    save_csv("parameter_fault_sweep.csv", all_rows)
    save_csv("parameter_fault_sweep_summary.csv", summary)
    print()
    print("Notes:")
    print("  *  marks the healthy row; its ratios are 1.0 by construction.")
    print("  amp = |R_2|/|r_h|; converges to 15 at large eps.")
    print("  static baseline = std(E_ref) over the trajectory (a simplified")
    print("    instance of the canonical-invariant baseline of Gebrekidan and Gebrerufael (2026)).")
    print("  h=0.025 corrected baseline is at double-precision round-off;")
    print("    its threshold is round-off limited, not truncation limited.")


if __name__ == "__main__":
    main()