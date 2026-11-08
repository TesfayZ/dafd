"""Refinement-order fingerprint: residuals, exponents, classifier, order-adapted weights.

r^(m) = [E_ref(Psi_{h/m}^m x) - E_ref(x)]/h.  For a declared integrator order q,
    r^(m) = D + m^-q a h^q + (alpha1 m^-q + alpha2 m^-(q+1)) h^(q+1) + ...
"""
import numpy as np


def richardson_weights(ms, q, n_kill):
    """Weights w with sum_m w_m = 1 (keeps D) and sum_m w_m m^-(q+j) = 0, j=0..n_kill-1.
    len(ms) must equal n_kill+1. For ms=(1,2,4), q=4, n_kill=2,
    w = (1, -48, 512)/465 and R2 = -15 * sum(w*r)."""
    ms = np.asarray(ms, float)
    assert len(ms) == n_kill + 1
    A = np.vstack([np.ones(len(ms))] + [ms ** (-(q + j)) for j in range(n_kill)])
    b = np.zeros(len(ms)); b[0] = 1.0
    return np.linalg.solve(A, b)


def exponents(r1, r2, r4, r8):
    """consecutive exponents a=1->2, b=2->4, c=4->8 from mean|r^(m)|."""
    return (np.log2(r1 / r2), np.log2(r2 / r4), np.log2(r4 / r8))


EPS_MACH = 2.220446049250313e-16
ROUNDOFF_FACTOR = 50.0


def roundoff_floor(E_mean_abs, h):
    """Float64 round-off floor of one residual, 50 * eps_mach * (|E| + 1) / h (Definition 1, guard phi_round)."""
    return ROUNDOFF_FACTOR * EPS_MACH * (float(E_mean_abs) + 1.0) / h


def classify(r1, r2, r4, r8, q_decl=4, floor=0.0, e_scale=None, *,
             c_thr=0.5, delta=0.4, theta=None, order_rule="mean"):
    """Class from the pattern of consecutive exponents (a,b,c).
    ...
    Returns "blowup" if any residual is non-finite (the step map produced a
    non-finite state); "undetermined" if the finest residual is at/below the
    noise floor (or any residual is zero); "unstable" if e_scale is given and the finest residual exceeds
    50% of it (an abstention, not a fault); otherwise one of the classified labels.
    """
    if not np.all(np.isfinite([r1, r2, r4, r8])):
        return "blowup"
    if r8 <= floor or min(r1, r2, r4, r8) <= 0:
        return "undetermined"
    if e_scale is not None and r8 > 0.5 * e_scale:     # Excessive residual rate per unit model time (see guarded_classify)
        return "unstable"
    a, b, c = exponents(r1, r2, r4, r8)
    thr = q_decl - (1.0 if q_decl >= 3 else 0.5) if theta is None else theta
    if min(a, b, c) >= thr:
        return "consistent"
    if c < c_thr or (c < min(a, b) - delta and c < thr):
        return "model"
    k = (a + b + c) / 3 if order_rule == "mean" else c
    return f"order-{int(round(k))}" if k > 0.5 else "model"


def guarded_classify(r1, r2, r4, r8, q_decl, E_mean_abs, h, *,
                     noise_floor=0.0, roundoff_factor=ROUNDOFF_FACTOR, **rule):
    """Complete benchmark rule: round-off AND noise floors plus rate guard.

    E_mean_abs is the mean absolute start energy on the evaluated steps.
    The instability guard retains the original transfer protocol: a residual
    rate above 0.5*(E_mean_abs+1) per unit model time causes abstention.
    Energies and times must use the benchmark's declared units. It is a
    heuristic rate guard, not a unit-invariant bound on per-step energy change.
    Use classify() directly only for explicitly unguarded controls.
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if not np.isfinite(noise_floor) or noise_floor < 0:
        raise ValueError("noise_floor must be finite and nonnegative")
    if not np.isfinite(roundoff_factor) or roundoff_factor <= 0:
        raise ValueError("roundoff_factor must be finite and positive")
    if not np.all(np.isfinite([r1, r2, r4, r8, E_mean_abs])):
        return "blowup"
    if E_mean_abs < 0:
        raise ValueError("E_mean_abs must be nonnegative")
    floor = max(roundoff_factor * EPS_MACH * (E_mean_abs + 1) / h, noise_floor)
    return classify(r1, r2, r4, r8, q_decl, floor, E_mean_abs + 1, **rule)
