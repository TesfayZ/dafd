"""Shipped Gymnasium AcrobotEnv: fault injection and refinement-residual trials.

All faults are injected into the real environment object and observed through env.step only:
  parameter attributes (LINK_*), patched integrators (Euler, midpoint RK2), the 'nips' variant.
LINK_LENGTH_2 is never read by AcrobotEnv._dsdt, so it is a null fault and is excluded.
"""
import contextlib
import numpy as np
import gymnasium.envs.classic_control.acrobot as GA
import acrobot as A
from results_io import n_trials

H, T = 0.2, 10.0
N = int(round(T / H))
MS = (1, 2, 4, 8)
NOMINAL = dict(LINK_MASS_1=1.0, LINK_MASS_2=1.0, LINK_LENGTH_1=1.0,
               LINK_COM_POS_1=0.5, LINK_COM_POS_2=0.5, LINK_MOI=1.0)
EPS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
N_HEALTHY, N_FAULT = n_trials(200, 20), n_trials(40, 8)

u = GA.AcrobotEnv()
u.reset(seed=0)
_rk4 = GA.rk4


def euler_patch(derivs, y0, t):
    dt = t[-1] - t[0]
    y = np.asarray(y0, float)
    return (y + dt * np.asarray(derivs(y)))[:4]


def rk2_patch(derivs, y0, t):
    dt = t[-1] - t[0]
    y = np.asarray(y0, float)
    k1 = np.asarray(derivs(y))
    k2 = np.asarray(derivs(y + 0.5 * dt * k1))
    return (y + dt * k2)[:4]


@contextlib.contextmanager
def fault(kind=None, attr=None, eps=0.0, integ=None):
    for k, v in NOMINAL.items():
        setattr(u, k, v)
    if attr is not None:
        setattr(u, attr, NOMINAL[attr] * (1.0 + eps))
    GA.rk4 = {"euler": euler_patch, "rk2": rk2_patch}.get(integ, _rk4)
    try:
        yield
    finally:
        for k, v in NOMINAL.items():
            setattr(u, k, v)
        GA.rk4 = _rk4


def ship_step(s, h):
    u.state = np.array(s, dtype=float)
    u.dt = h
    u.step(1)
    return np.array(u.state, dtype=float)


def run_trial(s0):
    s = s0.copy()
    E0 = np.empty(N)
    Ee = {m: np.empty(N) for m in MS}
    for k in range(N):
        E0[k] = A.energy(s)
        nxt = None
        for m in MS:
            x = s
            for _ in range(m):
                x = ship_step(x, H / m)
            Ee[m][k] = A.energy(x)
            if m == 1:
                nxt = x
        s = nxt
    return E0, Ee


def stats(E0, Ee, sigma=0.0, rng=None):
    if sigma > 0:
        E0 = E0 + sigma * rng.standard_normal(E0.shape)
        Ee = {m: v + sigma * rng.standard_normal(v.shape) for m, v in Ee.items()}
    if not (np.all(np.isfinite(E0)) and all(np.all(np.isfinite(v)) for v in Ee.values())):
        return dict(drift=np.inf, raw=np.inf, sub8=np.inf, R2=np.inf,
                    r1=np.inf, r2=np.inf, r4=np.inf, r8=np.inf)
    r = {m: (Ee[m] - E0) / H for m in MS}
    R2 = (-r[1] + 48 * r[2] - 512 * r[4]) / 31.0
    d = dict(drift=float(np.std(E0)), raw=float(np.mean(np.abs(r[1]))),
             sub8=float(np.mean(np.abs(r[8]))), R2=float(np.mean(np.abs(R2))))
    d["E_abs_mean"] = float(np.mean(np.abs(E0)))
    for m in MS:
        d[f"r{m}"] = float(np.mean(np.abs(r[m])))
    return d


def exps(d, floor=0.0):
    """refinement exponents; None when the finest residual is below `floor`."""
    if d["r8"] <= max(floor, 0.0) or d["r1"] == np.inf or d["r8"] == 0:
        return None
    p18 = np.log2(d["r1"] / d["r8"]) / 3.0
    p48 = np.log2(d["r4"] / d["r8"])
    p24 = np.log2(d["r2"] / d["r4"]) if d["r4"] > 0 else np.nan
    return p18, p24, p48


def classify_p18_legacy(d, floor=0.0):
    e = exps(d, floor)
    if e is None:
        return "undetermined"
    p = e[0]
    if p >= 3.0:
        return "consistent"
    if p >= 1.6:
        return "order-2"
    if p >= 0.6:
        return "order-1"
    return "model"


def sample_states(rng, n):
    return np.column_stack([rng.uniform(-1.5, 1.5, n), rng.uniform(-1.5, 1.5, n),
                            rng.uniform(-1.5, 1.5, n), rng.uniform(-1.5, 1.5, n)])


def collect(kw, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    with fault(**kw):
        for s0 in sample_states(rng, n):
            out.append(run_trial(s0))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - w), min(1, c + w))



@contextlib.contextmanager
def case(kw):
    """fault(...) plus the optional nips switch (kw may contain nips=True)."""
    kw = dict(kw)
    nips = kw.pop("nips", False)
    with fault(**kw):
        u.book_or_nips = "nips" if nips else "book"
        try:
            yield
        finally:
            u.book_or_nips = "book"
