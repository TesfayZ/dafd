"""Shipped Gymnasium CartPole (declared integrator: explicit Euler, order 1), passive (force_mag = 0).

Energy E = 1/2 (M+m) xd^2 + m l xd thd cos(th) + 2/3 m l^2 thd^2 + m g l cos(th) is conserved by the
frictionless model; validated against a 1e-12 ODE solve in benchmark_cartpole_stale_constants.py.
set_params(kind, name, eps): 'nominal' | 'spec' (consistent update, tested vs nominal) |
'stale' (attribute only; derived constants NOT updated) | 'consistent' (all derived updated).
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
import gymnasium.envs.classic_control.cartpole as CP
from scipy.integrate import solve_ivp
import fingerprint as F
from results_io import n_trials

u = CP.CartPoleEnv(); u.reset(seed=0)
G = 9.8
NOM = dict(masscart=1.0, masspole=0.1, length=0.5)
H, T = 0.02, 2.0
N = int(round(T / H)); MS = (1, 2, 4, 8)

def energy(s, mc, mp, l):
    x, xd, th, thd = s
    c = np.cos(th)
    return (0.5*(mc+mp)*xd**2 + mp*l*xd*thd*c + 0.5*(4/3)*mp*l**2*thd**2 + mp*G*l*c)

def ode(t, s, mc=1.0, mp=0.1, l=0.5):
    x, xd, th, thd = s; tm = mc+mp; pml = mp*l
    c, sn = np.cos(th), np.sin(th)
    temp = (pml*thd**2*sn)/tm
    tha = (G*sn - c*temp)/(l*(4/3 - mp*c**2/tm))
    xa = temp - pml*tha*c/tm
    return [xd, xa, thd, tha]


def set_params(kind, name, eps):
    for k, v in NOM.items(): setattr(u, k, v)
    u.total_mass = u.masscart + u.masspole; u.polemass_length = u.masspole * u.length
    if kind == "nominal":
        return dict(NOM)
    new = NOM[name]*(1+eps)
    setattr(u, name, new)
    if kind in ("spec", "consistent"):          # derived constants updated
        u.total_mass = u.masscart + u.masspole; u.polemass_length = u.masspole * u.length
    declared = dict(NOM); declared[name] = new
    return declared if kind in ("stale", "consistent") else dict(NOM)

def step(s, h, integ):
    u.state = np.array(s, float); u.tau = h; u.kinematics_integrator = integ
    u.steps_beyond_terminated = None; u.force_mag = 0.0
    u.step(1)
    return np.array(u.state, float)

def trial(s0, E, integ):
    s = np.array(s0, float); E0 = np.empty(N); Ee = {m: np.empty(N) for m in MS}
    for k in range(N):
        E0[k] = E(s)
        for m in MS:
            x = s
            for _ in range(m): x = step(x, H/m, integ)
            Ee[m][k] = E(x)
            if m == 1: nxt = x
        s = nxt
    return E0, Ee

def stats(E0, Ee):
    return {m: float(np.mean(np.abs((Ee[m]-E0)/H))) for m in MS}

