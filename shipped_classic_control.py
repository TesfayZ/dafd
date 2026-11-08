"""Shared helpers for the domain-randomisation audits: Gymnasium Acrobot, CartPole, Pendulum.

Each environment exposes step(s,h), an energy built from the environment's OWN declared attribute
values, and a start-state sampler.  cell() tallies fingerprint classes over random starts.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import gymnasium.envs.classic_control.acrobot as GA
import gymnasium.envs.classic_control.cartpole as CP
import gymnasium.envs.classic_control.pendulum as PD
import fingerprint as F
from results_io import n_trials
MS = (1, 2, 4, 8)

def cell(step, energy, sampler, h, n, q, ntr, seed):
    rng = np.random.default_rng(seed); cls = {}
    for _ in range(ntr):
        s0 = sampler(rng); s = np.array(s0, float)
        E0 = np.empty(n); Ee = {m: np.empty(n) for m in MS}
        for k in range(n):
            E0[k] = energy(s)
            for m in MS:
                x = s
                for _ in range(m): x = step(x, h/m)
                Ee[m][k] = energy(x)
                if m == 1: nxt = x
            s = nxt
        r = {m: float(np.mean(np.abs((Ee[m]-E0)/h))) for m in MS}
        c = (F.guarded_classify(r[1], r[2], r[4], r[8], q, np.mean(np.abs(E0)), h)     # frozen round-off guard
             if np.all(np.isfinite(list(r.values()))) else "blowup")
        cls[c] = cls.get(c, 0)+1
    return cls

def short(c):
    k = max(c, key=c.get); return f"{k} {c[k]}/{sum(c.values())}" + ("" if len(c)==1 else " (+"+",".join(f"{a}:{b}" for a,b in c.items() if a!=k)+")")

# ---------------- Acrobot
ua = GA.AcrobotEnv(); ua.reset(seed=0)
A_NOM = dict(LINK_LENGTH_1=1.0, LINK_LENGTH_2=1.0, LINK_MASS_1=1.0, LINK_MASS_2=1.0,
             LINK_COM_POS_1=0.5, LINK_COM_POS_2=0.5, LINK_MOI=1.0)
def a_step(s, h):
    ua.state = np.array(s, float); ua.dt = h; ua.step(1); return np.array(ua.state, float)
def a_E(s):
    q1, q2, d1, d2 = s; u = ua; I = u.LINK_MOI
    c2 = np.cos(q2)
    M11 = u.LINK_MASS_1*u.LINK_COM_POS_1**2 + u.LINK_MASS_2*(u.LINK_LENGTH_1**2+u.LINK_COM_POS_2**2+2*u.LINK_LENGTH_1*u.LINK_COM_POS_2*c2) + 2*I
    M12 = u.LINK_MASS_2*(u.LINK_COM_POS_2**2+u.LINK_LENGTH_1*u.LINK_COM_POS_2*c2) + I
    M22 = u.LINK_MASS_2*u.LINK_COM_POS_2**2 + I
    T = 0.5*(M11*d1**2 + 2*M12*d1*d2 + M22*d2**2)
    V = -(u.LINK_MASS_1*u.LINK_COM_POS_1 + u.LINK_MASS_2*u.LINK_LENGTH_1)*9.8*np.cos(q1) - u.LINK_MASS_2*u.LINK_COM_POS_2*9.8*np.cos(q1+q2)
    return T+V
def a_samp(rng): return rng.uniform(-1.5, 1.5, 4)

# ---------------- CartPole
uc = CP.CartPoleEnv(); uc.reset(seed=0)
C_NOM = dict(masscart=1.0, masspole=0.1, length=0.5, gravity=9.8)
def c_step(s, h):
    uc.state = np.array(s, float); uc.tau = h; uc.kinematics_integrator = "euler"
    uc.steps_beyond_terminated = None; uc.force_mag = 0.0; uc.step(1); return np.array(uc.state, float)
def c_E(s):
    x, xd, th, thd = s; mc, mp, l, g = uc.masscart, uc.masspole, uc.length, uc.gravity; c = np.cos(th)
    return 0.5*(mc+mp)*xd**2 + mp*l*xd*thd*c + (2/3)*mp*l**2*thd**2 + mp*g*l*c
def c_samp(rng): return [rng.uniform(-.5,.5), rng.uniform(-.5,.5), rng.uniform(-.3,.3), rng.uniform(-.5,.5)]

# ---------------- Pendulum
up = PD.PendulumEnv(); up.reset(seed=0)
P_NOM = dict(g=10.0, l=1.0, m=1.0)
def p_step(s, h):
    up.state = np.array(s, float); up.dt = h; up.step(np.array([0.0])); return np.array(up.state, float)
def p_E(s):
    th, thd = s; return 0.5*thd**2 + 1.5*up.g/up.l*np.cos(th)
def p_samp(rng): return [np.pi + rng.uniform(-1, 1), rng.uniform(-1, 1)]

def reset(env, nom):
    for k, v in nom.items(): setattr(env, k, v)
    if env is uc: uc.total_mass = uc.masscart + uc.masspole; uc.polemass_length = uc.masspole*uc.length
