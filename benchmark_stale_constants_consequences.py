"""Consequence of the CartPole stale-constants bug, and a no-op attribute scan.
Paper: Section 6.1.  Outputs: Results/stale_constants_divergence.csv, Results/attribute_sensitivity.csv"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import shipped_classic_control as L
from results_io import save_csv, n_trials
DIV, SENS = [], []
uc = L.uc
def run_traj(attr, f, mode, s0, T=1.0, h=0.02):
    L.reset(uc, L.C_NOM); setattr(uc, attr, L.C_NOM[attr]*f)
    if mode == "updated":
        uc.total_mass = uc.masscart+uc.masspole; uc.polemass_length = uc.masspole*uc.length
    s = np.array(s0, float); out = [s]
    for _ in range(int(T/h)):
        s = L.c_step(s, h); out.append(s)
    L.reset(uc, L.C_NOM); return np.array(out)
rng = np.random.default_rng(5)
starts = [L.c_samp(rng) for _ in range(n_trials(40, 8))]
print("CartPole: divergence after 1 s between the stale (direct) env and the intended (updated) env")
print(f"{'attr':<10}{'factor':>7}  {'median |dtheta| (rad)':>22} {'median |dx| (m)':>16}  max |dtheta|")
for attr in ("masscart", "masspole", "length"):
    for f in (0.5, 1.5, 2.0):
        dth, dx = [], []
        for s0 in starts:
            a = run_traj(attr, f, "direct", s0); b = run_traj(attr, f, "updated", s0)
            dth.append(np.max(np.abs(a[:,2]-b[:,2]))); dx.append(np.max(np.abs(a[:,0]-b[:,0])))
        print(f"{attr:<10}{f:>7}  {np.median(dth):>22.3e} {np.median(dx):>16.3e}  {np.max(dth):.3e}")
        DIV.append(dict(attribute=attr, factor=f, n_starts=len(starts), horizon_s=1.0, median_max_dtheta_rad=float(np.median(dth)), median_max_dx_m=float(np.median(dx)), max_dtheta_rad=float(np.max(dth))))
# no-op scan: does changing the attribute change the one-step map at all?
print("\nOne-step sensitivity (max |step(attr x2) - step(nominal)| over 20 states; 0 => attribute is a no-op)")
def sens(env, nom, attr, step, samp, n=20):
    rng = np.random.default_rng(1); m = 0.0
    for _ in range(n):
        s0 = np.array(samp(rng), float)
        L.reset(env, nom); a = step(s0, 0.02 if env is uc else (0.2 if env is L.ua else 0.05))
        setattr(env, attr, nom[attr]*2.0); b = step(s0, 0.02 if env is uc else (0.2 if env is L.ua else 0.05))
        m = max(m, float(np.max(np.abs(a-b)))); L.reset(env, nom)
    return m
for env, nom, step, samp, nm in [(L.ua, L.A_NOM, L.a_step, L.a_samp, "Acrobot"), (uc, L.C_NOM, L.c_step, L.c_samp, "CartPole"), (L.up, L.P_NOM, L.p_step, L.p_samp, "Pendulum")]:
    for attr in nom:
        v = sens(env, nom, attr, step, samp)
        print(f"  {nm:<9}{attr:<16}{v:.3e}")
        SENS.append(dict(environment=nm, attribute=attr, factor=2.0, max_step_change=v, no_op=(v == 0.0)))

save_csv("stale_constants_divergence.csv", DIV)
save_csv("attribute_sensitivity.csv", SENS)
