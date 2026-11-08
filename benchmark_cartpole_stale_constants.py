"""CartPole: declared order 1 recovered; stale-derived-constants bug detected from the simulator's own
declared parameters; consistent updates pass.  Paper: Section 6.1 (CartPole).
Output: Results/cartpole_stale_constants.csv
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import solve_ivp
import fingerprint as F
from results_io import save_csv, n_trials
from shipped_cartpole import *

ROWS = []

def validate():
    s0 = [0.1, 0.3, 0.4, -0.5]
    sol = solve_ivp(ode, [0, 2], s0, rtol=1e-12, atol=1e-14, method="DOP853", dense_output=True)
    Es = [energy(sol.sol(t), 1.0, 0.1, 0.5) for t in np.linspace(0, 2, 50)]
    print(f"ODE solve: max |E - E0| = {max(abs(e-Es[0]) for e in Es):.2e}  (E0={Es[0]:.4f})")

def run(kind, name, eps, integ="euler", n=100, seed=3, q=1):
    rng = np.random.default_rng(seed)
    decl = set_params(kind, name, eps)
    E = lambda s: energy(s, decl["masscart"], decl["masspole"], decl["length"])
    cls = {}; ex = []
    for _ in range(n):
        s0 = [rng.uniform(-.5,.5), rng.uniform(-.5,.5), rng.uniform(-.3,.3), rng.uniform(-.5,.5)]
        E0, Ee = trial(s0, E, integ)
        r = stats(E0, Ee)
        c = F.guarded_classify(r[1], r[2], r[4], r[8], q, np.mean(np.abs(E0)), H)
        cls[c] = cls.get(c, 0)+1; ex.append(F.exponents(r[1], r[2], r[4], r[8]))
    set_params("nominal", None, 0)
    return cls, np.median(np.array(ex), 0).round(2)


def report(group, label, kind, name, eps, integ="euler", n=100, seed=3):
    cls, med = run(kind, name, eps, integ, n=n, seed=seed)
    print(f"  {label:<24}", cls, med)
    ROWS.append(dict(group=group, kind=kind, attribute=name, eps=eps, integrator=integ, n=n,
                     classes=str(cls), median_a=med[0], median_b=med[1], median_c=med[2]))


def main():
    validate()
    nh, nf = n_trials(200, 20), n_trials(100, 20)
    print("\nhealthy, declared q=1")
    for integ in ("euler", "semi-implicit"):
        report("healthy", integ, "nominal", None, 0, integ, n=nh, seed=1)
    print("\nspec mismatch (consistent update, tested vs NOMINAL spec): masspole")
    for e in [1e-3, 1e-2, 1e-1, 0.5]:
        report("spec_mismatch", f"masspole +{e:g}", "spec", "masspole", e, n=nf)
    print("\nSTALE constants, tested vs the simulator's own declared parameters")
    for name in ("masspole", "length"):
        for e in [1e-3, 1e-2, 1e-1, 0.5]:
            report("stale", f"{name} +{e:g} stale", "stale", name, e, n=nf)
    print("\nconsistent update (control), tested vs declared parameters")
    for name in ("masspole", "length"):
        for e in [1e-1, 0.5]:
            report("updated_control", f"{name} +{e:g} updated", "consistent", name, e, n=nf)
    save_csv("cartpole_stale_constants.csv", ROWS)


if __name__ == "__main__":
    main()
