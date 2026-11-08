"""Domain-randomisation consistency audit of Gymnasium classic-control environments.

For every randomisable attribute and factor in {0.5, 1.0, 1.5, 2.0}: assign it the way practitioners do
(env.unwrapped.attr = value) and test whether the simulator still conserves the energy built from its OWN
declared attribute values.  No healthy reference run, no external spec.  CartPole also gets the
'cached constants updated' variant.  Paper: Section 6.1.
Output: Results/domain_randomisation_audit.csv
"""
from shipped_classic_control import *
from results_io import save_csv

FACTORS = (0.5, 1.0, 1.5, 2.0)


def main():
    rows = []

    def run(env_name, env, nom, attr, f, mode, step, E, samp, h, n, q, ntr):
        reset(env, nom)
        setattr(env, attr, nom[attr] * f)
        if mode == "updated" and env is uc:
            uc.total_mass = uc.masscart + uc.masspole
            uc.polemass_length = uc.masspole * uc.length
        c = cell(step, E, samp, h, n, q, ntr, seed=17)
        reset(env, nom)
        rows.append(dict(environment=env_name, attribute=attr, update=mode, factor=f,
                         declared_order=q, n_starts=ntr, classes=str(c),
                         dominant_class=max(c, key=c.get), dominant_count=c[max(c, key=c.get)]))
        return c

    print(f"{'env':<9} {'attribute':<15} {'update':<8} " + "  ".join(f"x{f:<22}" for f in FACTORS), flush=True)
    for attr in A_NOM:
        out = [short(run("Acrobot", ua, A_NOM, attr, f, "direct", a_step, a_E, a_samp, 0.2, 50, 4, n_trials(24, 6))) for f in FACTORS]
        print(f"{'Acrobot':<9} {attr:<15} {'direct':<8} " + "  ".join(f"{o:<23}" for o in out), flush=True)
    for attr in C_NOM:
        for mode in ("direct", "updated"):
            out = [short(run("CartPole", uc, C_NOM, attr, f, mode, c_step, c_E, c_samp, 0.02, 100, 1, n_trials(40, 8))) for f in FACTORS]
            print(f"{'CartPole':<9} {attr:<15} {mode:<8} " + "  ".join(f"{o:<23}" for o in out), flush=True)
    for attr in P_NOM:
        out = [short(run("Pendulum", up, P_NOM, attr, f, "direct", p_step, p_E, p_samp, 0.05, 40, 1, n_trials(40, 8))) for f in FACTORS]
        print(f"{'Pendulum':<9} {attr:<15} {'direct':<8} " + "  ".join(f"{o:<23}" for o in out), flush=True)
    save_csv("domain_randomisation_audit.csv", rows)


if __name__ == "__main__":
    main()
