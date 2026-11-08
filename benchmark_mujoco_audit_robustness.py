"""Robustness of the MuJoCo integrator-order audit.  Paper: Section 6.2.
Outputs: Results/mujoco_audit_shipped_step.csv, Results/mujoco_audit_seeds.csv

(1) Shipped step.  The main audit multiplies the macro step by K until the finest residual clears the float64 floor, so K is
    chosen by the outcome.  Here every environment is audited at K = 1, the native engine timestep, and the label
    distribution is reported as is (undetermined = defect below float64 round-off).
(2) Seeds.  Ant, Humanoid and Hopper with 3 seeds x 24 starts x 20 steps at the K chosen by the main audit; per-seed class counts."""
import warnings, sys; warnings.filterwarnings("ignore")
import benchmark_mujoco_integrator_order as B
import pandas as pd
from results_io import save_csv, n_trials, RESULTS

ENVS = ["Reacher-v5", "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Swimmer-v5", "HalfCheetah-v5",
        "Hopper-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5", "HumanoidStandup-v5", "Pusher-v5"]


def main():
    rows, trials = [], []
    for e in ENVS:
        iname, q, dt0, fs, out = B.audit(e, ntr=n_trials(24, 4), nsteps=n_trials(20, 8), Ks=(1,), trial_rows=trials)
        h, cls, ex, r8 = out[1]
        print(f"{e:<28} K=1 h={h:g} {cls} median(a,b,c)={ex}", flush=True)
        rows.append(dict(environment=e, declared=iname, declared_order=q, macro_h=h, classes=str(cls),
                         median_a=None if ex is None else ex[0], median_b=None if ex is None else ex[1],
                         median_c=None if ex is None else ex[2]))
    save_csv("mujoco_audit_shipped_step.csv", rows)
    save_csv("mujoco_audit_shipped_step_trials.csv", trials)
    rows, trials = [], []
    selected = pd.read_csv(RESULTS / "mujoco_integrator_order_audit.csv").set_index("environment").K
    for e in ["Ant-v5", "Humanoid-v5", "Hopper-v5"]:
        K = int(selected[e])
        for seed in (0, 1, 2):
            iname, q, dt0, fs, out = B.audit(e, ntr=n_trials(24, 4), nsteps=n_trials(20, 8), seed=seed, Ks=(K,), trial_rows=trials)
            h, cls, ex, r8 = out[K]
            print(f"{e:<14} seed {seed} K={K} h={h:g} {cls} median(a,b,c)={ex}", flush=True)
            rows.append(dict(environment=e, seed=seed, K=K, macro_h=h, classes=str(cls),
                             median_a=None if ex is None else ex[0], median_b=None if ex is None else ex[1],
                             median_c=None if ex is None else ex[2]))
    save_csv("mujoco_audit_seeds.csv", rows)
    save_csv("mujoco_audit_seed_trials.csv", trials)


if __name__ == "__main__":
    main()
