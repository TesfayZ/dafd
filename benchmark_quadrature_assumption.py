"""Direct numerical check of assumption Q on the heavy-top replica.

Compare the actual RK4 stage-velocity mean with an augmented DOP853 solve
that integrates angular velocity. This is retrospective numerical support,
not a proof or an Ant/Humanoid experiment. Seed 3 and 12 starts match the
local-error study; each start is reused at all four step sizes. Two reference
tolerances expose numerical sensitivity of the reference integral.
"""
import numpy as np
from scipy.integrate import solve_ivp

from benchmark_quaternion_mechanism import Top, build, XMLS, step_mujoco_style, RK4_B
from results_io import n_trials, save_csv


class RecordedStages:
    def __init__(self, top):
        self.top = top
        self.velocities = []

    def wdot(self, q, w):
        self.velocities.append(w.copy())
        return self.top.wdot(q, w)


def reference_mean(top, q, w, h, rtol, atol):
    def rhs(t, y):
        qq = y[:4] / np.linalg.norm(y[:4])
        ww = y[4:7]
        return np.r_[top.qdot(qq, ww), top.wdot(qq, ww), ww]

    sol = solve_ivp(rhs, (0, h), np.r_[q, w, np.zeros(3)],
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[7:, -1] / h


def main():
    top = Top(build(XMLS["heavy top, ball joint"]))
    rng = np.random.default_rng(3)
    starts = []
    for _ in range(n_trials(12, 4)):
        q = rng.normal(size=4)
        starts.append((q / np.linalg.norm(q), rng.uniform(-2, 2, 3)))
    trials, summary = [], []
    for model in ("heavy top (gravity)", "torque-free asymmetric body"):
        if model.startswith("torque"):
            top.g = np.zeros(3)
        previous = None
        for h in (0.05, 0.025, 0.0125, 0.00625):
            errors, sensitivity = [], []
            for start, (q, w) in enumerate(starts):
                stages = RecordedStages(top)
                step_mujoco_style(stages, q, w, h)
                if len(stages.velocities) != 4:
                    raise RuntimeError("Expected four RK4 stages")
                stage_mean = RK4_B @ np.array(stages.velocities)
                coarse = reference_mean(top, q, w, h, 1e-12, 1e-14)
                fine = reference_mean(top, q, w, h, 1e-13, 1e-15)
                error = float(np.linalg.norm(stage_mean - fine))
                delta = float(np.linalg.norm(coarse - fine))
                errors.append(error)
                sensitivity.append(delta)
                trials.append(dict(model=model, h=h, start=start,
                                   initial_state=repr(np.r_[q, w].tolist()),
                                   quadrature_error=error, reference_change=delta))
            median = float(np.median(errors))
            slope = None if previous is None else float(np.log2(previous / median))
            summary.append(dict(model=model, h=h, n=len(starts),
                                error_median=median, error_min=min(errors),
                                error_max=max(errors), halving_slope=slope,
                                reference_change_max=max(sensitivity)))
            print(model, h, "median error", median, "slope", slope)
            previous = median
    save_csv("quadrature_assumption_trials.csv", trials)
    save_csv("quadrature_assumption_summary.csv", summary)


if __name__ == "__main__":
    main()
