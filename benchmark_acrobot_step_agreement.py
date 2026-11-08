"""One-step maximum component difference, shipped versus analytic RK4.

Fixed example plus 1000 uniform starts on [-1.5,1.5]^4, seed 0, h=0.2.
No angle wrapping or velocity clipping occurs in this start population.
"""
import numpy as np
import acrobot as A
import shipped_acrobot as S
from results_io import save_csv


def main():
    starts = [np.array([.1, -.2, .3, -.15])]
    starts.extend(np.random.default_rng(0).uniform(-1.5, 1.5, (1000, 4)))
    rows = []
    with S.fault():
        for i, x in enumerate(starts):
            diff = np.max(np.abs(S.ship_step(x, .2) - A.rk4_step(x, 0, .2)))
            rows.append(dict(population="example" if i == 0 else "random",
                             start=i-1, h=.2, max_abs_difference=diff,
                             initial_state=repr(x.tolist())))
    save_csv("acrobot_step_agreement.csv", rows)


if __name__ == "__main__":
    main()
