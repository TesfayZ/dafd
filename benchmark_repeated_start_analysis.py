"""Start-level endpoints for repeated order/step configurations.

Pooled evaluation counts are descriptive. Wilson intervals below apply to a
single binary endpoint per independently sampled start WITHIN one start set
or system; configurations of that start are never treated as new samples.
"""
import numpy as np
import pandas as pd
from results_io import RESULTS, save_csv
from benchmark_finite_step_classifier_validation import wilson

ABSTAIN = {"undetermined", "unstable", "blowup"}


def summarize(group, dataset, population, event):
    keys = ["true_order", "h"]
    if group.duplicated(["cluster_id"] + keys).any():
        raise ValueError("Duplicate start/configuration")
    by_start = group.groupby("cluster_id", sort=True)
    sizes = by_start.size()
    n_configs = group[keys].drop_duplicates().shape[0]
    if not (sizes == n_configs).all():
        raise ValueError("Incomplete repeated-start configuration grid")
    if event == "any_flag":
        outcomes = by_start.cls.apply(lambda x: any(c != "consistent" and c not in ABSTAIN for c in x))
    elif event == "all_model":
        outcomes = by_start.cls.apply(lambda x: all(c == "model" for c in x))
    else:
        raise ValueError(event)
    k, n = int(outcomes.sum()), len(outcomes)
    lo, hi = wilson(k, n)
    abstained = int(group.cls.isin(ABSTAIN).sum())
    return dict(dataset=dataset, population=population, event=event, n_starts=n, n_configs=n_configs,
                successes=k, rate=k/n, wilson_lo=lo, wilson_hi=hi,
                n_evaluations=len(group), n_abstained=abstained,
                coverage=1-abstained/len(group),
                starts_with_abstention=int(by_start.cls.apply(lambda x: any(c in ABSTAIN for c in x)).sum()))


def main():
    transfer = pd.read_csv(RESULTS / "transfer_trials.csv")
    finite = pd.read_csv(RESULTS / "finite_step_trials.csv")
    rows = []
    for dataset, data, population_key in (("transfer", transfer, "system"), ("finite_step", finite, "start_set")):
        for population, group in data.groupby(population_key, sort=True):
            if population == "near_equilibrium":
                continue
            healthy = group[group.eps == 0]
            rows.append(summarize(healthy, dataset, population, "any_flag"))
            faults = group[(group.eps == 1e-2) & (group.true_order >= 2)]
            rows.append(summarize(faults, dataset, population, "all_model"))
    save_csv("repeated_start_summary.csv", rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
