"""Same-call invariant fingerprint versus reference-free state self-convergence.

Frozen post-hoc protocol: doc/SELF_CONVERGENCE_PROTOCOL.md.
No healthy trajectory or finer reference is used by either diagnostic.
"""
import json
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import acrobot as A
import fingerprint as F
import shipped_acrobot as I
from results_io import RESULTS, save_csv, n_trials

SEED = 20261104
CASES = (
    ("healthy RK4", {}),
    ("RK4 mass +0.1%", dict(attr="LINK_MASS_2", eps=1e-3)),
    ("RK4 Coriolis omission", dict(nips=True)),
    ("midpoint RK2", dict(integ="rk2")),
    ("Euler", dict(integ="euler")),
    ("Euler mass +1%", dict(integ="euler", attr="LINK_MASS_2", eps=1e-2)),
)


def state_distance(a, b):
    delta = np.array(a) - np.array(b)
    delta[:2] = np.arctan2(np.sin(delta[:2]), np.cos(delta[:2]))
    return float(np.linalg.norm(delta))


def trial(s0, h, nsteps):
    s = s0.copy()
    records = []
    for step in range(nsteps):
        E0 = float(A.energy(s))
        ends = {}
        for m in I.MS:
            x = s.copy()
            for _ in range(m):
                x = I.ship_step(x, h/m)
                if not np.all(np.isfinite(x)):
                    return None, records, "nonfinite"
                if abs(x[2]) >= I.u.MAX_VEL_1 or abs(x[3]) >= I.u.MAX_VEL_2:
                    return None, records, "velocity-clipped"
            ends[m] = x
        records.append(dict(step=step, E0=E0,
                            **{f"r{m}": abs(float(A.energy(ends[m]))-E0)/h for m in I.MS},
                            **{f"d{m}": state_distance(ends[m], ends[2*m]) for m in (1, 2, 4)},
                            state_floor=50*F.EPS_MACH*(1+max(np.linalg.norm(x) for x in ends.values()))))
        s = ends[1]
    means = {key: float(np.mean([r[key] for r in records]))
             for key in ("r1", "r2", "r4", "r8", "d1", "d2", "d4")}
    Em = float(np.mean([abs(r["E0"]) for r in records]))
    means["cls"] = F.guarded_classify(*(means[f"r{m}"] for m in I.MS), 4, Em, h)
    means["E_abs_mean"] = Em
    means["state_floor"] = max(r["state_floor"] for r in records)
    resolved = min(means[f"d{m}"] for m in (1, 2, 4)) > means["state_floor"]
    means["state_status"] = "resolved" if resolved else "undetermined"
    means["p12"] = float(np.log2(means["d1"]/means["d2"])) if resolved else np.nan
    means["p24"] = float(np.log2(means["d2"]/means["d4"])) if resolved else np.nan
    return means, records, "smooth"


def main():
    starts = I.sample_states(np.random.default_rng(SEED), n_trials(40, 6))
    rows, steps = [], []
    for name, kw in CASES:
        with I.case(kw):
            for h in (.2, .05):
                for k, s0 in enumerate(starts):
                    values, records, status = trial(s0, h, n_trials(20, 8))
                    identity = dict(case=name, h=h, start=k, seed=SEED,
                                    initial_state=json.dumps(s0.tolist()))
                    rows.append(dict(**identity, status=status, **(values or {})))
                    steps.extend(dict(**identity, **r) for r in records)
    save_csv("self_convergence_trials.csv", rows)
    save_csv("self_convergence_steps.csv", steps)
    summaries = []
    for (name, h), group in pd.DataFrame(rows).groupby(["case", "h"], sort=False):
        valid = group[group.status == "smooth"]
        resolved = valid[valid.state_status == "resolved"]
        summaries.append(dict(case=name, h=h, n_starts=len(group), n_smooth=len(valid),
                              n_state_resolved=len(resolved), classes=str(valid.cls.value_counts().to_dict()),
                              median_p12=resolved.p12.median(), median_p24=resolved.p24.median(),
                              p10_p24=resolved.p24.quantile(.1), p90_p24=resolved.p24.quantile(.9)))
    save_csv("self_convergence_summary.csv", summaries)
    print(pd.DataFrame(summaries).to_string(index=False))
    root = Path(__file__).resolve().parent
    sources = ["benchmark_self_convergence.py", "doc/SELF_CONVERGENCE_PROTOCOL.md",
               "shipped_acrobot.py", "acrobot.py", "fingerprint.py", "results_io.py"]
    outputs = [RESULTS / f"self_convergence_{suffix}.csv" for suffix in ("trials", "steps", "summary")]
    manifest = dict(protocol="doc/SELF_CONVERGENCE_PROTOCOL.md", post_hoc=True,
                    seed=SEED, starts=len(starts), steps=n_trials(20, 8),
                    macro_steps=[.2, .05], python=platform.python_version(),
                    packages={p: version(p) for p in ("numpy", "pandas", "gymnasium")},
                    source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
                    output_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in outputs})
    (RESULTS / "self_convergence_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
