#!/usr/bin/env python3
"""Run the benchmarks and write Results/*.csv (+ Results/logs/*.log).

    python run_all_experiments.py                 # everything (runtime depends on environment; see README)
    python run_all_experiments.py --quick         # reduced-size smoke run -> Results/quick/ (about 3 min)
    python run_all_experiments.py --only mujoco   # scripts whose name contains 'mujoco'
    python run_all_experiments.py --skip-slow     # skip scripts estimated above 5 minutes
    python run_all_experiments.py --jobs 1        # sequential trial-file dependencies
    python run_all_experiments.py --resume        # skip scripts already completed (marker files in Results/logs/)
Scripts run sequentially in dependency order. The head-to-head benchmarks need the comparison repository in external/ (see README).
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# (script, approximate minutes, needs comparison implementation, needs gymnasium/mujoco)
SCRIPTS = [
    ("benchmark_acrobot_step_agreement.py", 0, False, False),
    ("benchmark_self_convergence.py", 1, False, True),
    # exact layer and asymptotics
    ("benchmark_symbolic_lie_derivative.py", 1, False, False),
    ("benchmark_rk4_defect_high_precision.py", 1, False, False),
    ("benchmark_signed_defect_and_r2_asymptotics.py", 1, False, False),
    # calibration-free classifier on the shipped Acrobot, finite-step and transfer validation, sensitivity
    ("benchmark_shipped_acrobot_fingerprint.py", 4, False, True),
    ("benchmark_shipped_acrobot_calibrated_detectors.py", 5, False, True),
    ("benchmark_false_alarm_floor_noise_guard.py", 6, False, True),
    ("benchmark_finite_step_classifier_validation.py", 8, False, False),
    ("benchmark_transfer_validation.py", 12, False, False),
    ("benchmark_repeated_start_analysis.py", 1, False, False),
    ("benchmark_classifier_sensitivity.py", 2, False, False),   # needs the two trial files above
    ("benchmark_noise_sources_operating_envelope.py", 6, False, False),
    # audits that need no healthy reference run
    ("benchmark_cartpole_stale_constants.py", 2, False, True),
    ("benchmark_domain_randomisation.py", 3, False, True),
    ("benchmark_stale_constants_consequences.py", 1, False, True),
    ("benchmark_mujoco_integrator_order.py", 3, False, True),
    ("benchmark_mujoco_audit_robustness.py", 6, False, True),
    ("benchmark_quaternion_joint_order_control.py", 1, False, True),
    ("benchmark_quaternion_mechanism.py", 4, False, True),
    ("benchmark_quadrature_assumption.py", 1, False, True),
    ("benchmark_mujoco_trajectory_error.py", 3, False, True),
    ("benchmark_order_estimator_comparison.py", 4, False, True),
    ("benchmark_regime_gated_fingerprint.py", 3, False, True),
    # comparison with Gebrekidan and Gebrerufael (2026), and the supporting extrapolation statistic
    ("benchmark_head_to_head_predecessor_screen.py", 12, True, False),
    ("benchmark_matched_step_control.py", 1, True, False),
    ("benchmark_cartpole_order_adapted_richardson.py", 1, True, True),
    ("benchmark_parameter_fault_sweep.py", 3, False, False),
    ("benchmark_matched_compute_noise_fingerprint.py", 6, False, True),
    ("benchmark_floor_distribution.py", 6, False, False),
    ("benchmark_parameter_identification_from_invariant.py", 3, False, False),
]


def run_one(script, quick):
    env = dict(os.environ, DAFD_QUICK="1" if quick else "0",
           PYTHONUNBUFFERED="1", PYTHONHASHSEED="0")
    logdir = ROOT / "Results" / ("quick" if quick else "") / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / (Path(script).stem + ".done")).unlink(missing_ok=True)
    t0 = time.time()
    with open(logdir / (Path(script).stem + ".log"), "w") as log:
        rc = subprocess.call([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if rc == 0:
        (logdir / (Path(script).stem + ".done")).touch()
    return script, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-slow", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.jobs != 1:
        ap.error("Use --jobs 1: trial-file dependencies require sequential execution")
    todo = [s for s in SCRIPTS if a.only in s[0] and not (a.skip_slow and s[1] > 5)]
    if a.resume:
        ld = ROOT / "Results" / ("quick" if a.quick else "") / "logs"
        todo = [s for s in todo if not (ld / (Path(s[0]).stem + ".done")).exists()]
    pred = (ROOT / "external" / "algebraicRLtest" / "ideal_diff.py").exists() or os.environ.get("PREDECESSOR_REPO")
    if not pred:
        skipped = [s[0] for s in todo if s[2]]
        if skipped:
            raise SystemExit("Comparison repository required for selected benchmarks; "
                             "run make predecessor or set PREDECESSOR_REPO (see README).")
    print(f"Running {len(todo)} scripts (quick={a.quick}, jobs={a.jobs})")
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    timings = []
    failures = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for script, rc, dt in ex.map(lambda s: run_one(s[0], a.quick), todo):
            timings.append(dict(script=script, exit_code=rc, wall_seconds=dt))
            if rc != 0:
                failures.append(script)
            print(f"  {'OK ' if rc == 0 else 'FAIL'} {script:<62} {dt / 60:5.1f} min")
    full = not (a.quick or a.only or a.skip_slow or a.resume)
    report = dict(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                  scope="full Python benchmark suite" if full else "selected Python benchmarks",
                  includes_native_source_build=False, includes_paper_build=False,
                  wall_seconds=time.perf_counter()-t0, jobs=a.jobs, quick=a.quick,
                  python=platform.python_version(), platform=platform.platform(),
                  processor=platform.processor(), logical_cpus=os.cpu_count(),
                  versions={p: importlib.metadata.version(p) for p in
                            ("numpy", "scipy", "sympy", "pandas", "gymnasium", "mujoco")},
                  scripts=timings, failures=failures)
    output = ROOT / "Results" / ("quick" if a.quick else "")
    output.mkdir(parents=True, exist_ok=True)
    (output / ("runtime_full.json" if full else "runtime_selected.json")).write_text(
        json.dumps(report, indent=2)+"\n")
    print("Results in", ROOT / "Results" / ("quick" if a.quick else ""))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
