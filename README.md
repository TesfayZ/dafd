# Diagnosing Model and Discretisation Errors in Reinforcement Learning Simulators Using Invariants
Python implementation and reproducibility tools for **Diagnosing Model and Discretisation Errors in Reinforcement Learning Simulators Using Invariants**.
The code tests symbolic invariance, classifies energy-defect refinement patterns,
and audits parameter consistency and numerical integration in Gymnasium and MuJoCo.
The comparison benchmarks use the reference-based invariant screen from prior
work by Gebrekidan and Gebrerufael, *Diagnosing Faults in Reinforcement Learning
Simulators with Canonical Polynomial Invariants*
([arXiv:2609.13194v2](https://arxiv.org/abs/2609.13194v2)). Its code is a separately
supplied dependency; see [comparison setup](#comparison-dependency).

## Installation and first run

Use Python 3.14.6 to match the reference environment.
`requirements-lock.txt` pins the Python packages, including Gymnasium 1.4.0 and MuJoCo 3.16.0. `requirements.txt` provides
broader package bounds while retaining those simulator pins. The lock file does
not install Python or system build tools.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-lock.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# Run an audit without the external comparison dependency:
python run_all_experiments.py --only mujoco_integrator_order

# Run the implementation checks:
python -m unittest test_corrections -v
```

For the complete suite, first configure the [comparison dependency](#comparison-dependency):

```bash
export PREDECESSOR_REPO=/path/to/comparison-checkout
make predecessor
make quick                 # reduced smoke suite -> Results/quick/
make all                   # full Python experiment suite -> Results/
make source-audit          # fetch, build and audit the pinned MuJoCo source
```

The native MuJoCo source audit uses Linux or WSL with network access, Git,
CMake, GNU Make, `g++` and `cmp`. CMake is included in the lock file; install
the other tools separately. The experiment runner can also be invoked directly
without Make.

| Command | Output or purpose |
|---|---|
| `make all` | Rerun all Python benchmarks into `Results/`. |
| `make resume` | Continue using completion markers in `Results/logs/`; use `make all` after changing source. |
| `make source-audit` | Build and audit the separately pinned upstream engine. |
| `python run_all_experiments.py --only mujoco` | Run scripts matching a name substring. |
| `python -m unittest test_corrections -v` | Run implementation and reproduction checks. |

Keep `JOBS=1`: sensitivity analysis consumes trial files from earlier benchmarks.
The runner rejects a missing comparison dependency and exits with a failure
status if any selected benchmark fails.
Smoke outputs under `Results/quick/` do not replace the full result files.

## Implementation

- [fingerprint.py](fingerprint.py): refinement exponents, round-off floor,
  `guarded_classify` and order-adapted extrapolation weights.
- `acrobot.py`, `acrobot_parametric.py`, `acrobot_mpmath.py`: analytic,
  parameterised and high-precision Acrobot models.
- `shipped_acrobot.py`, `shipped_cartpole.py`, `shipped_classic_control.py`:
  access to internal Gymnasium state, timestep controls and fault injection.
- `benchmark_*.py`: experiment entry points writing CSVs under `Results/`.
- `run_all_experiments.py`: sequential runner, logs, resume markers and timing reports.
- `results_io.py`: output routing; `DAFD_QUICK=1` selects `Results/quick/`.
- `check_mujoco_source.py`, `audit_mujoco_source.cc`, `reproduce_mujoco_source.py`:
  pinned native engine build and comparison.

The diagnostic restores the same start and integrates a fixed interval `h`
with 1, 2, 4 and 8 substeps. It classifies the change in mean absolute energy
defect across those refinements. It requires a specified invariant, restorable
float64 internal state and a callable step with adjustable timestep.

## Experiments and result files

Paths in the last column are relative to `Results/`. Section numbers refer to
the accompanying paper. Trial CSVs retain individual outcomes; summary CSVs
feed the generated tables. Counts across reused configurations are evaluations,
not independent starts.

| Paper | Script | Result files (`Results/`) |
|---|---|---|
| 3.3.1, App. A | benchmark_symbolic_lie_derivative.py | symbolic_lie_derivative_acrobot.csv |
| App. G | benchmark_rk4_defect_high_precision.py | rk4_defect_high_precision.csv |
| App. G, App. C | benchmark_signed_defect_and_r2_asymptotics.py | rk4_signed_defect.csv, r2_asymptotics.csv |
| 5.1.1 | benchmark_shipped_acrobot_fingerprint.py | shipped_acrobot_fingerprint_summary.csv, shipped_acrobot_fingerprint_trials.csv |
| 5.2.1 | benchmark_shipped_acrobot_calibrated_detectors.py | shipped_acrobot_calibrated_detectors.csv |
| 5.1.1 | benchmark_false_alarm_floor_noise_guard.py | false_alarm_healthy.csv, class_level_floor.csv, noise_guard.csv |
| 5.1.2 | benchmark_finite_step_classifier_validation.py | finite_step_validation.csv, finite_step_sign_change_scan.csv |
| 5.3.4 | benchmark_repeated_start_analysis.py | repeated_start_summary.csv |
| 5.3.1 | benchmark_transfer_validation.py | transfer_validation.csv, transfer_trials.csv, transfer_horizon.csv, energy_preserving_integrator.csv |
| 5.3.3 | benchmark_classifier_sensitivity.py | classifier_sensitivity.csv |
| 5.3.2 | benchmark_noise_sources_operating_envelope.py | noise_sources_operating_envelope.csv |
| 5.4.1 | benchmark_cartpole_stale_constants.py, benchmark_domain_randomisation.py, benchmark_stale_constants_consequences.py | cartpole_stale_constants.csv, domain_randomisation_audit.csv, stale_constants_divergence.csv, attribute_sensitivity.csv |
| 5.4.2 | benchmark_mujoco_integrator_order.py, benchmark_mujoco_audit_robustness.py | mujoco_integrator_order_audit.csv, mujoco_audit_shipped_step.csv, mujoco_audit_seeds.csv |
| 5.4.3, App. D and F | benchmark_quaternion_joint_order_control.py, benchmark_quaternion_mechanism.py | quaternion_joint_order_control.csv, quaternion_mechanism_{order,replication,convergence,local_error}.csv |
| App. F | benchmark_quadrature_assumption.py | quadrature_assumption_summary.csv, quadrature_assumption_trials.csv |
| 5.4.3 | benchmark_mujoco_trajectory_error.py, benchmark_order_estimator_comparison.py | mujoco_trajectory_error.csv, order_estimator_comparison.csv |
| App. B | benchmark_regime_gated_fingerprint.py | regime_gated_fingerprint.csv |
| 5.2.2 | benchmark_self_convergence.py | self_convergence_summary.csv, self_convergence_trials.csv, self_convergence_steps.csv |
| 5.2.3 | benchmark_head_to_head_predecessor_screen.py, benchmark_matched_step_control.py | head_to_head_predecessor_screen.csv, matched_step_control.csv |
| App. C | benchmark_cartpole_order_adapted_richardson.py, benchmark_parameter_fault_sweep.py, benchmark_matched_compute_noise_fingerprint.py, benchmark_floor_distribution.py | cartpole_order_adapted_richardson.csv, parameter_fault_sweep*.csv, matched_compute_noise_fingerprint.csv, floor_distribution.csv |
| App. E | benchmark_parameter_identification_from_invariant.py | parameter_identification.csv |

The finite-step benchmark also supports partitioned runs (`--sets`, `--orders`, `--tag`, then `--merge`); see its argument parser.

## Results and audit provenance

Experiment scripts write trial and summary files under `Results/`. These files
provide the numerical evidence for the validation studies and simulator audits
reported in the paper.

- **Acrobot validation and comparisons:** `shipped_acrobot_fingerprint_*.csv`,
  `finite_step_validation.csv` and `shipped_acrobot_calibrated_detectors.csv`
  retain classification and detector outcomes. The calibrated comparison uses
  the same 100 held-out healthy starts and 40 fault starts within each row;
  threshold-based methods also use 100 separate healthy calibration starts.
- **State self-convergence:** `self_convergence_*.csv` and
  `self_convergence_manifest.json` record a same-call control on 40 starts reused
  across six cases and two steps. State self-convergence estimates numerical
  order by comparing the endpoints obtained with 1, 2, 4 and 8 substeps over
  the same interval. The energy diagnostic reuses those endpoints, so the two
  diagnostics have the same simulator-call budget.
- **Transfer and uncertainty:** `transfer_validation.csv`, `transfer_trials.csv`
  and `repeated_start_summary.csv` separate configuration-level rates from
  start-level summaries. Configuration-level rates and Wilson intervals use
  starts that are reused across configurations.
- **Parameter consistency audits:** `domain_randomisation_audit.csv`,
  `cartpole_stale_constants.csv`, `stale_constants_divergence.csv` and
  `attribute_sensitivity.csv` connect parameter assignments, cached constants
  and executed dynamics.
- **Released-engine audit:** `mujoco_integrator_order_audit.csv`,
  `mujoco_audit_shipped_step*.csv` and `mujoco_audit_seed*.csv` record the
  MuJoCo 3.16.0 tests. These use conservative configurations and distinguish
  native engine timesteps from enlarged audit intervals and Gymnasium frame skip.
- **Mechanism controls:** `quaternion_mechanism_*.csv`,
  `quaternion_joint_order_control.csv` and `quadrature_assumption_*.csv`
  connect the heavy-top replica, orientation controls and numerical assumption
  checks to the mechanism analysis. The CF4 update is a replica control.
- **Separate upstream build:** `mujoco_main_trials.csv`, `mujoco_main_audit.csv`,
  `mujoco_main_trajectory.csv` and `mujoco_source_check.json` identify the tested
  development revision, outputs and source/build hashes. These results have a
  separate version scope from the released-engine tables.

`Results/refinement_run_manifest.json` records the reference experiment run.
Random draws use fixed seeds; label-derived seeds use CRC32 and the runner sets
`PYTHONHASHSEED=0`. Classifier constants were developed on Acrobot; sensitivity
variants and extrapolation studies are exploratory. The transfer benchmarks use
the frozen classifier. Protocols and trial files retain abstentions and failures.

The runner writes per-script timings, package versions, platform information and
failure status to `Results/runtime_full.json` for a full run and
`Results/runtime_selected.json` for partial or resumed runs. Python-suite timing
excludes native compilation. The native build records its
own timing and build reuse information in `Results/runtime_source_build.json`.
Consult these reports and the run manifests for execution-specific provenance.

## Comparison dependency

The three comparison scripts use the implementation accompanying Gebrekidan
and Gebrerufael (2026), [arXiv:2609.13194v2](https://arxiv.org/abs/2609.13194v2),
at revision `687c868c204b8ec537451b5e046532de80197d9d`.
Obtain the source at that revision and retain its licence. Required modules
are `bugs.py`, `envs.py`, `ideal_diff.py` and `deflation.py`. From the project root:

```bash
git clone https://github.com/TesfayZ/algebraicRLtest.git external/algebraicRLtest
git -C external/algebraicRLtest checkout --detach 687c868c204b8ec537451b5e046532de80197d9d
make predecessor PY=venv/bin/python
```

For an existing checkout, select the same revision and set
`PREDECESSOR_REPO=/path/to/comparison-checkout` before running `make predecessor`.
That command checks imports; it does not clone the repository or verify its
revision. Only
`benchmark_head_to_head_predecessor_screen.py`, `benchmark_matched_step_control.py`
and `benchmark_cartpole_order_adapted_richardson.py` require this dependency.
The `PREDECESSOR_REPO` variable, target and filenames are interface names for
locating the external comparison implementation.

The comparisons evaluate the screen of Gebrekidan and Gebrerufael at matched
steps and against a direct reference step, as specified in the paper. They
cover one shared fault in a noise-free, unpaired test. They do not evaluate
that method's full invariant recovery, localisation or observation-channel diagnosis.

## Pinned MuJoCo source audit

The source audit targets commit `8892bf44b07c20f0e4dcd8b8c34ca46683b97b76`. `cmp -s` on the extracted `mj_RungeKutta` functions from release tag **3.16.0** and this commit returns **1 (different)**. The development implementation maps free/ball-joint stage angular velocities through the inverse differential of the exponential map (`rkCorrectVel`), a Runge–Kutta–Munthe-Kaas (RKMK) correction. The released and development implementations are audited separately.

The audited files are the [released implementation](https://github.com/google-deepmind/mujoco/blob/3.16.0/src/engine/engine_forward.c)
and [pinned corrected implementation](https://github.com/google-deepmind/mujoco/blob/8892bf44b07c20f0e4dcd8b8c34ca46683b97b76/src/engine/engine_forward.c).
The source build uses that upstream revision without a local RKMK patch; the
comparison does not isolate one upstream change from all other revision changes.
The independent CF4 remedy is `step_cf4` in
[benchmark_quaternion_mechanism.py](benchmark_quaternion_mechanism.py).
It modifies the replica and is not an engine patch. Exact source and extracted
function SHA-256 values are archived in
[mujoco_source_check.json](Results/mujoco_source_check.json).

Run from the repository root after generating the released-engine results:

```bash
make source-audit PY=venv/bin/python
```

The script creates `external/mujoco-source/source/` for the pinned upstream
checkout and `external/mujoco-source/build/` for compiled files. It downloads,
configures and builds the engine, runs the native comparison and records source,
library and worker hashes. These directories are generated locally and are
excluded from version control.

To choose another working directory or compilation job count:

```bash
venv/bin/python reproduce_mujoco_source.py --work-dir external/mujoco-audit --jobs 4
```

`--work-dir` is a directory created by the script, relative to the current working
directory unless an absolute path is supplied. Use a new directory for a different
revision: the script refuses to alter an existing checkout at another revision.

The C++ worker links directly to the newly built shared library. Installed Python bindings supply Gymnasium XML assets and start positions only; they do not execute the new engine. As a protocol check, the same worker built against installed 3.16.0 is compared against the Python audit’s archived native-step residuals. The script requires a clean checkout at the pinned commit and records function, library and worker hashes.

For each model, 3 seeds × 24 starts × 20 steps use the same conservative setup and initial velocities across tested steps. At native steps, **72/72 starts abstain for each model**. At the first level resolving at least half the starts, Ant has **48 consistent, 24 undetermined** (h=0.04, 0.16, 0.04 s by seed); Humanoid has **72 consistent** at h=0.012 s. No selected label is order-2 or model. An eight-start trajectory control over 0.48 s compares the full position vector against a same-engine h0/256 reference: successive median-error slopes are **4.001, 4.000, 4.001** (Ant, h0=0.04 s) and **3.736, 3.995, 3.975** (Humanoid, h0=0.048 s). These controls recover fourth-order convergence; the released-engine finding does not extend to this revision.

`Results/mujoco_main_trials.csv`, `mujoco_main_audit.csv`, `mujoco_main_trajectory.csv` and `mujoco_source_check.json` retain the separate source-build results. The released-engine tables describe the pinned released engine, 3.16.0.

## Auditing a new simulator

Use the following workflow to apply `fingerprint.guarded_classify` to a simulator.
The quantisation bounds and optional follow-up measurements below are practical
recommendations, not new validated classifier variants.

### Prepare a reproducible step

1. Specify the nominal parameters, invariant, energy units and energy offset.
   Keep these fixed throughout the audit. If symbolic equations are available,
   run the exact layer on that specification. Its certificate covers the
   symbolic model; it does not certify the executable.
2. Check that the internal state is accessible in double-precision floating-point format (float64) and can be restored.
   Include simulation time, cached quantities, controller memory and random
   state if they affect the step. Use a deterministic passive smooth segment.
   Restoring only the observation can leave different hidden simulator states.
3. Choose a macro interval `h`, initially the native engine timestep. From each
   identical restored start, integrate `m` times with timestep `h/m` for
   `m = 1, 2, 4, 8`. Each branch must end at the same physical time.
4. At every sampled start, compute `abs(E(end_m) - E(start))/h`. Average these
   absolute residuals separately for each `m`; taking the absolute value after
   averaging signed defects can cancel the signal. Apply
   `fingerprint.guarded_classify(R1, R2, R4, R8, q, mean_abs_start_energy, h,
   noise_floor=...)`. The optional `noise_floor` is in energy/time units,
   not an energy standard deviation.
5. Archive the residuals, exponents, floor, labels, starts, parameters, timestep,
   declared order and software revision. Repeat a suspicious result over more
   starts and smaller macro intervals before assigning a physical cause.

For Gymnasium classic control, the wrappers in `shipped_acrobot.py` and
`shipped_cartpole.py` expose the internal state and step size. Avoid reconstructing
Acrobot state from its float32 trigonometric observation. Check clipping and
termination separately; nonsmooth branches are outside the smooth theory.

For MuJoCo, follow `benchmark_mujoco_integrator_order.py`: use native engine
steps, restore engine state and recompute derived quantities before reading
energy. The archived experiment disables actuation, damping, friction,
contacts, equality constraints and joint limits. This creates a conservative
audit configuration; it does not test the original task's policy dynamics.
Treat Gymnasium `frame_skip` separately from `model.opt.timestep`.

### Interpret the result and choose the next action

| Observation | Interpretation | Next action |
|---|---|---|
| `consistent` | Resolved energy exponents meet the declared-order rule. | Retain the measured exponents; use state self-convergence if trajectory order matters. Invariant-preserving errors can pass. |
| `model` | A persistent or flattening defect suggests invariant-breaking dynamics; finite-step cancellation and unmodelled noise can mimic it. | Check invariant parameters and caches, then repeat at smaller `h` and other starts. |
| `order-k` | Numerical error dominates the resolved energy signal. | Check the executed integrator and state self-convergence. A smaller model error can remain masked. |
| `undetermined` | The finest defect is unresolved, or energy is preserved too accurately to infer order. | Check precision and noise; use another physical invariant or state self-convergence. Enlarge `h` only within a credible smooth regime. |
| `unstable` / `blowup` | Excessive defect rate / nonfinite results cause abstention. | Reduce `h`, inspect resets and constraints, and preserve the failed case. |
| Contact, limit, reset or changing control law within a branch | The smooth passive argument does not justify a label. | Abstain or justify a separate regime gate/balance-law experiment. |
| Unknown declared order | Exponents are computable, but conformance is unspecified. | Report `(a,b,c)` and their spread without a `consistent` label or a certified trajectory order. |

### Estimate noise without treating quantisation as random variation

For independently varying readout noise, repeatedly evaluate energy from fresh
reads of the same frozen state. One estimate is
`sd(E_read_1 - E_read_2)/sqrt(2)` across independent pairs. Repeat across
representative states, since the energy sensitivity can vary with state.
Temporal correlation or common bias invalidates the independent-pair formula;
record that uncertainty rather than substituting zero. The paper's Gaussian
floor is `5*sqrt(2/pi)*sqrt(2)*sigma_E/h`, a heuristic margin without a
distribution-free false-alarm guarantee.

Deterministic quantisation gives identical repeated reads. If the rounding
format and rule are known, construct the cell of possible pre-rounding states.
For round-to-nearest, component boundaries are the midpoints to the adjacent
representable numbers; cells can be asymmetric at exponent boundaries. Include
sensor noise or clipping uncertainty separately. Interval evaluation of `E` on
this cell can bound its error. Alternatively, if component errors satisfy
`|dx_i| <= d_i` and derivative bounds `L_i >= sup_cell |dE/dx_i|` are available,
the mean-value theorem gives `B_E <= sum_i L_i*d_i`. A gradient evaluated only
at the reported state is a first-order estimate, not a guaranteed bound.
Account for energy-evaluation arithmetic as well.

Endpoint bounds `B_start` and `B_end_m` imply residual error at most
`(B_start + B_end_m)/h`, also for the absolute residual. Average these bounds
over sampled starts for each refinement. If the resulting intervals include
zero or allow incompatible exponent interpretations, abstain. This interval
check is separate from the frozen Gaussian guard; it is not automatically
implemented by `guarded_classify`. Uniform half-ulp dithering probes one
assumed subcell distribution and cannot identify the actual unknown error.
If cells or energy-error bounds are unavailable, the access requirement remains
unmet. More repeated identical reads cannot repair it.

### Bound the scope of extensions

Additional levels such as `m=16,32` can reveal curvature, but can also move the
signal below the floor. Robust log--log regression assumes a power law;
`D + T*m**(-q)` can flatten or cross zero, violating that assumption. No measured
improvement over the frozen four-level rule is claimed. A new estimator needs
a separate protocol, a noise model accounting for shared start energies, and
evaluation retaining all failures and abstentions.

Changing `frame_skip` at fixed engine timestep or thinning logs changes the
physical interval, not the integration resolution at fixed interval. It cannot
stand in for the required refinement family. A genuinely adjustable internal
step can support the method if all branches cover the same interval.

No actuated balance-law experiment is reported. Such an extension must integrate
the same continuous-time input on every branch and subtract work evaluated with
sufficient-order quadrature. Controller resampling, actuator state, dissipation
and stochastic forcing all require explicit treatment before transferring the
passive classifier's interpretation or guards.

For exact reproduction of the engine comparison, use the README's
[pinned source audit](#pinned-mujoco-source-audit).
The RKMK correction belongs to the unmodified upstream revision, whereas
`step_cf4` in `benchmark_quaternion_mechanism.py` is a separate replica control.

## Interpretation and limits

`consistent` means that resolved energy exponents meet the declared-order rule;
it does not certify the dynamics. `model` indicates a persistent invariant defect;
finite-step cancellation and noise can mimic it. An `order-k` label reports the
resolved energy pattern and can mask smaller model errors. `undetermined`,
`unstable` and `blowup` are abstentions.

The implemented validation covers passive conservative smooth dynamics.
Actuated, dissipative and contact balance laws need separate treatment.
Invariant-preserving model errors can pass, and energy-preserving integrators
can leave no usable refinement signal. The asymptotic theory supplies no
finite-step error guarantee. Readout noise can make labels unresolved; the
noise experiments do not establish a universal sensor threshold. No policy-level
effect is measured by these audits.

## Minimal MuJoCo model specification

The quaternion-joint control uses a base with a hinged arm, gravity, RK4, native timestep 0.01 and audited macro step 0.05:

```xml
<mujoco><option gravity="0 0 -9.8" integrator="RK4" timestep="0.01"/>
 <worldbody><body pos="0 0 2">
  <freejoint/>   <!-- or: three slide joints then three hinge joints -->
  <geom type="box" size=".2 .1 .05" mass="1"/>
  <body pos=".3 0 0"><joint type="hinge" axis="0 1 0"/>
   <geom type="capsule" fromto="0 0 0 .5 0 0" size=".03" mass=".5"/></body>
 </body></worldbody></mujoco>
```

The ordinary-joint control replaces the free joint with three slide joints followed by three hinge joints, as implemented in `benchmark_quaternion_joint_order_control.py`. The torque-free control removes the arm and sets gravity to zero. The heavy-top replica uses a box of half-sizes (0.2, 0.1, 0.05), mass 1, with its centre of mass offset 0.3 along z from a ball joint at the origin. Energy is MuJoCo kinetic plus potential energy after `mj_forward`, with energy computation enabled. Each macro-step refinement takes m engine steps with timestep h/m for m in {1,2,4,8}.

## Direct numerical check of assumption Q

`python benchmark_quadrature_assumption.py` records the four actual stage velocities from the heavy-top RK4 replica and compares their weighted mean with an augmented DOP853 solve integrating angular velocity. It uses seed 3, 12 starts reused across four halvings from h=0.05, both with gravity and torque-free. [Results/quadrature_assumption_summary.csv](Results/quadrature_assumption_summary.csv) reports median errors and halving slopes; [Results/quadrature_assumption_trials.csv](Results/quadrature_assumption_trials.csv) retains all 96 per-start records. Gravity-case slopes are 3.991, 3.921 and 3.985; torque-free slopes are approximately 4.000. Reference solves at (rtol, atol)=(1e-12, 1e-14) and (1e-13, 1e-15) provide a tolerance-sensitivity check, not a certified reference-error bound. These measurements support Q on the replica; they do not prove Q for general coupled models.
