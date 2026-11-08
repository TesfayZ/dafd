#!/usr/bin/env python3
"""Assert that the headline numbers quoted in doc/DAFD-paper.tex match Results/*.csv.

Run after the experiments (make check).  Exit status 1 if any claim fails.  Each check names the paper location it
guards, so a failure points at the sentence that has to be updated (or at a regression in the code).
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "Results"
FAILED, TOTAL = [], 0


def R(name):
    return pd.read_csv(RES / f"{name}.csv")


def cls(s):
    return ast.literal_eval(s)


def count(df, label, col="classes"):
    return sum(cls(c).get(label, 0) for c in df[col])


def check(where, what, got, want, tol=None):
    """tol=None: exact equality; otherwise |got-want| <= tol (absolute)."""
    global TOTAL
    TOTAL += 1
    ok = (got == want) if tol is None else bool(abs(got - want) <= tol)
    if not ok:
        FAILED.append(f"[{where}] {what}: got {got}, paper says {want}")
    print(("  ok   " if ok else "  FAIL ") + f"[{where}] {what}: {got}")


def main():
    # ---- 5.1.1 shipped Acrobot -------------------------------------------------------------------------------------------
    fa = R("false_alarm_healthy").iloc[0]
    check("abstract, 5.1.1", "healthy shipped-Acrobot runs consistent", (int(fa.n_healthy), int(fa.n_flagged)), (1000, 0))
    t = R("shipped_acrobot_fingerprint_trials")
    h = t[t.case == "healthy"]
    ex = np.log2(np.c_[h.r1 / h.r2, h.r2 / h.r4, h.r4 / h.r8])
    check("5.1.1", "healthy exponent minima", tuple(ex.min(0).round(2)), (4.14, 3.96, 3.97))
    check("5.1.1", "healthy exponent medians", tuple(np.median(ex, 0).round(2)), (4.38, 4.11, 4.03))
    check("5.1.1", "healthy exponent maxima", tuple(ex.max(0).round(2)), (4.95, 4.90, 4.49))
    check("5.1.1", "guard margin median (r8/phi_round), healthy", round(h.guard_margin.median() / 1e6, 1), 3.5, tol=0.05)
    check("5.1.1", "guard margin minimum, healthy (>=1.5e4)", bool(h.guard_margin.min() >= 1.5e4), True)
    s = R("shipped_acrobot_fingerprint_summary")
    get = lambda c: cls(s[s.case == c].classes.iloc[0])
    check("5.1.1", "rk2", get("rk2"), {"order-2": 20, "order-3": 8, "model": 1, "consistent": 1})
    check("5.1.1", "euler", get("CAT euler"), {"order-1": 30})
    check("5.1.1", "euler + m2 +1e-2", get("euler + m2 +1e-2"), {"order-1": 30})
    check("5.1.1", "COM_POS_1 1e-5", get("LINK_COM_POS_1 +1e-05"), {"model": 26, "consistent": 4})
    check("5.1.1", "MASS_1 1e-5", get("LINK_MASS_1 +1e-05"), {"model": 29, "consistent": 1})
    sel = s[s.case.str.startswith("LINK_") & s.case.str.endswith("+1e-05")]
    check("5.3.4", "pooled 1e-5 detection", (int(sel.n_model.sum()), int(sel.n.sum())), (175, 180))
    check("abstract", "minimum 1e-5 detection (percent, six attributes)",
          (len(sel), round(float((sel.n_model / sel.n).min()) * 100, 1)), (6, 86.7))
    floor_data = R("class_level_floor")
    weak = floor_data[np.isclose(floor_data.eps, 1e-6, rtol=1e-10, atol=0)]
    check("abstract", "1e-6 model count range, 30 starts per attribute",
          (int(weak.n_model.min()), int(weak.n_model.max()), set(weak.n)), (1, 14, {30}))
    agreement = R("acrobot_step_agreement")
    random = agreement[agreement.population == "random"].max_abs_difference
    check("5.1.1", "single-state / 1000-start median / maximum step difference",
          (len(random), tuple(float(f"{x:.3g}") for x in
            (agreement[agreement.population == "example"].max_abs_difference.iloc[0], random.median(), random.max()))),
          (1000, (2.78e-17, 2.22e-16, 8.88e-16)))
    cd = R("shipped_acrobot_calibrated_detectors")
    hh = cd[cd.case == "healthy"].iloc[0]
    check("5.1.1", "calibrated false alarms drift/raw/sub8/R2", (hh.det_drift, hh.det_raw, hh.det_sub8, hh.det_R2), (0, 4, 6, 0))
    m1 = cd[(cd.case == "LINK_MASS_1") & np.isclose(cd.eps, 1e-4)].iloc[0]
    check("5.1.1", "LINK_MASS_1 1e-4: calibrated R2 / fingerprint", (float(m1.det_R2), int(m1.det_fingerprint)), (15.0, 40))
    cf = R("class_level_floor")
    for att, want in (("LINK_MASS_2", (1, 14, 26)), ("LINK_MOI", (1, 14, 26)), ("LINK_LENGTH_1", (0, 10, 23)), ("LINK_COM_POS_1", (0, 1, 7))):
        d = cf[cf.attribute == att].set_index("eps").n_model
        check("5.1.1", f"class-level floor {att} at 1e-7,1e-6,3e-6", tuple(int(d[e]) for e in (1e-7, 1e-6, 3e-6)), want)
    ng = R("noise_guard")
    r = ng[(ng.case == "healthy") & np.isclose(ng.sigma_E, 1e-6)].iloc[0]
    check("5.1.1", "healthy @1e-6 unguarded model", cls(r.unguarded).get("model"), 37)
    check("5.1.1", "healthy @1e-6 guarded undetermined", cls(r.guarded).get("undetermined"), 59)

    # ---- 5.1.2 finite-step ----------------------------------------------------------------------------------------------
    fs = R("finite_step_validation").drop_duplicates(subset=["start_set", "true_order", "h", "eps"])
    for ss in ("development", "held_out"):
        d = fs[(fs.start_set == ss) & (fs.eps == 0)]
        check("5.1.2", f"healthy model labels, {ss}", (int(d.n.sum()), int(round((d.model_rate * d.n).sum()))), (480, 0))
    d = fs[(fs.start_set == "held_out") & (fs.eps == 0) & (fs.true_order == 4) & np.isclose(fs.h, 0.2)].iloc[0]
    check("5.1.2", "held-out RK4 h=0.2 healthy classes", cls(d.classes), {"consistent": 38, "order-4": 2})
    d = fs[(fs.start_set == "near_equilibrium") & (fs.eps == 0) & (fs.true_order == 4)]
    check("5.1.2", "near-equilibrium RK4 healthy undetermined 40/40 at all h", [cls(c) for c in d.classes], [{"undetermined": 40}] * 3)
    d = fs[(fs.start_set == "near_equilibrium") & (fs.eps == 0.01) & (fs.true_order >= 2)]
    check("5.1.2", "near-equilibrium 1e-2 detected for q>=2", [cls(c) for c in d.classes], [{"model": 40}] * 9)
    sc = R("finite_step_sign_change_scan")
    f = sc[sc.cls_with_floor != "undetermined"]
    check("5.1.2", "with floor, every h<=0.043 undetermined", float(f.h.min()) > 0.05, True)

    # ---- 5.3.1 transfer ----------------------------------------------------------------------------------------------
    tv = R("transfer_validation")
    h0 = tv[tv.eps == 0]
    check("abstract, 5.3.1", "transfer healthy runs / flagged", (int(h0.n.sum()), int(h0.flagged.sum())), (1440, 0))
    f2 = tv[(tv.eps == 1e-2) & (tv.true_order >= 2)]
    check("5.3.1", "1e-2 faults labelled model (q>=2)", (int(round((f2.model_rate * f2.n).sum())), int(f2.n.sum())), (1080, 1080))
    f4 = tv[(tv.eps == 1e-4) & (tv.true_order == 4)]
    check("5.3.1", "RK4 1e-4 faults labelled model", (int(round((f4.model_rate * f4.n).sum())), int(f4.n.sum())), (360, 360))
    ab = lambda sysn, q: sum(v for c in h0[(h0.system == sysn) & (h0.true_order == q)].classes for k, v in cls(c).items()
                             if k in ("undetermined", "unstable", "blowup"))
    check("5.3.1", "healthy abstentions HH q=1 / Euler top q=4 / Kepler q=4", (ab("henon_heiles", 1), ab("euler_top", 4), ab("kepler", 4)), (108, 29, 2))
    et = tv[(tv.system == "euler_top") & (tv.true_order == 1) & (tv.eps == 1e-2)]
    check("5.3.1", "Euler top q=1 1e-2 model rate", round((et.model_rate * et.n).sum() / et.n.sum(), 2), 0.11)
    ho = R("transfer_horizon")
    check("5.3.1", "horizon: healthy consistent and 1e-3 model for T=20,200",
          all((cls(c) == {"consistent": 20}) if e == 0 else (cls(c) == {"model": 20}) for c, e in zip(ho.classes, ho.eps)), True)
    ep = R("energy_preserving_integrator")
    check("5.3.1", "implicit midpoint abstains", [cls(c) for c in ep.classes], [{"undetermined": 40}] * 3)

    # ---- 5.3.2 noise ----------------------------------------------------------------------------------------------------
    ns = R("noise_sources_operating_envelope")
    check("5.3.2", "healthy false alarm in every noise row", float(ns.healthy_false_alarm.max()), 0.0)
    check("5.3.2", "number of noise rows", len(ns), 17)

    # ---- 5.3.3 sensitivity ----------------------------------------------------------------------------------------------
    se = R("classifier_sensitivity")
    u = se[se.analysis == "unified"].set_index("setting")
    check("5.3.3", "no variant flags a healthy run (dev + transfer)",
          int((u.dev_healthy_model + u.transfer_healthy_model + u.transfer_healthy_other_flag).sum()), 0)
    check("5.3.3", "c_thr/delta variants change no label",
          bool((u.loc[["c_thr = 0.25", "c_thr = 1.0", "delta = 0.1", "delta = 0.8"], ["dev_fault_1e-5_model", "transfer_fault_1e-2_model",
                                                                                      "transfer_healthy_abstain"]].values
                == u.loc["default (as in the paper)", ["dev_fault_1e-5_model", "transfer_fault_1e-2_model", "transfer_healthy_abstain"]].values).all()), True)
    check("5.3.3", "theta variants on dev 1e-5 faults (q-1.5, default, q-0.5)",
          tuple(int(u.loc[k, "dev_fault_1e-5_model"]) for k in ("theta = q - 1.5 (all q)", "default (as in the paper)", "theta = q - 0.5 (all q)")), (166, 175, 180))
    check("5.3.3", "round-off factor 10/50/500 abstentions",
          tuple(int(u.loc[k, "transfer_healthy_abstain"]) for k in ("round-off factor 10", "default (as in the paper)", "round-off factor 500")), (127, 139, 175))
    check("5.3.3", "1e-2 transfer faults still 1080/1080 in every variant", bool((u["transfer_fault_1e-2_model"] == 1080).all()), True)
    check("5.3.3", "weak HH 1e-4 faults under each sensitivity variant",
          (set(u["transfer_hh_fault_1e-4_n"]), tuple(u["transfer_hh_fault_1e-4_model"].astype(int))),
          ({360}, (254,254,254,254,254,270,241,239,254,254)))
    dq = se[se.analysis == "declared_q"].set_index("setting")
    check("5.3.3", "declared q=5 / q=6 healthy flagged", (int(dq.loc["q=5", "healthy_not_consistent"]), int(dq.loc["q=6", "healthy_not_consistent"])), (57, 1000))
    olr = se[se.analysis == "order_label_rule"].set_index("setting")
    check("5.3.3", "order label rule mean vs c: RK2 order-2 / healthy label 4",
          (int(olr.loc["mean", "rk2_labelled_order2"]), int(olr.loc["c", "rk2_labelled_order2"]), int(olr.loc["mean", "healthy_label_equals_4"]),
           int(olr.loc["c", "healthy_label_equals_4"])), (20, 25, 955, 986))
    eu = se[se.analysis == "c_thr_delta"].set_index("setting")
    check("5.3.3", "Euler order-1 label vs c_thr (0.5 / 1.0 / 1.5)",
          tuple(int(eu.loc[f"c_thr={c}, delta=0.4", "euler_labelled_order1"]) for c in (0.5, 1.0, 1.5)), (30, 7, 0))
    check("5.3.3", "minimum healthy exponent over 1000 runs", float(se[se.analysis == "healthy_margin"].value.iloc[0]), 3.831, tol=0.001)

    # ---- 5.4.1 randomisation audit -------------------------------------------------------------------------------------
    dr = R("domain_randomisation_audit")
    cp = dr[dr.environment == "CartPole"]
    for att in ("masscart", "masspole", "length"):
        d = cp[(cp.attribute == att) & (cp["update"] == "direct") & (cp.factor != 1.0)]
        check("5.4.1", f"CartPole {att} direct -> model 40/40 at x0.5,1.5,2", [cls(c) for c in d.classes], [{"model": 40}] * 3)
    d = cp[cp["update"] == "updated"]
    check("5.4.1", "CartPole updated consistent everywhere", all(cls(c) == {"consistent": 40} for c in d.classes), True)
    check("5.4.1", "CartPole gravity direct consistent", all(cls(c) == {"consistent": 40} for c in cp[cp.attribute == "gravity"].classes), True)
    check("5.4.1", "factor 1.0 consistent in every row", bool((dr[dr.factor == 1.0].dominant_class == "consistent").all()), True)
    ac = dr[dr.environment == "Acrobot"]
    check("5.4.1", "Acrobot all dominant-consistent; weakest cell 23/24", (bool((ac.dominant_class == "consistent").all()), int(ac.dominant_count.min())), (True, 23))
    cs = R("cartpole_stale_constants")
    g = lambda **kw: cs[np.logical_and.reduce([(cs[k] == v) | (cs[k].isna() & (v is None)) for k, v in kw.items()])]
    check("5.4.1", "healthy CartPole euler / semi-implicit consistent 200",
          [cls(c) for c in cs[cs.group == "healthy"].classes], [{"consistent": 200}] * len(cs[cs.group == "healthy"]))

    # ---- 5.4.2 MuJoCo ----------------------------------------------------------------------------------------------------
    mj = R("mujoco_integrator_order_audit").set_index("environment")
    order2 = [e for e, r in mj.iterrows() if cls(r.classes) == {"order-2": 12}]
    check("5.4.2", "environments at order 2 (12/12 each)", sorted(order2), ["Ant-v5", "Humanoid-v5", "HumanoidStandup-v5"])
    check("5.4.2", "all eleven environments audited", len(mj), 11)
    ok = [e for e, r in mj.iterrows() if e not in order2 and "order-2" not in r.classes and count(mj.loc[[e]], "model") == 0]
    check("abstract, 5.4.2", "other environments at declared order", len(ok), 8)
    ss = R("mujoco_audit_shipped_step").set_index("environment")
    check("5.4.2", "HumanoidStandup native order-2", cls(ss.loc["HumanoidStandup-v5", "classes"]), {"order-2": 24})
    check("5.4.2", "native step: Ant/Humanoid order-2 24; R/H/W undetermined",
          (cls(ss.loc["Ant-v5", "classes"]), cls(ss.loc["Humanoid-v5", "classes"]),
           [cls(ss.loc[e, "classes"]) for e in ("Reacher-v5", "Hopper-v5", "Walker2d-v5")]),
          ({"order-2": 24}, {"order-2": 24}, [{"undetermined": 24}] * 3))
    sd = R("mujoco_audit_seeds")
    hu = sd[sd.environment == "Humanoid-v5"]
    check("5.4.2", "Humanoid 3 seeds: order-2 / order-3 / model", (count(hu, "order-2"), count(hu, "order-3"), count(hu, "model")), (71, 0, 0))
    check("5.4.2", "Ant order-2 70/72, Hopper consistent 72/72", (count(sd[sd.environment == "Ant-v5"], "order-2"), count(sd[sd.environment == "Hopper-v5"], "consistent")), (70, 72))

    # ---- 5.4.3 mechanism ----------------------------------------------------------------------------------------------
    source = json.loads((RES / "mujoco_source_check.json").read_text())
    check("5.4.3", "pinned source comparison (different; RKMK present)",
          (source["commit"], source["cmp_exit_status"], source["byte_identical"], source["rkmk_present"]),
          ("8892bf44b07c20f0e4dcd8b8c34ca46683b97b76", 1, False, True))
    main_audit = R("mujoco_main_trials")
    for env, want in (("Ant-v5", (48, 24)), ("Humanoid-v5", (72, 0))):
        native = main_audit[(main_audit.environment == env) & (main_audit.K == 1)]
        selected = main_audit[(main_audit.environment == env) & main_audit.selected]
        check("5.4.3", f"source-built {env} native abstentions", (len(native), int((native.cls == "undetermined").sum())), (72, 72))
        check("5.4.3", f"source-built {env} selected consistent / abstained",
              (int((selected.cls == "consistent").sum()), int((selected.cls == "undetermined").sum())), want)
    trajectory = R("mujoco_main_trajectory")
    for env, want in (("Ant-v5", (4.001, 4.000, 4.001)), ("Humanoid-v5", (3.736, 3.995, 3.975))):
        med = trajectory[trajectory.environment == env].groupby("level").qpos_error.median().values
        check("5.4.3", f"source-built {env} trajectory slopes", tuple(np.log2(med[:-1]/med[1:]).round(3)), want)
    qc = R("quaternion_joint_order_control")
    qf = qc[qc.base_joints.str.startswith("free (quaternion)") & (qc.declared_integrator == "RK4")].iloc[0]
    check("5.4.3", "quaternion free joint RK4: order-2 24/24", cls(qf.classes), {"order-2": 24})
    qj = qc[qc.base_joints.str.startswith("3 slides") & (qc.declared_integrator == "RK4")].iloc[0]
    check("5.4.3", "six ordinary joints RK4: >=23/24 consistent, no order-2", (cls(qj.classes).get("consistent", 0) >= 23, "order-2" in qj.classes), (True, False))
    qt = qc[qc.base_joints.str.startswith("torque-free")].iloc[0]
    check("5.4.3", "torque-free: no order-2, no model label", ("order-2" in qt.classes, "model" in qt.classes), (False, False))
    le = R("quaternion_mechanism_local_error")
    quadrature = R("quadrature_assumption_summary")
    qs = quadrature.halving_slope.dropna()
    check("5.4.3", "direct Q check: 12 starts at each step in two models",
          (len(quadrature), set(quadrature.n)), (8, {12}))
    check("5.4.3", "direct Q check: median-error halving slopes between 3.9 and 4.1",
          bool(qs.between(3.9, 4.1).all()) and len(qs) == 6, True)
    check("5.4.3", "direct Q check: reference tolerance changes below 0.1 percent of median error",
          bool((quadrature.reference_change_max < 0.001 * quadrature.error_median).all()), True)
    last = le[np.isclose(le.h, 0.00625)]
    check("5.4.3", "Magnus coefficient at smallest h within 0.5% of -1/12", bool((abs(last.coef_median + 1 / 12) < 0.005 / 12).all()), True)
    check("5.4.3", "cosine with -w x wdot >= 0.999 at smallest h", bool((-last.cos_median >= 0.999).all()), True)
    g = le[le.model.str.startswith("heavy")].sort_values("h", ascending=False).energy_defect_median.values
    tf = le[le.model.str.startswith("torque")].sort_values("h", ascending=False).energy_defect_median.values
    slope = lambda v: np.log2(v[:-1] / v[1:])
    check("5.4.3", "energy-defect slope with gravity ~3 (final halving 2.7-3.3)", bool(2.7 <= slope(g)[-1] <= 3.3), True)
    check("5.4.3", "energy-defect slope torque-free ~5 (4.7-5.3)", bool(4.7 <= slope(tf)[-1] <= 5.3), True)
    cv = R("quaternion_mechanism_convergence").set_index("scheme")
    sl = lambda k: ast.literal_eval(cv.loc[k, "slopes"])
    check("5.4.3", "trajectory slopes: MuJoCo ~2, replica ~2, CF4 ~4, CF4 reversed ~2",
          (all(1.9 < x < 2.1 for x in sl("MuJoCo")), all(1.9 < x < 2.1 for x in sl("replica")), all(3.95 < x < 4.1 for x in sl("CF4")),
           all(1.9 < x < 2.1 for x in sl("CF4 reversed"))), (True, True, True, True))
    rp = R("quaternion_mechanism_replication").dropna(subset=["classes"]).set_index("scheme")
    check("5.4.3", "replica order-2 24/24; embedded and CF4 consistent 24/24",
          (cls(rp.loc["replica of MuJoCo RK4 update", "classes"]), cls(rp.loc["embedded RK4 + renormalisation", "classes"]),
           cls(rp.loc["commutator-free Lie RK4 (CF4)", "classes"])), ({"order-2": 24}, {"consistent": 24}, {"consistent": 24}))
    oe = R("order_estimator_comparison")
    check("5.4.3", "trajectory slope median 2.01 (Ant, Humanoid)",
          [round(float(oe[(oe.environment == e) & oe.estimator.str.startswith("trajectory")]["median"].iloc[0]), 2) for e in ("Ant-v5", "Humanoid-v5")], [2.01, 2.01])


    # ---- 5.4.1 CartPole fingerprint prose (declared order 1) ----------------------------------------------------------
    st = cs[cs.group == "stale"].set_index(["attribute", "eps"])
    check("5.4.1", "stale masspole +1e-3/+1e-2 consistent 100; +0.1 model 64; +0.5 model 100",
          (cls(st.loc[("masspole", 0.001), "classes"]), cls(st.loc[("masspole", 0.01), "classes"]), cls(st.loc[("masspole", 0.1), "classes"]),
           cls(st.loc[("masspole", 0.5), "classes"])), ({"consistent": 100}, {"consistent": 100}, {"model": 64, "consistent": 36}, {"model": 100}))
    check("5.4.1", "updated control consistent 100 in all four rows", all(cls(c) == {"consistent": 100} for c in cs[cs.group == "updated_control"].classes), True)
    dv = R("stale_constants_divergence").set_index(["attribute", "factor"])
    check("5.4.1", "median max pole-angle difference, masscart x0.5/x1.5/x2",
          tuple(round(float(dv.loc[("masscart", f), "median_max_dtheta_rad"]), 3) for f in (0.5, 1.5, 2.0)), (0.115, 0.041, 0.062))

    # ---- 5.4.3 trajectory-level effect -------------------------------------------------------------------------------
    te = R("mujoco_trajectory_error")
    a1 = te[(te.environment == "Ant-v5") & (te.level == 1)].iloc[0]
    h1 = te[(te.environment == "Humanoid-v5") & (te.level == 1)].iloc[0]
    check("5.4.3", "median position error at shipped step: Ant 2.4e-6, Humanoid 6.3e-7",
          (round(a1.median_max_body_position_error_m * 1e6, 1), round(h1.median_max_body_position_error_m * 1e7, 1)), (2.4, 6.3))
    check("5.4.3", "Ant convergence slopes 2.00, 2.02, 2.07",
          tuple(round(float(x), 2) for x in te[te.environment == "Ant-v5"].convergence_slope_to_next.dropna()), (2.0, 2.02, 2.07))
    check("5.4.3", "Hopper and Walker2d at round-off (<2e-13 m)", bool(te[te.environment.isin(["Hopper-v5", "Walker2d-v5"])].median_max_body_position_error_m.max() < 2e-13), True)
    check("5.4.3", "actual Ant/Humanoid trajectory horizons", (float(a1.horizon_s), float(h1.horizon_s)), (.5, .501))
    check("5.4.3", "all trajectory controls retain eight valid starts", set(te.n_valid), {8})
    velocity = R("quaternion_mechanism_local_error")
    gravity = velocity[velocity.model == "heavy top (gravity)"].omega_err_median.to_numpy()
    torque_free = velocity[velocity.model == "torque-free asymmetric body"].omega_err_median.to_numpy()
    check("5.4.3", "gravity-case velocity local error supports condition V",
          bool(np.all((np.log2(gravity[:-1]/gravity[1:]) > 3.9) & (np.log2(gravity[:-1]/gravity[1:]) < 4.3))), True)
    check("5.4.3", "torque-free velocity endpoint has local order five",
          bool(np.all((np.log2(torque_free[:-1]/torque_free[1:]) > 4.9) & (np.log2(torque_free[:-1]/torque_free[1:]) < 5.2))), True)

    # ---- App. B contacts ----------------------------------------------------------------------------------------------
    gt = R("regime_gated_fingerprint")
    check("App. B", "ungated: Hopper/Walker2d/Ant all rate-guard abstentions", all(cls(r.ungated) == {"unstable": 24} for _, r in gt.iterrows() if not r.environment.startswith("HalfCheetah")), True)
    check("App. B", "gated: no model label on Hopper/Walker2d", all("model" not in r.gated for _, r in gt.iterrows() if r.environment in ("Hopper-v5", "Walker2d-v5")), True)
    ant_gate = gt[gt.environment == "Ant-v5"].iloc[0]
    check("App B", "corrected Ant vertical perturbation: order-2 / insufficient steps",
          cls(ant_gate.gated), {"order-2": 10, "no-admissible-steps": 14})
    check("App B", "corrected Ant admissible-step coverage (percent)",
          round(float(ant_gate.admissible_fraction)*100, 1), 31.5)
    healthy_cd = R("shipped_acrobot_calibrated_detectors")
    healthy_cd = healthy_cd[healthy_cd.case == "healthy"].iloc[0]
    check("5.1.1", "fixed threshold / fingerprint healthy denominators",
          (int(healthy_cd.n), int(healthy_cd.n_fingerprint)), (100, 100))

    # ---- 5.2.2 reference-based comparison ----------------------------------------------------------------------------------------------
    hd = R("head_to_head_predecessor_screen").set_index("dt")
    mc = R("matched_step_control").set_index("dt")
    check("5.2.2", "screen floor at dt (1e-8 abs): 0.2, 0.1, 0.05", tuple(hd.loc[d, "floor_screen(1e-8)"] for d in (0.2, 0.1, 0.05)), (0.1, 0.01, 1e-4))
    check("5.2.2", "refinement 8 sub-steps floor at 0.2, 0.1, 0.05", tuple(hd.loc[d, "floor_sub8"] for d in (0.2, 0.1, 0.05)), (1e-5, 1e-7, 1e-8))
    check("5.2.2", "screen at dt/8 matches (0.2, 0.1, 0.05)", tuple(mc.loc[d, "screen_at_dt/8"] for d in (0.2, 0.1, 0.05)), (1e-5, 1e-7, 1e-8))
    check("5.2.2", "direct step compare floor 1e-13 at 0.2, 0.1, 0.05", tuple(mc.loc[d, "direct_reference_step_compare"] for d in (0.2, 0.1, 0.05)), (1e-13,) * 3)
    mcn = R("matched_compute_noise_fingerprint")
    h4 = mcn[(mcn.iloc[:, 0] == "H4") & (mcn.iloc[:, 3] == "shipped_vs_analytic_step")]
    check("5.1.1", "shipped step vs analytic RK4 step 2.8e-17", round(float(h4.iloc[0, 4]) * 1e17, 1), 2.8)

    # ---- correction-specific protocol and denominator checks ----
    for name, expected in (("CAT m2=1.3", {"model": 25, "unstable": 5}),
                           ("CAT l1=1.2", {"model": 28, "unstable": 2}),
                           ("CAT lc2=0.6", {"model": 27, "unstable": 3}),
                           ("CAT I=1.5 (both links)", {"model": 26, "unstable": 4}),
                           ("CAT nips (Coriolis dropped)", {"model": 19, "unstable": 11}),
                           ("euler + m2 +1", {"unstable": 30})):
        check("5.1.1", name + " complete guard", get(name), expected)
    repeated = R("repeated_start_summary")
    check("5.3.4", "40 independent starts per population/endpoint", set(repeated.n_starts), {40})
    transfer_start = repeated[(repeated.dataset == "transfer") & (repeated.event == "any_flag")]
    check("5.3.1", "transfer start-level any-flag counts and configurations",
          (int(transfer_start.successes.sum()), int(transfer_start.n_starts.sum()), set(transfer_start.n_configs)), (0, 120, {12}))
    check("5.3.1", "start-level Wilson upper bound rounds to 8.8 percent",
          set((100*transfer_start.wilson_hi).round(1)), {8.8})
    starts = R("transfer_trials")
    healthy = starts[starts.eps == 0]
    check("5.3.1", "healthy consistent/abstained evaluations",
          (int((healthy.cls == "consistent").sum()), int(healthy.cls.isin(["undetermined", "unstable", "blowup"]).sum())), (1301, 139))
    for env, expected in (("Ant-v5", {"order-2": 70, "undetermined": 2}),
                          ("Humanoid-v5", {"order-2": 71, "undetermined": 1})):
        group = sd[sd.environment == env]
        counts = {label: count(group, label) for label in expected}
        check("5.4.2", env + " native-step seed coverage", counts, expected)
    native_trials = R("mujoco_audit_shipped_step_trials")
    check("5.4.2", "K=1 is native timestep, not frame_skip*timestep",
          bool(np.allclose(native_trials.macro_h, native_trials.native_timestep, rtol=0, atol=0)), True)
    audit_trials = R("mujoco_integrator_order_trials")
    check("5.4.2", "same initial velocities at each enlarged step",
          bool((audit_trials.groupby(["environment", "seed", "start"]).initial_velocity.nunique() == 1).all()), True)

    # Same-call self-convergence control, added after the main validation.
    sc = R("self_convergence_summary")
    st = R("self_convergence_trials")
    ss = R("self_convergence_steps")
    check("5.2.1", "self-convergence full grid and smooth starts",
          (len(sc), len(st), len(ss), set(sc.n_starts), set(sc.n_smooth), set(sc.n_state_resolved)),
          (12, 480, 9600, {40}, {40}, {40}))
    check("5.2.1", "self-convergence matched starts in every cell",
          bool((st.groupby("start").initial_state.nunique() == 1).all()), True)
    for name, expected in (
        ("healthy RK4", ({"consistent": 40}, {"consistent": 40})),
        ("RK4 mass +0.1%", ({"model": 40}, {"model": 40})),
        ("RK4 Coriolis omission", ({"model": 27, "unstable": 13}, {"model": 36, "unstable": 4})),
        ("midpoint RK2", ({"order-2": 35, "order-3": 4, "model": 1},
                          {"order-2": 37, "order-3": 1, "model": 2})),
        ("Euler", ({"order-1": 40}, {"order-1": 40})),
        ("Euler mass +1%", ({"order-1": 40}, {"order-1": 40})),
    ):
        for h, counts in zip((.2, .05), expected):
            row = sc[(sc["case"] == name) & (sc.h == h)].iloc[0]
            check("5.2.1", f"self-convergence energy labels {name} h={h}", cls(row.classes), counts)
            wanted = 4 if name.startswith("RK4") or name == "healthy RK4" else (2 if name == "midpoint RK2" else 1)
            check("5.2.1", f"state order {name} h={h}", float(row.median_p24), wanted, tol=.1)
    group_keys = ["case", "h", "start"]
    averages = ss.groupby(group_keys)[["r1", "r2", "r4", "r8", "d1", "d2", "d4"]].mean()
    saved = st.set_index(group_keys).loc[averages.index]
    check("5.2.1", "per-step means reproduce stored diagnostics",
          bool(np.allclose(averages.to_numpy(), saved[averages.columns].to_numpy(), rtol=1e-12, atol=1e-15)), True)
    check("5.2.1", "saved state exponents use consecutive endpoint differences",
          bool(np.allclose(np.log2(saved.d2/saved.d4), saved.p24, rtol=1e-12, atol=1e-12)), True)

    paper = (RES.parent / "doc/DAFD-paper.tex").read_text()
    for location, fragment in (
        ("abstract", r"\ge86.7\%"),
        ("abstract", r"$1/30$--$14/30$"),
        ("abstract", r"$2/40$ healthy starts using fourth-order Runge--Kutta integration at a timestep of $0.2$ seconds receive"),
        ("5.1.1", r"median $2.22\times10^{-16}$ and maximum"),
        ("5.1.1", r"$8.88\times10^{-16}$"),
        ("5.3.3", r"fraction whose rounded exponent estimate equals 4"),
        ("5.3.3", r"from $955/1000$ to $986/1000$"),
    ):
        check(location, "prose includes " + fragment, fragment in " ".join(paper.split()), True)

    # ---- appendices ------------------------------------------------------------------------------------------------
    fp = R("parameter_fault_sweep_summary")
    print(f"  info [App C] parameter_fault_sweep_summary rows: {len(fp)}")

    print(f"\n{TOTAL - len(FAILED)}/{TOTAL} claims match")
    if FAILED:
        print("\nFAILED:")
        for f in FAILED:
            print("  -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
