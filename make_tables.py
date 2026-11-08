#!/usr/bin/env python3
"""Generate every results-derived LaTeX table of the paper from Results/*.csv into doc/tables/ (no hand transcription).

Tables written: noise_sources, finite_step, transfer, transfer_by_step, repeated_starts,
sensitivity, mujoco_audit, quaternion_controls, mechanism, mechanism_conv, contact_gate, predecessor.
Intervals use independent starts within a configuration or a stated start-level endpoint;
pooled repeated-start evaluation rates are descriptive.
"""
import ast
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES, OUT = ROOT / "Results", ROOT / "doc" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - w), min(1.0, c + w)


def prob(x):
    return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def sig(x):
    return "0" if x == 0 else f"$10^{{{int(round(np.log10(x)))}}}$"


def cls(s):
    """Parse the stored class-count dict."""
    return ast.literal_eval(s) if isinstance(s, str) else {}


def fmt_cls(d):
    """'consistent 9, undetermined 3' (largest first)."""
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")


def sci(x):
    """LaTeX scientific notation."""
    m, e = f"{x:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


# ----------------------------------------------------------------------------------------------------------------
def noise_sources():
    d = pd.read_csv(RES / "noise_sources_operating_envelope.csv")
    names = {"float64": "float64 (no added noise)", "start_state_noise": "start-state noise",
             "readout_state_noise": "readout state noise", "energy_noise": "energy-evaluation noise",
             "float32_readout": "float32 readout"}
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Source & $\sigma$ & healthy flagged / undet. & model $\varepsilon{=}10^{-4}$ & $10^{-3}$ & $10^{-2}$ \\", r"\midrule"]
    for _, r in d.iterrows():
        cells = []
        for e in ("0.0001", "0.001", "0.01"):
            cells.append(f"{r[f'eps{e}_model_rate']:.2f} ({r[f'eps{e}_undetermined_rate']:.2f})")
        s = "--" if r["source"] in ("float64", "float32_readout") else sig(r["sigma"])
        L.append(f"{names[r['source']]} & {s} & {r['healthy_false_alarm']:.2f} / {r['healthy_undetermined']:.2f} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("noise_sources.tex", L)


def finite_step():
    d = pd.read_csv(RES / "finite_step_validation.csv").drop_duplicates(subset=["start_set", "true_order", "h", "eps"], keep="first")
    h = d[d.start_set == "held_out"]
    L = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
         r"Order $q$ & $h$ & healthy flagged & model rate $\varepsilon{=}10^{-4}$ & $10^{-3}$ & $10^{-2}$ & $10^{-1}$ \\", r"\midrule"]
    for q in (1, 2, 3, 4):
        for hh in (0.05, 0.1, 0.2):
            cells = []
            for e in (1e-4, 1e-3, 1e-2, 1e-1):
                r = h[(h.true_order == q) & np.isclose(h.h, hh) & np.isclose(h.eps, e)].iloc[0]
                cells.append(f"{prob(r.model_rate)} [{prob(r.model_rate_lo)},{prob(r.model_rate_hi)}]")
            r0 = h[(h.true_order == q) & np.isclose(h.h, hh) & np.isclose(h.eps, 0.0)].iloc[0]
            L.append(f"{q} & {hh} & {int(round(r0.flagged_rate * r0.n))}/{int(r0.n)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("finite_step.tex", L)


def transfer():
    d = pd.read_csv(RES / "transfer_validation.csv")
    nice = {"henon_heiles": r"H\'enon--Heiles", "kepler": "Kepler", "euler_top": "Euler top"}
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"System & $q$ & healthy flagged & healthy abstained & model $\varepsilon{=}10^{-4}$ & $10^{-3}$ & $10^{-2}$ \\", r"\midrule"]
    for sysn in ("henon_heiles", "kepler", "euler_top"):
        for q in (1, 2, 3, 4):
            sub = d[(d.system == sysn) & (d.true_order == q)]
            h0 = sub[sub.eps == 0.0]
            n0 = int(h0.n.sum()); fl = int(h0.flagged.sum())
            ab = sum(v for c in h0.classes for k, v in cls(c).items() if k in ("undetermined", "unstable", "blowup"))
            cells = []
            for e in (1e-4, 1e-3, 1e-2):
                s = sub[np.isclose(sub.eps, e)]
                k = int(round((s.model_rate * s.n).sum())); n = int(s.n.sum())
                cells.append(f"{k / n:.2f}")
            L.append(f"{nice[sysn] if q == 1 else ''} & {q} & {fl}/{n0} & {ab}/{n0} & " + " & ".join(cells) + r" \\")
        if sysn != "euler_top":
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("transfer.tex", L)


def transfer_by_step():
    d = pd.read_csv(RES / "transfer_validation.csv")
    names = {"henon_heiles": r"H\'enon--Heiles", "kepler": "Kepler", "euler_top": "Euler top"}
    lines = []
    for system, name in names.items():
        lines += [r"\begin{table}[ht]", r"\centering\small",
                  r"\begin{tabular}{rrrrrrr}", r"\toprule",
                  r"$q$ & $h$ & healthy flagged & abstained & model $10^{-4}$ & $10^{-3}$ & $10^{-2}$ \\",
                  r"\midrule"]
        sub = d[d.system == system]
        for (q, h), group in sub.groupby(["true_order", "h"], sort=True):
            healthy = group[group.eps == 0].iloc[0]
            n = int(healthy.n)
            abstained = sum(v for k, v in cls(healthy.classes).items()
                            if k in ("undetermined", "unstable", "blowup"))
            cells = []
            for eps in (1e-4, 1e-3, 1e-2):
                row = group[np.isclose(group.eps, eps)].iloc[0]
                cells.append(f"{row.model_rate:.2f} [{row.model_lo:.2f},{row.model_hi:.2f}]")
            lines.append(f"{q} & {h:g} & {int(healthy.flagged)}/{n} & {abstained}/{n} & "
                         + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}",
                  rf"\caption{{{name}: transfer results at each step size. Model columns give rates and $95\%$ Wilson intervals; each configuration uses $40$ starts.}}",
                  r"\end{table}"]
    write("transfer_by_step.tex", lines)


def repeated_starts():
    d = pd.read_csv(RES / "repeated_start_summary.csv")
    names = {"development": "Acrobot development", "held_out": "Acrobot held-out",
             "henon_heiles": r"H\'enon--Heiles", "kepler": "Kepler", "euler_top": "Euler top"}
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Population & Start-level endpoint & $k/n$ starts & configurations & 95\% Wilson \\", r"\midrule"]
    for _, r in d.iterrows():
        endpoint = "any healthy fault flag" if r.event == "any_flag" else r"$1\%$ fault: all model"
        L.append(f"{names[r.population]} & {endpoint} & {int(r.successes)}/{int(r.n_starts)} & {int(r.n_configs)} & "
                 f"[{r.wilson_lo:.3f},{r.wilson_hi:.3f}] " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("repeated_starts.tex", L)


def sensitivity():
    d = pd.read_csv(RES / "classifier_sensitivity.csv")
    u = d[d.analysis == "unified"]
    if u.empty:
        return
    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{Acrobot RK4 (development)} & \multicolumn{5}{c}{Three transfer systems} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-8}",
         r"Variant & healthy model & faults $10^{-5}$ & healthy model & healthy other flag & healthy abstain & faults $10^{-2}$ & HH faults $10^{-4}$ \\", r"\midrule"]
    for _, r in u.iterrows():
        name = (r.setting.replace("theta = q - ", r"$\theta=q-").replace(" (all q)", r"$ (all $q$)").replace("c_thr = ", r"$c_{\mathrm{thr}}=")
                .replace("delta = ", r"$\delta=")) if r.setting != "default (as in the paper)" else r.setting
        if name.startswith("$") and not name.rstrip().endswith("$)") and not name.endswith("$"):
            name += "$"
        L.append(f"{name} & "
                 f"{int(r.dev_healthy_model)}/{int(r.dev_healthy_n)} & {int(r['dev_fault_1e-5_model'])}/{int(r['dev_fault_1e-5_n'])} & "
                 f"{int(r.transfer_healthy_model)}/{int(r.transfer_healthy_n)} & {int(r.transfer_healthy_other_flag)} & "
                 f"{int(r.transfer_healthy_abstain)} & {int(r['transfer_fault_1e-2_model'])}/{int(r['transfer_fault_1e-2_n'])} & "
                 f"{int(r['transfer_hh_fault_1e-4_model'])}/{int(r['transfer_hh_fault_1e-4_n'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("sensitivity.tex", L)


def mujoco_audit():
    d = pd.read_csv(RES / "mujoco_integrator_order_audit.csv")
    L = [r"\begin{tabular}{lcccl}", r"\toprule", r"Environment & Declared & Macro step $h$ & Median $(a,b,c)$ & Classes (12 starts) \\", r"\midrule"]
    for _, r in d.iterrows():
        env = r.environment.replace("-v5", "")
        row = f"{env} & {r.declared_integrator} & {r.macro_h:g} & $({r.median_a:.2f},{r.median_b:.2f},{r.median_c:.2f})$ & {fmt_cls(cls(r.classes))} \\\\"
        if "order-2" in r.classes:
            row = row.replace(env, rf"\textbf{{{env}}}", 1)
        L.append(row)
    L += [r"\bottomrule", r"\end{tabular}"]
    write("mujoco_audit.tex", L)


def quaternion_controls():
    d = pd.read_csv(RES / "quaternion_joint_order_control.csv")
    L = [r"\begin{tabular}{llcl}", r"\toprule", r"Model & Declared & Median $(a,b,c)$ & Classes (24 starts) \\", r"\midrule"]
    for _, r in d.iterrows():
        med = "--" if pd.isna(r.median_a) else f"$({r.median_a:.2f},{r.median_b:.2f},{r.median_c:.2f})$"
        L.append(f"{r.base_joints} & {r.declared_integrator} & {med} & {fmt_cls(cls(r.classes))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("quaternion_controls.tex", L)


def mechanism():
    e = pd.read_csv(RES / "quaternion_mechanism_local_error.csv")
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Model & $h$ & $e\,/\,(h^3\,\omega{\times}\dot\omega)$, median [range] & $\cos(e,-\omega{\times}\dot\omega)$ & $\lvert\Delta E\rvert$ \\", r"\midrule"]
    for _, r in e.iterrows():
        L.append(f"{r.model if r.h == 0.05 else ''} & {r.h:g} & ${r.coef_median:+.4f}$ $[{r.coef_min:+.4f},{r.coef_max:+.4f}]$ & "
                 f"${-r.cos_median:+.4f}$ & ${sci(r.energy_defect_median)}$ \\\\")
        if r.h == 0.00625 and r.model.startswith("heavy"):
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("mechanism.tex", L)

    c = pd.read_csv(RES / "quaternion_mechanism_convergence.csv")
    rep = pd.read_csv(RES / "quaternion_mechanism_replication.csv")
    names = {"MuJoCo": r"MuJoCo \texttt{RK4}", "replica": "NumPy replica of MuJoCo's update", "embedded RK4": "embedded RK4 + renormalisation",
             "CF4": "commutator-free Lie RK4 (CF4)", "CF4 reversed": "CF4 with the two exponentials in the wrong order"}
    fp = {"replica": "replica of MuJoCo RK4 update", "embedded RK4": "embedded RK4 + renormalisation", "CF4": "commutator-free Lie RK4 (CF4)"}
    L = [r"\begin{tabular}{lcc}", r"\toprule", r"Orientation update (same heavy top) & trajectory slopes & energy exponents $(a,b,c)$; class \\", r"\midrule"]
    for _, r in c.iterrows():
        sl = ", ".join(f"{x:.2f}" for x in ast.literal_eval(r.slopes))
        if r.scheme in fp:
            q = rep[rep.scheme == fp[r.scheme]].iloc[0]
            f = f"$({q.median_a:.2f},{q.median_b:.2f},{q.median_c:.2f})$; {fmt_cls(cls(q.classes))}"
        else:
            f = "(Table~\\ref{tab:mj-quat})" if r.scheme == "MuJoCo" else "--"
        L.append(f"{names[r.scheme]} & {sl} & {f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("mechanism_conv.tex", L)


def contact_gate():
    d = pd.read_csv(RES / "regime_gated_fingerprint.csv")
    L = [r"\begin{tabular}{lrcll}", r"\toprule", r"Environment (declared) & Macro $h$ & Admissible steps & Ungated (24 runs) & Gated \\", r"\midrule"]
    for _, r in d.iterrows():
        env = r.environment.replace("-v5", "")
        integ = "RK4" if r.declared_order == 4 else "Euler"
        L.append(f"{env} ({integ}) & {r.macro_h:g} & {100 * r.admissible_fraction:.0f}\\% & {fmt_cls(cls(r.ungated))} & "
                 f"{fmt_cls(cls(r.gated)).replace('no-admissible-steps', 'none')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("contact_gate.tex", L)


def self_convergence():
    d = pd.read_csv(RES / "self_convergence_summary.csv")
    L = [r"\begin{tabular}{lrcll}", r"\toprule",
         r"Candidate (declared $q=4$) & $h$ & Smooth / starts & State $p_{24}$ [10th, 90th] & Energy labels \\",
         r"\midrule"]
    for _, r in d.iterrows():
        name = r["case"].replace("%", r"\%")
        L.append(f"{name} & {r.h:g} & {r.n_smooth}/{r.n_starts} & "
                 f"{r.median_p24:.2f} [{r.p10_p24:.2f}, {r.p90_p24:.2f}] & "
                 f"{fmt_cls(cls(r.classes))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("self_convergence.tex", L)


def diagnostic_outcomes():
    """Keep fault attribution, order flags and abstention distinct."""
    d = pd.read_csv(RES / "self_convergence_summary.csv")
    rows = [(f"{r['case']} ($h={r.h:g}$)", cls(r.classes), 4)
            for _, r in d.iterrows() if r["case"] in
            ("midpoint RK2", "Euler mass +1%")]
    finite = pd.read_csv(RES / "finite_step_validation.csv")
    held = finite[(finite.start_set == "held_out") & (finite.true_order == 4)
                  & np.isclose(finite.h, .2) & (finite.eps == 0)].iloc[0]
    assert len(rows) == 4, "Expected both steps for midpoint and mixed Euler cases"
    rows.insert(0, ("healthy RK4, held-out ($h=0.2$)", cls(held.classes), 4))
    transfer = pd.read_csv(RES / "transfer_validation.csv")
    counts = {}
    for value in transfer[transfer.eps == 0].classes:
        for label, n in cls(value).items():
            counts[label] = counts.get(label, 0) + n
    # No order labels occur in this aggregate, so declared orders need no pooling.
    assert not any(label.startswith("order-") for label in counts)
    rows.append(("three transfer systems (pooled)", counts, 4))
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Candidate & consistent & model & order mismatch & order-$q$ & abstained & coverage \\",
         r"\midrule"]
    for name, counts, q in rows:
        n = sum(counts.values())
        abstain = sum(counts.get(k, 0) for k in ("undetermined", "unstable", "blowup"))
        mismatch = sum(v for k, v in counts.items() if k.startswith("order-") and k != f"order-{q}")
        cells = (counts.get("consistent", 0), counts.get("model", 0), mismatch,
                 counts.get(f"order-{q}", 0), abstain)
        L.append(name.replace("%", r"\%") + " & " + " & ".join(f"{k}/{n}" for k in cells)
                 + f" & {100*(n-abstain)/n:.1f}\\% " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("diagnostic_outcomes.tex", L)


def extrapolation_comparison():
    d = pd.read_csv(RES / "shipped_acrobot_calibrated_detectors.csv")
    healthy = d[d.case == "healthy"].iloc[0]
    assert healthy.n == healthy.n_fingerprint
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Candidate & $n$ & drift flag & raw flag & sub8 flag & $R_2$ flag & fingerprint model \\",
         r"\midrule"]
    selected = [("healthy (held-out)", healthy)]
    for eps in (1e-6, 1e-5, 1e-4):
        row = d[(d.case == "LINK_MASS_1") & np.isclose(d.eps, eps)].iloc[0]
        selected.append((rf"mass 1 $+10^{{{int(np.log10(eps))}}}$", row))
    for case in ("euler", "rk2", "euler+MASS_2"):
        selected.append(({"euler": "Euler", "rk2": "midpoint RK2",
                          "euler+MASS_2": r"Euler + mass 2 $+1\%$"}[case],
                         d[d.case == case].iloc[0]))
    for name, row in selected:
        L.append(f"{name} & {int(row.n)} & " + " & ".join(str(int(row[f'det_{key}']))
                 for key in ("drift", "raw", "sub8", "R2", "fingerprint")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extrapolation_comparison.tex", L)


def predecessor():
    h = pd.read_csv(RES / "head_to_head_predecessor_screen.csv")
    m = pd.read_csv(RES / "matched_step_control.csv")
    L = [r"\begin{tabular}{rcccccccc}", r"\toprule",
         r" & \multicolumn{4}{c}{reference-based screen, simulator step} & \multicolumn{2}{c}{refinement at $\Delta t$} & & \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
         r"$\Delta t$ & $\Delta t$ & $\Delta t/2$ & $\Delta t/4$ & $\Delta t/8$ & 8 sub-steps & $R_2$ & raw $r^{(1)}$ & \shortstack{direct reference\\step comparison} \\", r"\midrule"]
    for _, r in h.iterrows():
        mm = m[np.isclose(m["dt"], r["dt"])]
        if len(mm):
            q = mm.iloc[0]
            mid = [sig(q["screen_at_dt/2"]), sig(q["screen_at_dt/4"]), sig(q["screen_at_dt/8"])]
            direct = sig(q["direct_reference_step_compare"])
        else:
            mid, direct = ["--"] * 3, "--"
        L.append(f"{r["dt"]:g} & {sig(r['floor_screen(1e-8)'])} & " + " & ".join(mid) + f" & {sig(r.floor_sub8)} & {sig(r.floor_R2)} & {sig(r.floor_raw)} & {direct} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("predecessor.tex", L)


if __name__ == "__main__":
    for fn in (noise_sources, finite_step, transfer, transfer_by_step, repeated_starts, sensitivity, mujoco_audit, quaternion_controls, mechanism,
               contact_gate, predecessor, self_convergence, diagnostic_outcomes, extrapolation_comparison):
        try:
            fn()
            print("  wrote", fn.__name__)
        except FileNotFoundError as e:
            print("  skipped", fn.__name__, "-", e)
    print("tables in doc/tables/")
