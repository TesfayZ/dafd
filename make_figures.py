#!/usr/bin/env python3
"""Regenerate the paper's figures into doc/fig/ (needs Results/ from run_all_experiments.py)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
from pathlib import Path
import acrobot_mpmath as H
import fingerprint as F

OUT = Path(__file__).resolve().parent / "doc" / "fig"
RES = Path(__file__).resolve().parent / "Results"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 150})


def fig_sign():
    mp.mp.dps = 40
    st = [mp.mpf("0.1"), mp.mpf("-0.2"), mp.mpf("0.3"), mp.mpf("-0.15")]
    hs = np.logspace(np.log10(0.2), -4, 60)
    v = np.array([float(H.local_residual(st, mp.mpf(float(h))) / mp.mpf(float(h)) ** 4) for h in hs])
    ar = v * hs ** 4
    # Retain the dense curve actually plotted, in addition to the benchmark grid.
    pd.DataFrame(dict(h=hs, signed_r=ar, r_over_h4=v)).to_csv(RES / "figure_sign_data.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax[0].semilogx(hs, v, "k-"); ax[0].axhline(0, color="gray", lw=.5); ax[0].axhline(0.01225, color="C3", ls="--", lw=.8)
    ax[0].set_xlabel("step size $h$"); ax[0].set_ylabel(r"$r(h)/h^4$")
    ax[0].text(.12, .82, r"$c_4\approx+0.0123$", color="C3", transform=ax[0].transAxes)
    ax[0].set_title("(a) signed RK4 energy defect", fontsize=9)
    ax[1].loglog(hs[v > 0], np.abs(ar[v > 0]), "C0.", label="$r>0$"); ax[1].loglog(hs[v < 0], np.abs(ar[v < 0]), "C3.", label="$r<0$")
    ax[1].loglog(hs, 0.01225 * hs ** 4, "k:", lw=.8, label=r"$c_4h^4$"); ax[1].set_xlabel("step size $h$"); ax[1].set_ylabel("$|r(h)|$")
    ax[1].legend(frameon=False, fontsize=7); ax[1].set_title("(b) sign change near $h\\approx0.011$", fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "fig_sign.pdf", metadata={"CreationDate": None, "ModDate": None}); plt.close()


def fig_fingerprint():
    d = pd.read_csv(RES / "shipped_acrobot_fingerprint_trials.csv").dropna(subset=["r1"])
    d["a"], d["b"], d["c"] = zip(*[F.exponents(r.r1, r.r2, r.r4, r.r8) for r in d.itertuples()])
    groups = [("healthy RK4", d.case == "healthy", "C0", "o"),
              ("model faults, $\\varepsilon\\in[10^{-5},10^{-2}]$", d.case.str.startswith("LINK_"), "C3", "x"),
              ("catalogue model faults", d.case.str.startswith("CAT") & ~d.case.str.contains("euler"), "C1", "+"),
              ("Euler (order 1)", d.case == "CAT euler", "C2", "s"), ("RK2 (order 2)", d.case == "rk2", "C4", "^")]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for lab, m, c, mk in groups:
        ax[0].plot(d.a[m], d.c[m], mk, color=c, ms=3.5, alpha=.7, label=lab)
    # Include every selected point. The former fixed c >= -0.3 limit clipped one.
    selected = np.logical_or.reduce([mask.to_numpy() for _, mask, _, _ in groups])
    points = d.loc[selected, ["a", "c"]].to_numpy()
    lower = min(-.3, float(np.min(points))-.1)
    upper = max(5.3, float(np.max(points))+.1)
    ax[0].plot([0, 5], [0, 5], "k:", lw=.6); ax[0].set_xlim(lower, upper); ax[0].set_ylim(lower, upper)
    ax[0].set_xlabel(r"$a=\log_2(r^{(1)}/r^{(2)})$"); ax[0].set_ylabel(r"$c=\log_2(r^{(4)}/r^{(8)})$")
    ax[0].legend(frameon=False, fontsize=5.8, loc="upper left"); ax[0].set_title("(a) shipped Acrobot, random starts", fontsize=8)
    eps = [1e-5, 1e-4, 1e-3, 1e-2]
    for a_, col in zip(["LINK_MASS_1", "LINK_MASS_2", "LINK_LENGTH_1", "LINK_COM_POS_1", "LINK_COM_POS_2", "LINK_MOI"], plt.cm.tab10.colors):
        fr = [np.mean(d[d.case == f"{a_} +{e:g}"].cls == "model") for e in eps]
        ax[1].semilogx(eps, fr, "-o", ms=3, color=col, label=a_.replace("LINK_", ""))
    ax[1].set_xlabel(r"relative parameter error $\varepsilon$"); ax[1].set_ylabel('fraction classified "model"')
    ax[1].legend(frameon=False, fontsize=5.8); ax[1].set_title("(b) class-level detection", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "fig_fingerprint2.pdf", metadata={"CreationDate": None, "ModDate": None}); plt.close()


if __name__ == "__main__":
    fig_sign(); print("wrote doc/fig/fig_sign.pdf")
    if (RES / "shipped_acrobot_fingerprint_trials.csv").exists():
        fig_fingerprint(); print("wrote doc/fig/fig_fingerprint2.pdf")
    else:
        print("run benchmark_shipped_acrobot_fingerprint.py first for fig_fingerprint2.pdf")
