"""Calibration-free fingerprint classifier on the shipped Gymnasium AcrobotEnv (dt = 0.2).
Paper: Section 5.2.
Outputs: Results/shipped_acrobot_fingerprint_summary.csv (class counts per case),
         Results/shipped_acrobot_fingerprint_trials.csv (per-trial mean |r^(m)| and class)."""
import numpy as np
import zlib
import shipped_acrobot as I
from shipped_acrobot import case
import fingerprint as F
from results_io import save_csv, n_trials

N_H, N_F = n_trials(150, 20), n_trials(30, 6)

def run(kw, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    with case(dict(kw)):
        for s0 in I.sample_states(rng, n):
            E0, Ee = I.run_trial(s0)
            if not (np.all(np.isfinite(E0)) and all(np.all(np.isfinite(v)) for v in Ee.values())):
                out.append(None); continue
            r = {m: float(np.mean(np.abs((Ee[m] - E0) / I.H))) for m in I.MS}
            r["E"] = float(np.mean(np.abs(E0)))             # mean |E| of the trajectory, for the round-off floor
            out.append(r)
    return out

def label(r):
    """Definition 1 with the float64 round-off guard (it never binds here: the finest residuals are > 1e6 x phi_round)."""
    return "blowup" if r is None else F.guarded_classify(r[1], r[2], r[4], r[8], 4, r["E"], I.H)


def summarize(res):
    cls = {}
    for r in res:
        c = label(r)
        cls[c] = cls.get(c, 0) + 1
    return cls

def main():
    cases = [("healthy", dict(), N_H, 1)]
    for a in ["LINK_MASS_1", "LINK_MASS_2", "LINK_LENGTH_1", "LINK_COM_POS_1", "LINK_COM_POS_2", "LINK_MOI"]:
        for e in [1e-5, 1e-4, 1e-3, 1e-2]:
            seed = zlib.crc32(f"{a}|{e:g}".encode("utf-8")) % 10**6
            cases.append((f"{a} +{e:g}", dict(attr=a, eps=e), N_F, seed))
    cases += [("CAT m2=1.3", dict(attr="LINK_MASS_2", eps=0.3), N_F, 11),
              ("CAT l1=1.2", dict(attr="LINK_LENGTH_1", eps=0.2), N_F, 12),
              ("CAT lc2=0.6", dict(attr="LINK_COM_POS_2", eps=0.2), N_F, 13),
              ("CAT I=1.5 (both links)", dict(attr="LINK_MOI", eps=0.5), N_F, 14),
              ("CAT nips (Coriolis dropped)", dict(nips=True), N_F, 15),
              ("CAT euler", dict(integ="euler"), N_F, 16),
              ("rk2", dict(integ="rk2"), N_F, 17),
              ("euler + m2 +1e-2", dict(integ="euler", attr="LINK_MASS_2", eps=1e-2), N_F, 18),
              ("euler + m2 +1", dict(integ="euler", attr="LINK_MASS_2", eps=1.0), N_F, 19),
              ("rk2 + m2 +1e-2", dict(integ="rk2", attr="LINK_MASS_2", eps=1e-2), N_F, 20),
              ("rk2 + m2 +1e-1", dict(integ="rk2", attr="LINK_MASS_2", eps=1e-1), N_F, 21)]
    store = {}
    print(f"{'case':<30} n  classes", flush=True)
    for name, kw, n, seed in cases:
        res = run(kw, n, seed)
        store[name] = res
        print(f"{name:<30} {n:<3}{summarize(res)}", flush=True)
    summ, trials = [], []
    for name, res in store.items():
        cl = summarize(res)
        summ.append(dict(case=name, n=len(res), classes=str(cl), n_model=cl.get("model", 0), n_consistent=cl.get("consistent", 0)))
        for k, r in enumerate(res):
            if r is None:
                trials.append(dict(case=name, trial=k, r1=None, r2=None, r4=None, r8=None, E_abs_mean=None, guard_margin=None, cls="blowup")); continue
            trials.append(dict(case=name, trial=k, r1=r[1], r2=r[2], r4=r[4], r8=r[8], E_abs_mean=r["E"],
                               guard_margin=r[8] / F.roundoff_floor(r["E"], I.H), cls=label(r)))
    save_csv("shipped_acrobot_fingerprint_summary.csv", summ)
    save_csv("shipped_acrobot_fingerprint_trials.csv", trials)
    a = np.array([F.exponents(r[1], r[2], r[4], r[8]) for r in store["healthy"] if r])
    print("healthy exponents (a,b,c) min/median/max:", a.min(0).round(2), np.median(a, 0).round(2), a.max(0).round(2))

if __name__ == "__main__":
    main()
