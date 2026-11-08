"""(a) false alarms on 1000 healthy shipped-Acrobot runs, (b) class-level detection floor, (c) energy-evaluation
noise with and without the noise-floor guard.  Paper: Section 5.2.
Outputs: Results/false_alarm_healthy.csv, Results/class_level_floor.csv, Results/noise_guard.csv"""
import numpy as np
import shipped_acrobot as I
from shipped_acrobot import case
import fingerprint as F
from results_io import save_csv, n_trials

def trial_arrays(kw, n, seed):
    rng = np.random.default_rng(seed); out = []
    with case(dict(kw)):
        for s0 in I.sample_states(rng, n):
            out.append(I.run_trial(s0))
    return out

def cls_from_arrays(E0, Ee, sigma=0.0, rng=None, guard=True):
    if sigma > 0:
        E0 = E0 + sigma*rng.standard_normal(E0.shape)
        Ee = {m: v + sigma*rng.standard_normal(v.shape) for m, v in Ee.items()}
    r = {m: float(np.mean(np.abs((Ee[m]-E0)/I.H))) for m in I.MS}
    floor = 5*np.sqrt(2/np.pi)*np.sqrt(2)*sigma/I.H if (guard and sigma > 0) else 0.0
    if not guard:  # deliberate no-floor control
        return F.classify(r[1], r[2], r[4], r[8], 4)
    return F.guarded_classify(r[1], r[2], r[4], r[8], 4, np.mean(np.abs(E0)), I.H, noise_floor=floor)

def tally(cs):
    d = {}
    for c in cs: d[c] = d.get(c, 0)+1
    return d

def main():
    FA, FL, NG = [], [], []
    print("(a) healthy false alarms", flush=True)
    rng = np.random.default_rng(777); cs = []
    with case({}):
        for s0 in I.sample_states(rng, n_trials(1000, 40)):
            E0, Ee = I.run_trial(s0); cs.append(cls_from_arrays(E0, Ee))
    t = tally(cs); print("  healthy:", t, flush=True)
    nh = sum(t.values()); k = nh - t.get("consistent", 0)
    print("  false alarm rate %.4f; one-sided 95%% upper bound (rule of three) %.4f" % (k/nh, 3/nh if k == 0 else (k+2)/nh))
    FA.append(dict(n_healthy=nh, classes=str(t), n_flagged=k, false_alarm_rate=k/nh, upper_95_rule_of_three=(3/nh if k == 0 else (k+2)/nh)))
    print("(b) class-level floor, 30 trials each", flush=True)
    for a in ["LINK_MASS_2", "LINK_COM_POS_1", "LINK_MOI", "LINK_LENGTH_1"]:
        for e in [1e-9, 1e-8, 1e-7, 1e-6, 3e-6]:
            tr = trial_arrays(dict(attr=a, eps=e), n_trials(30, 6), 5)
            tl = tally([cls_from_arrays(*x) for x in tr])
            print(f"  {a:<15} {e:7.0e}", tl, flush=True)
            FL.append(dict(attribute=a, eps=e, n=len(tr), classes=str(tl), n_model=tl.get("model", 0)))
    print("(c) noise on energy evaluations, guard on/off", flush=True)
    nn, nm = n_trials(60, 12), n_trials(30, 6)
    sets = {"healthy": trial_arrays({}, nn, 9),
            "m2+1e-2 (model)": trial_arrays(dict(attr="LINK_MASS_2", eps=1e-2), nm, 9),
            "m2+1e-4 (model)": trial_arrays(dict(attr="LINK_MASS_2", eps=1e-4), nm, 9),
            "euler": trial_arrays(dict(integ="euler"), nm, 9)}
    for nm, tr in sets.items():
        for sg in [0.0, 1e-10, 1e-8, 1e-6, 1e-4]:
            rng = np.random.default_rng(3)
            ng = tally([cls_from_arrays(*x, sg, rng, guard=False) for x in tr])
            rng = np.random.default_rng(3)
            g = tally([cls_from_arrays(*x, sg, rng, guard=True) for x in tr])
            print(f"  {nm:<18} sigma_E={sg:7.0e}  unguarded {ng}   guarded {g}", flush=True)
            NG.append(dict(case=nm, sigma_E=sg, n=len(tr), unguarded=str(ng), guarded=str(g)))
    save_csv("false_alarm_healthy.csv", FA)
    save_csv("class_level_floor.csv", FL)
    save_csv("noise_guard.csv", NG)

if __name__ == "__main__":
    main()
