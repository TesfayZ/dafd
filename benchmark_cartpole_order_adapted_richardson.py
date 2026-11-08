"""CartPole (order 1): do order-adapted Richardson weights lower the detection floor?
Stale-constants masspole fault tested against declared parameters, 8 paired ICs, same exact
permutation test as the head-to-head benchmark (alpha=0.01).  Paper: Appendix D (CartPole, order 1).
Output: Results/cartpole_order_adapted_richardson.csv  (requires the implementation of Gebrekidan and Gebrerufael (2026): see predecessor.py)"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import shipped_cartpole as J
import fingerprint as F
import predecessor
from results_io import save_csv
_, ideal_diff = predecessor.require()
W = F.richardson_weights((1, 2, 4, 8), 1, 3)
print("weights (m=1,2,4,8; q=1, kill orders 1,2,3):", W.round(4))
ROWS = []
rng = np.random.default_rng(11)
ICs = [[rng.uniform(-.5,.5), rng.uniform(-.5,.5), rng.uniform(-.3,.3), rng.uniform(-.5,.5)] for _ in range(8)]

def stats(kind, name, eps):
    decl = J.set_params(kind, name, eps)
    E = lambda s: J.energy(s, decl["masscart"], decl["masspole"], decl["length"])
    out = []
    for s0 in ICs:
        E0, Ee = J.trial(s0, E, "euler")
        r = {m: (Ee[m]-E0)/J.H for m in J.MS}
        Rq = sum(W[i]*r[m] for i, m in enumerate(J.MS))
        out.append(dict(raw=np.mean(np.abs(r[1])), sub8=np.mean(np.abs(r[8])), Rq1=np.mean(np.abs(Rq))))
    J.set_params("nominal", None, 0)
    return out

base = stats("nominal", None, 0)
print("healthy baselines (mean over ICs):", {k: f"{np.mean([b[k] for b in base]):.2e}" for k in base[0]})
ROWS.append(dict(eps=0.0, **{f"baseline_{k}": float(np.mean([b[k] for b in base])) for k in base[0]}))
res = {k: [] for k in ("raw", "sub8", "Rq1")}
print(f"{'eps':>8} {'raw':>5} {'sub8':>5} {'Rq1':>5}   mean ratio to healthy (raw, sub8, Rq1)")
for e in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    t = stats("stale", "masspole", e)
    line = []; rat = []
    for k in res:
        a = np.array([x[k] for x in t]); b = np.array([x[k] for x in base])
        p = ideal_diff._permutation_greater(a, b, a, b, floor=1e-12)
        line.append("Y" if p < 0.01 else "."); rat.append(a.mean()/b.mean())
    print(f"{e:>8.0e} {line[0]:>5} {line[1]:>5} {line[2]:>5}   " + "  ".join(f"{r:.1e}" for r in rat), flush=True)
    ROWS.append(dict(eps=e, **{f"detected_{k}": (l == "Y") for k, l in zip(res, line)}, **{f"ratio_to_healthy_{k}": r for k, r in zip(res, rat)}))

save_csv("cartpole_order_adapted_richardson.csv", ROWS)
