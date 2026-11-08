"""Calibrated fixed-threshold detectors vs the calibration-free fingerprint, shipped AcrobotEnv.

Healthy runs calibrate drift / raw / 8-sub-step / R2 thresholds (max over a calibration half);
false alarms are counted on the held-out half.  Paper: Section 5.2 (fixed-threshold paragraph).
Output: Results/shipped_acrobot_calibrated_detectors.csv
"""
import numpy as np
import zlib
import fingerprint as F
from shipped_acrobot import *          # fault injection, trials, stats, constants
from results_io import save_csv

def main():
    rows = []
    print("collecting healthy trials ...", flush=True)
    healthy = collect(dict(), N_HEALTHY, seed=101)
    hs = [stats(*t) for t in healthy]
    cal, held = hs[:N_HEALTHY // 2], hs[N_HEALTHY // 2:]
    dets = ["drift", "raw", "sub8", "R2"]
    thr = {k: max(x[k] for x in cal) for k in dets}
    print("calibration thresholds (max of %d healthy):" % len(cal),
          {k: f"{v:.2e}" for k, v in thr.items()})
    fa = {k: sum(x[k] > thr[k] for x in held) for k in dets}
    cls_h = {}
    for x in held:
        c = F.guarded_classify(x["r1"], x["r2"], x["r4"], x["r8"], 4, x["E_abs_mean"], H)
        cls_h[c] = cls_h.get(c, 0) + 1
    fa["fingerprint"] = sum(v for k, v in cls_h.items()
                            if k not in ("consistent", "undetermined", "unstable", "blowup"))
    print("false alarms on held-out healthy (n=%d):" % len(held),
          {k: v for k, v in fa.items() if k != "fingerprint"},
          "| fingerprint (same %d held-out healthy):" % len(held), fa["fingerprint"])
    print("healthy classification:", cls_h, flush=True)
    rows.append(dict(case="healthy", eps="", n=len(held), **{f"det_{k}": fa[k] for k in dets},
                     det_fingerprint=fa["fingerprint"], n_fingerprint=len(held), classes=str(cls_h)))

    cases = []
    for a in NOMINAL:
        for e in EPS:
            cases.append((f"{a}", e, dict(attr=a, eps=e)))
    cases.append(("euler", 0.0, dict(integ="euler")))
    cases.append(("rk2", 0.0, dict(integ="rk2")))
    cases.append(("euler+MASS_2", 1e-2, dict(integ="euler", attr="LINK_MASS_2", eps=1e-2)))

    print("\n%-16s %7s | %6s %6s %6s %6s %6s | classes" % ("fault", "eps", "drift", "raw", "sub8", "R2", "model"))
    stored = {}
    for name, e, kw in cases:
        seed = zlib.crc32(f"{name}|{e:g}".encode("utf-8")) % 10**6
        tr = collect(kw, N_FAULT, seed=seed)
        ss = [stats(*t) for t in tr]
        if name in ("LINK_MASS_2",) and e in (1e-2,) or name in ("euler", "rk2"):
            stored[(name, e)] = tr
        det = {k: sum(x[k] > thr[k] for x in ss) for k in dets}
        cls = {}
        for x in ss:
            c = F.guarded_classify(x["r1"], x["r2"], x["r4"], x["r8"], 4, x["E_abs_mean"], H)
            cls[c] = cls.get(c, 0) + 1
        det["fingerprint"] = cls.get("model", 0)   # class-level fingerprint detections
        n = len(ss)
        print("%-16s %7.0e | %6s %6s %6s %6s %6s | %s" % (
            name, e, *(f"{det[k]}/{n}" for k in dets), f"{det['fingerprint']}/{n}", cls), flush=True)
        rows.append(dict(case=name, eps=e, n=n, **{f"det_{k}": det[k] for k in dets},
                         det_fingerprint=det["fingerprint"], n_fingerprint=n, classes=str(cls)))
    stored[("healthy", 0)] = healthy

    # noise study on stored trials: fingerprint with / without noise-floor guard
    print("\nFingerprint under energy measurement noise (healthy n=%d, others n=%d)" % (60, N_FAULT))
    print("%-12s %9s | %-34s | %-34s" % ("case", "sigma_E", "unguarded classes", "guarded (r8 > 5*floor)"))
    for key in [("healthy", 0), ("euler", 0.0), ("LINK_MASS_2", 1e-2)]:
        trials = stored[key][:60] if key[0] == "healthy" else stored[key]
        for sg in [0.0, 1e-10, 1e-8, 1e-6]:
            rng = np.random.default_rng(5)
            cu, cg = {}, {}
            # noise floor of mean|r| for independent N(0,s) energies: sqrt(2/pi)*sqrt(2)*s/h
            floor = 5 * np.sqrt(2 / np.pi) * np.sqrt(2) * sg / H
            for t in trials:
                d = stats(*t, sigma=sg, rng=rng)
                c1 = F.classify(d["r1"], d["r2"], d["r4"], d["r8"], 4, 0.0)
                c2 = F.guarded_classify(d["r1"], d["r2"], d["r4"], d["r8"], 4, d["E_abs_mean"], H, noise_floor=floor)
                cu[c1] = cu.get(c1, 0) + 1
                cg[c2] = cg.get(c2, 0) + 1
            print("%-12s %9.0e | %-34s | %-34s" % (key[0], sg, cu, cg), flush=True)
            rows.append(dict(case=f"noise:{key[0]}", eps=sg, n=len(trials),
                             classes=str(cu), det_fingerprint=str(cg), n_fingerprint=len(trials)))
    keys = sorted({k for r in rows for k in r})
    save_csv("shipped_acrobot_calibrated_detectors.csv", rows, keys)


if __name__ == "__main__":
    main()
