"""Operating envelope of the refinement diagnostics under SEPARATED noise sources (analytic Acrobot, h = 0.2).

Where noise enters decides whether it matters:
  start_state_noise : the diagnostic start state is a noisy logged observation, but the simulator is re-stepped from it
                      and energies are computed exactly (noise only moves the evaluation point).
  readout_state_noise: every state read back for an energy evaluation (start and each refined end state) carries
                      independent noise (a noisy observation channel on the simulator's OUTPUT).
  energy_noise       : independent noise on every energy evaluation (invariant-evaluation error).
  float32_readout    : states read back at float32 precision (Gymnasium's observation dtype).
Floating-point/integration error of the step map itself is the float64 row (sigma = 0).

For each condition: fraction of trials the frozen classifier labels model / consistent / order-k / undetermined
(guard floor ESTIMATED from repeated readouts of the same states, or, for the deterministic float32 quantiser,
from the declared precision by +-0.5 ulp dithering; never assumed), healthy false-alarm rate, and the
paired baselines (raw r, 8 sub-steps, R2; flagged if > 10x a paired healthy run under independent noise).
Paper: Section 5.5.  Output: Results/noise_sources_operating_envelope.csv
"""
import numpy as np
import acrobot as A
import acrobot_parametric as P
import fingerprint as F
from results_io import save_csv, n_trials

H, T = 0.2, 10.0
NST = int(round(T / H))
MS = (1, 2, 4, 8)
SIGMAS = [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
EPS = [1e-4, 1e-3, 1e-2]
GATE = 10.0
NTR = n_trials(40, 6)


def energy_vec(S):
    S = np.atleast_2d(S)
    q1, q2, d1, d2 = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    c1, c2, s1, s2 = np.cos(q1), np.cos(q2), np.sin(q1), np.sin(q2)
    T_ = 1.75 * d1**2 + 0.5 * c2 * d1**2 + 1.25 * d1 * d2 + 0.5 * c2 * d1 * d2 + 0.625 * d2**2
    V = -14.7 * c1 - 4.9 * c1 * c2 + 4.9 * s1 * s2
    return T_ + V


def advance(s, m2, m):
    x = s
    for _ in range(m):
        x = P.rk4_step(x, 0.0, H / m, m2)
    return x


def trajectory_states(s0, m2):
    """true start states and, for each m, the end states of the refined step (from the TRUE start)."""
    S0 = np.empty((NST, 4)); SE = {m: np.empty((NST, 4)) for m in MS}
    s = s0.copy()
    for k in range(NST):
        S0[k] = s
        for m in MS:
            SE[m][k] = advance(s, m2, m)
        s = SE[1][k]
    return S0, SE


def readout(S, source, sigma, rng):
    """energies obtained from states S through the chosen channel."""
    if source == "readout_state_noise":
        S = S + sigma * rng.standard_normal(S.shape)
    elif source == "float32_readout":
        S = S.astype(np.float32).astype(np.float64)
    E = energy_vec(S)
    if source == "energy_noise":
        E = E + sigma * rng.standard_normal(E.shape)
    return E


def stats_from(E0, Ee):
    r = {m: (Ee[m] - E0) / H for m in MS}
    R2 = (-r[1] + 48 * r[2] - 512 * r[4]) / 31.0
    return {m: float(np.mean(np.abs(r[m]))) for m in MS}, float(np.mean(np.abs(R2)))


def estimate_floor(S0, source, sigma, rng):
    """5x the expected mean |noise| of one residual, from the empirical sd of repeated readouts of the SAME states."""
    if source in ("float64", "start_state_noise"):
        return 0.0
    if source == "float32_readout":
        # Quantisation is deterministic, so repeated readouts show no spread.  Use the DECLARED precision instead:
        # dither the float32 readout by +-0.5 ulp (uniform) and take the spread of the resulting energies.
        Sq = S0.astype(np.float32)
        ulp = np.spacing(np.abs(Sq)).astype(np.float64)
        d = energy_vec(Sq.astype(np.float64) + ulp * rng.uniform(-0.5, 0.5, Sq.shape)) - energy_vec(Sq.astype(np.float64))
        sd = float(np.sqrt(np.mean(d ** 2)))
        return 5 * np.sqrt(2 / np.pi) * np.sqrt(2) * sd / H
    e1, e2 = readout(S0, source, sigma, rng), readout(S0, source, sigma, rng)
    sd = float(np.std(e1 - e2)) / np.sqrt(2)
    return 5 * np.sqrt(2 / np.pi) * np.sqrt(2) * sd / H


def one_condition(s0, m2, source, sigma, rng):
    if source == "start_state_noise":     # re-step from a noisy logged start; energies exact
        S0, SE = trajectory_states(s0, m2)
        Sn = S0 + sigma * rng.standard_normal(S0.shape)
        SEn = {m: np.array([advance(s, m2, m) for s in Sn]) for m in MS}
        E0 = energy_vec(Sn); Ee = {m: energy_vec(SEn[m]) for m in MS}
        floor = 0.0
    else:
        S0, SE = CACHE[(tuple(s0), m2)]
        E0 = readout(S0, source, sigma, rng); Ee = {m: readout(SE[m], source, sigma, rng) for m in MS}
        floor = estimate_floor(S0, source, sigma, rng)
    r, R2 = stats_from(E0, Ee)
    cls = F.guarded_classify(r[1], r[2], r[4], r[8], 4, np.mean(np.abs(E0)), H, noise_floor=floor)
    return cls, dict(raw=r[1], sub8=r[8], R2=R2)


CACHE = {}


def main():
    rng0 = np.random.default_rng(2026)
    starts = [rng0.uniform(-1.5, 1.5, 4) for _ in range(NTR)]
    m2s = [1.0] + [1.0 + e for e in EPS]
    print(f"precomputing {len(starts) * len(m2s)} trajectories ...", flush=True)
    for s0 in starts:
        for m2 in m2s:
            CACHE[(tuple(s0), m2)] = trajectory_states(s0, m2)
    sources = [("float64", [0.0]), ("start_state_noise", SIGMAS[1:]), ("readout_state_noise", SIGMAS[1:]),
               ("energy_noise", SIGMAS[1:]), ("float32_readout", [0.0])]
    rows = []
    print(f"{'source':<20}{'sigma':>8}  healthy(FA/undet)   " + "  ".join(f"eps={e:g}: model/undet [raw sub8 R2]" for e in EPS), flush=True)
    for source, sigs in sources:
        for sg in sigs:
            rng = np.random.default_rng(11)
            hc = {}; base = []
            for s0 in starts:                       # healthy + paired healthy reference (independent noise)
                c, st = one_condition(s0, 1.0, source, sg, rng); hc[c] = hc.get(c, 0) + 1
                _, st2 = one_condition(s0, 1.0, source, sg, rng); base.append((st, st2))
            n = len(starts)
            fa = n - sum(hc.get(k, 0) for k in ("consistent", "undetermined", "unstable", "blowup"))
            fa_base = {k: sum(b[0][k] > GATE * b[1][k] for b in base) / n for k in ("raw", "sub8", "R2")}
            row = dict(source=source, sigma=sg, n_starts=n, healthy_classes=str(hc), healthy_false_alarm=fa / n,
                       healthy_undetermined=hc.get("undetermined", 0) / n,
                       **{f"healthy_paired_false_alarm_{k}": v for k, v in fa_base.items()})
            line = f"{source:<20}{sg:>8.0e}  {fa}/{hc.get('undetermined', 0)} of {n:<9}"
            for e in EPS:
                cc = {}; det = {"raw": 0, "sub8": 0, "R2": 0}
                for k, s0 in enumerate(starts):
                    c, st = one_condition(s0, 1.0 + e, source, sg, rng); cc[c] = cc.get(c, 0) + 1
                    for kk in det:
                        det[kk] += st[kk] > GATE * base[k][0][kk]
                row.update({f"eps{e:g}_classes": str(cc), f"eps{e:g}_model_rate": cc.get("model", 0) / n,
                            f"eps{e:g}_undetermined_rate": cc.get("undetermined", 0) / n,
                            **{f"eps{e:g}_paired_detect_{kk}": v / n for kk, v in det.items()}})
                line += f"  {cc.get('model', 0)}/{cc.get('undetermined', 0)} [" + " ".join(f"{v / n:.2f}" for v in det.values()) + "]"
            rows.append(row); print(line, flush=True)
    save_csv("noise_sources_operating_envelope.csv", rows)


if __name__ == "__main__":
    main()
