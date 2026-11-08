"""RK4 energy-defect asymptote at 40-digit precision: does the local residual slope go to 4 or 5?
Paper: Section 4.4 (asymptotics of the RK4 defect).   Output: Results/rk4_defect_high_precision.csv
"""
import mpmath as mp
from acrobot_mpmath import *
from results_io import save_csv

def main():
    state = [mp.mpf('0.1'), mp.mpf('-0.2'), mp.mpf('0.3'), mp.mpf('-0.15')]
    print(f"state = {[float(x) for x in state]}")
    print(f"precision = {mp.mp.dps} digits")
    print()

    hs = [mp.mpf(s) for s in ['0.2', '0.1', '0.05', '0.025', '0.01',
                               '0.005', '0.0025', '0.001', '0.0005',
                               '0.0001', '1e-5', '1e-6']]
    rs = []
    rows = []
    print(f"{'h':>12}   {'|r|':>22}   {'adjacent slope':>16}")
    print("-" * 60)
    for i, h in enumerate(hs):
        r = local_residual(state, h)
        rs.append(r)
        slope_str = ""; slope = None
        if i > 0 and abs(rs[i]) > 0 and abs(rs[i - 1]) > 0:
            ratio = abs(rs[i - 1]) / abs(rs[i])
            if ratio > 0:
                slope = mp.log(ratio) / mp.log(hs[i - 1] / h)
                slope_str = f"{float(slope):>16.3f}"
        print(f"{float(h):>12.4e}   {float(abs(r)):>22.6e}   {slope_str}")
        rows.append(dict(h=float(h), abs_r=float(abs(r)), signed_r=float(r), adjacent_slope=(float(slope) if slope_str else None)))

    save_csv("rk4_defect_high_precision.csv", rows)
    print()
    print("Interpretation:")
    print("  Slope trending toward 4 as h -> 0: pre-asymptotic (float64 gave 5).")
    print("  Slope stable at 5 as h -> 0:      asymptotic superconvergence.")
    print("  Slope trending elsewhere:         report honestly.")




if __name__ == "__main__":
    main()
