"""Signed 40-digit RK4 defect (sign change near h = 0.011) and 50-digit asymptotics of the healthy R2 (order 6).
Paper: Section 4.4 and Appendix D.   Outputs: Results/rk4_signed_defect.csv, Results/r2_asymptotics.csv
"""
import mpmath as mp
import acrobot_mpmath as H
from results_io import save_csv
SIGN, R2ROWS = [], []
mp.mp.dps = 40
state = [mp.mpf('0.1'), mp.mpf('-0.2'), mp.mpf('0.3'), mp.mpf('-0.15')]
print("signed one-step defect r(h)=(E1-E0)/h, RK4, Acrobot start state")
hs = ['0.2','0.1','0.05','0.025','0.02','0.015','0.012','0.01','0.009','0.008','0.007','0.006','0.005','0.0025','0.001','0.0001']
prev=None
for s in hs:
    h = mp.mpf(s); r = H.local_residual(state, h)
    print(f"h={s:>7}  r={mp.nstr(r,6):>14}   r/h^4={mp.nstr(r/h**4,6):>12}  r/h^5={mp.nstr(r/h**5,6):>12}")
    SIGN.append(dict(h=float(h), signed_r=float(r), r_over_h4=float(r/h**4), r_over_h5=float(r/h**5)))
# fit c4,c5,c6 from small-h Richardson: r/h^4 -> c4
for h0 in ['1e-4','1e-5']:
    h=mp.mpf(h0); print('c4 est at',h0, mp.nstr(H.local_residual(state,h)/h**4,10))

# ---- healthy R2 asymptotics (50 digits) ----
mp.mp.dps = 50
def comp(state, h, m):
    s = state
    for _ in range(m): s = H.rk4_step(s, mp.mpf(0), h/m)
    return (H.energy(s)-H.energy(state))/h
def R2(state, h):
    r1, r2, r4 = comp(state,h,1), comp(state,h,2), comp(state,h,4)
    return (-r1 + 48*r2 - 512*r4)/31
state0 = [mp.mpf('0.1'), mp.mpf('-0.2'), mp.mpf('0.3'), mp.mpf('-0.15')]
print("healthy R2 (high precision), start state of the Richardson and sweep studies")
prev=None
hs=['0.2','0.1','0.05','0.025','0.0125','0.00625','0.003125','0.001','0.0005','0.0001']
for s in hs:
    h=mp.mpf(s); v=R2(state0,h)
    sl = '' if prev is None else f"{float(mp.log(abs(prev[1]/v))/mp.log(prev[0]/h)):8.3f}"
    print(f"h={s:>9}  R2={mp.nstr(v,6):>14}  R2/h^6={mp.nstr(v/h**6,6):>12} R2/h^7={mp.nstr(v/h**7,6):>12}  slope={sl}")
    R2ROWS.append(dict(kind="start_state", h=float(h), R2=float(v), R2_over_h6=float(v/h**6), slope=(None if sl == "" else float(sl))))
    prev=(h,v)
# also check amplification structure: D part identical -> use faulted model? (skip)
# multi-state check of asymptotic order of R2 at small h
import random
random.seed(1)
print("\nasymptotic slope of R2 between h=2e-3 and 1e-3, 8 random states")
for k in range(8):
    st=[mp.mpf(random.uniform(-3,3)),mp.mpf(random.uniform(-3,3)),mp.mpf(random.uniform(-4,4)),mp.mpf(random.uniform(-4,4))]
    a,b=mp.mpf('0.002'),mp.mpf('0.001')
    ra,rb=R2(st,a),R2(st,b)
    print(f"  state {k}: slope = {float(mp.log(abs(ra/rb))/mp.log(a/b)):.3f}  R2(h=1e-3)/h^6 = {mp.nstr(rb/b**6,5)}")
    R2ROWS.append(dict(kind=f"random_state_{k}", h=1e-3, R2=float(rb), R2_over_h6=float(rb/b**6), slope=float(mp.log(abs(ra/rb))/mp.log(a/b))))

save_csv("rk4_signed_defect.csv", SIGN)
save_csv("r2_asymptotics.csv", R2ROWS)
