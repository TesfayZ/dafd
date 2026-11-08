"""High-precision (mpmath, 40 digits) Acrobot: dynamics, energy, RK4 step, local energy-defect residual."""
import mpmath as mp

mp.mp.dps = 40  # 40 decimal digits

M1 = M2 = mp.mpf(1)
L1 = L2 = mp.mpf(1)
LC1 = LC2 = mp.mpf('0.5')
I1 = I2 = mp.mpf(1)
G = mp.mpf('9.8')
PHI1 = M1 * LC1 + M2 * L1
PHI2 = M2 * LC2
H_C = M2 * L1 * LC2


def dynamics(state, tau=mp.mpf(0)):
    q1, q2, dq1, dq2 = state
    c2 = mp.cos(q2); s2 = mp.sin(q2)
    s1 = mp.sin(q1); s12 = mp.sin(q1 + q2)
    d1 = M1*LC1**2 + M2*(L1**2 + LC2**2 + 2*L1*LC2*c2) + I1 + I2
    d2 = M2*(LC2**2 + L1*LC2*c2) + I2
    d3 = M2*LC2**2 + I2
    rhs1 = -PHI1*G*s1 - PHI2*G*s12 + H_C*s2*(2*dq1*dq2 + dq2**2)
    rhs2 = -PHI2*G*s12 - H_C*s2*dq1**2 + tau
    det = d1*d3 - d2*d2
    ddq1 = (d3*rhs1 - d2*rhs2) / det
    ddq2 = (-d2*rhs1 + d1*rhs2) / det
    return [dq1, dq2, ddq1, ddq2]


def energy(state):
    q1, q2, dq1, dq2 = state
    c1, c2 = mp.cos(q1), mp.cos(q2)
    s1, s2 = mp.sin(q1), mp.sin(q2)
    T = (mp.mpf('1.75')*dq1**2 + mp.mpf('0.5')*c2*dq1**2
         + mp.mpf('1.25')*dq1*dq2 + mp.mpf('0.5')*c2*dq1*dq2
         + mp.mpf('0.625')*dq2**2)
    V = -mp.mpf('14.7')*c1 - mp.mpf('4.9')*c1*c2 + mp.mpf('4.9')*s1*s2
    return T + V


def rk4_step(state, tau, h):
    k1 = dynamics(state, tau)
    k2 = dynamics([state[i] + h/2*k1[i] for i in range(4)], tau)
    k3 = dynamics([state[i] + h/2*k2[i] for i in range(4)], tau)
    k4 = dynamics([state[i] + h*k3[i] for i in range(4)], tau)
    return [state[i] + h/6*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(4)]


def local_residual(state, h):
    s1 = rk4_step(state, mp.mpf(0), h)
    return (energy(s1) - energy(state)) / h


