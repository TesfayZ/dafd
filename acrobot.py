"""
Analytic Acrobot model (Gymnasium defaults).

State convention: state = [q1, q2, dq1, dq2]  (angles and angular velocities)
Energy form:      the polynomial of the paper's Eq. (acrobot-energy)
Integrator:       rk4_step (the integrator of Gymnasium's Acrobot-v1)
"""

import numpy as np

# --- Parameters (Gymnasium defaults) ---
M1 = M2 = 1.0
L1 = L2 = 1.0
LC1 = LC2 = 0.5
I1 = I2 = 1.0
G = 9.8

PHI1 = M1 * LC1 + M2 * L1   # = 1.5
PHI2 = M2 * LC2             # = 0.5
H_C  = M2 * L1 * LC2        # = 0.5


def mass_matrix(q):
    """Return (M, h_c) with M the 2x2 mass matrix at joint angles q."""
    c2 = np.cos(q[1])
    d1 = M1 * LC1**2 + M2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * c2) + I1 + I2
    d2 = M2 * (LC2**2 + L1 * LC2 * c2) + I2
    d3 = M2 * LC2**2 + I2
    return np.array([[d1, d2], [d2, d3]]), H_C


def dynamics(state, tau):
    """
    state = [q1, q2, dq1, dq2]
    Returns d(state)/dt = [dq1, dq2, ddq1, ddq2].
    """
    q = state[:2]
    dq = state[2:]
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    M, h_c = mass_matrix(q)
    d1, d2, d3 = M[0, 0], M[0, 1], M[1, 1]

    rhs1 = (-PHI1 * G * np.sin(q[0])
            - PHI2 * G * np.sin(q[0] + q[1])
            + h_c * s2 * (2.0 * dq[0] * dq[1] + dq[1]**2))
    rhs2 = (-PHI2 * G * np.sin(q[0] + q[1])
            - h_c * s2 * dq[0]**2
            + tau)

    det = d1 * d3 - d2 * d2
    ddq1 = (d3 * rhs1 - d2 * rhs2) / det
    ddq2 = (-d2 * rhs1 + d1 * rhs2) / det
    return np.array([dq[0], dq[1], ddq1, ddq2])


def energy(state):
    """Mechanical energy in state variables (paper Eq. (2))."""
    q1, q2, dq1, dq2 = state
    c1, c2 = np.cos(q1), np.cos(q2)
    s1, s2 = np.sin(q1), np.sin(q2)
    T = (1.75 * dq1**2 + 0.5 * c2 * dq1**2
         + 1.25 * dq1 * dq2 + 0.5 * c2 * dq1 * dq2
         + 0.625 * dq2**2)
    V = -14.7 * c1 - 4.9 * c1 * c2 + 4.9 * s1 * s2
    return T + V


# ---------- Integrators ----------

def rk4_step(state, tau, h):
    """Consistent RK4 on the full state."""
    k1 = dynamics(state, tau)
    k2 = dynamics(state + 0.5 * h * k1, tau)
    k3 = dynamics(state + 0.5 * h * k2, tau)
    k4 = dynamics(state + h * k3, tau)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
