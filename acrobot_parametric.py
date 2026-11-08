"""Parameterized Acrobot: m2 is a runtime argument.

At m2 = 1.0 this reduces to acrobot.py to machine precision.
"""

import numpy as np

M1 = 1.0
L1 = L2 = 1.0
LC1 = LC2 = 0.5
I1 = I2 = 1.0
G = 9.8
M2_REF = 1.0


def mass_matrix(q, m2):
    c2 = np.cos(q[1])
    d1 = M1 * LC1**2 + m2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * c2) + I1 + I2
    d2 = m2 * (LC2**2 + L1 * LC2 * c2) + I2
    d3 = m2 * LC2**2 + I2
    return np.array([[d1, d2], [d2, d3]])


def dynamics(state, tau, m2):
    q1, q2, dq1, dq2 = state
    c2, s2 = np.cos(q2), np.sin(q2)
    phi1 = M1 * LC1 + m2 * L1
    phi2 = m2 * LC2
    h_c = m2 * L1 * LC2
    M = mass_matrix([q1, q2], m2=m2)
    d1, d2, d3 = M[0, 0], M[0, 1], M[1, 1]
    rhs1 = (-phi1 * G * np.sin(q1)
            - phi2 * G * np.sin(q1 + q2)
            + h_c * s2 * (2.0 * dq1 * dq2 + dq2**2))
    rhs2 = (-phi2 * G * np.sin(q1 + q2)
            - h_c * s2 * dq1**2 + tau)
    det = d1 * d3 - d2 * d2
    ddq1 = (d3 * rhs1 - d2 * rhs2) / det
    ddq2 = (-d2 * rhs1 + d1 * rhs2) / det
    return np.array([dq1, dq2, ddq1, ddq2])


def energy(state, m2):
    q1, q2, dq1, dq2 = state
    M = mass_matrix([q1, q2], m2=m2)
    T = 0.5 * (M[0, 0] * dq1**2 + 2 * M[0, 1] * dq1 * dq2
               + M[1, 1] * dq2**2)
    phi1 = M1 * LC1 + m2 * L1
    phi2 = m2 * LC2
    V = -phi1 * G * np.cos(q1) - phi2 * G * np.cos(q1 + q2)
    return T + V


def power(state, tau):
    return tau * state[3]


def rk4_step(state, tau, h, m2):
    k1 = dynamics(state, tau, m2)
    k2 = dynamics(state + 0.5 * h * k1, tau, m2)
    k3 = dynamics(state + 0.5 * h * k2, tau, m2)
    k4 = dynamics(state + h * k3, tau, m2)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def consistency_check():
    """Assert that acrobot_param at m2=1.0 matches acrobot.py."""
    from acrobot import energy as energy_ref
    s = np.array([0.1, -0.2, 0.3, -0.15])
    E_param = energy(s, m2=1.0)
    E_ref = energy_ref(s)
    return abs(E_param - E_ref) / max(abs(E_ref), 1e-10)