"""Shared P-LRU regularizers.

Canonical paper loss:
    L = L_task + tau * sum_j [q_j + c q_j (1 - q_j)],
    q_j = |lambda_j|^2.
"""

# Fixed main P-LRU recipe used for new paper runs.
# This is the canonical-q strong zero-phase P-LRU setting selected by Exp57
# sweep: weak slow-mode drain, strong polarization, with task gradients deciding
# the retained manifold dimension.
DEFAULT_TAU = 1e-3
DEFAULT_C = 50.0
DEFAULT_WARMUP_FRAC = 0.25
DEFAULT_STEPS_MULT = 2.0
DEFAULT_LR = 3e-3
DEFAULT_THETA_LR_MULT = 0.0


def canonical_q_loss_from_lam(lam, tau=DEFAULT_TAU, c=DEFAULT_C):
    q = lam ** 2
    return float(tau) * (q + float(c) * q * (1.0 - q)).sum()


def tau_c_from_energy(alpha, rho):
    """Convert old alpha * [q(1-q) + rho*q] notation to canonical tau/c."""
    return float(alpha) * float(rho), 1.0 / float(rho)
