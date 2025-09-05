"""
Worldline instanton computation and plotting for Schwinger effect in extremal Reissner-Nordström spacetime.

This script computes the worldline instanton action for different charge-to-mass
ratios z, and plots the normalized action S_wl/(Qm) against Δρ.

Dependencies:
    numpy
    scipy
    matplotlib

Usage:
    python worldline_action_ext_RN.py
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

Z_VALUES = [1.2, 1.5, 2, 3]  # charge-to-mass ratios, must be > 1
COLORS = ['r', 'g', 'b', 'm']

EPS = 1e-6        # tolerance to avoid singularities
QUAD_LIMIT = 200  # maximum subdivisions for numerical integration


# -------------------------------------------------------------------------
# Physics functions
# -------------------------------------------------------------------------

def effective_potential(rho: float, rho0: float, z: float) -> float:
    """
    Effective potential h(rho).

    Parameters
    ----------
    rho : float
        Radial coordinate.
    rho0 : float
        Turning point parameter.
    z : float
        Charge-to-mass ratio (>1).

    Returns
    -------
    float
        Value of the effective potential h(rho).
    """
    return (1 - z**2) * rho**2 + 2 * (z**2 * rho0 - 1) * rho + (1 - z**2 * rho0**2)


def euclidean_time_integrand(rho: float, rho0: float, z: float) -> float:
    """
    Integrand for derivative of Euclidean time τ with respect to radial coordinate ρ.
    """
    h_val = effective_potential(rho, rho0, z)
    if h_val <= 0:
        return 0
    num = rho0 - rho
    den = rho**2 * (1 - rho)**2 * np.sqrt(h_val)
    return num / den


def G_rhox(rhox: float, rho2: float, rho0: float, z: float) -> float:
    """
    Compute G(rhox) = ∫[rhox,rho2] (dτ/dρ) dρ.

    G(rhox) must vanish for consistency of the Euclidean worldline loop.

    Returns
    -------
    float
        Value of G(rhox). Returns np.inf on integration failure.
    """
    try:
        val, _ = quad(euclidean_time_integrand, rhox, rho2,
                      args=(rho0, z), limit=QUAD_LIMIT)
        return val
    except Exception:
        return np.inf


def worldline_integrand_first(rho: float, rho0: float, z: float) -> float:
    """
    Integrand for computing a/Q (normalization parameter), first term of the worldline action.
    """
    h_val = effective_potential(rho, rho0, z)
    if h_val <= 0:
        return 0
    return 2 / (rho**2 * np.sqrt(h_val))


def worldline_integrand_second(rho: float, rho0: float, rho1: float, rho2: float) -> float:
    """
    Integrand for I, the second term of the worldline action.
    """
    h_min = (rho2 - rho) * (rho - rho1)
    if h_min <= 0:
        return 0
    num = rho0 - rho
    den = rho * (1 - rho)**2 * np.sqrt(h_min)
    return num / den


# -------------------------------------------------------------------------
# Main computation & plotting
# -------------------------------------------------------------------------

def main():
    plt.figure()

    for z, color in zip(Z_VALUES, COLORS):
        rho0_vals = np.linspace(1 / z, 1, 100)
        delta_rho_vals = []
        S_inst_vals = []

        for rho0 in rho0_vals:
            # Find zeros rho1, rho2 of the effective potential
            rho1 = (z * rho0 - 1) / (z - 1)
            rho2 = (z * rho0 + 1) / (z + 1)

            if not (0 < rho1 < rho2 < 1):
                continue

            # Find rhox that makes G_rhox = 0
            try:
                rhox = brentq(G_rhox, rho1 + EPS, rho2 - EPS,
                              args=(rho2, rho0, z))
            except ValueError:
                continue

            a_over_Q, _ = quad(worldline_integrand_first, rhox, rho2,
                               args=(rho0, z), limit=QUAD_LIMIT, points=[rhox, rho2])

            I_val, _ = quad(worldline_integrand_second, rhox, rho2,
                            args=(rho0, rho1, rho2), limit=QUAD_LIMIT, points=[rhox, rho2])

            S_A_over_Qm = 2 * I_val * z**2 / np.sqrt(z**2 - 1)
            S_inst_over_Qm = a_over_Q + S_A_over_Qm

            delta_rho = 1 - rhox
            delta_rho_vals.append(delta_rho)
            S_inst_vals.append(S_inst_over_Qm)

        plt.plot(delta_rho_vals, S_inst_vals, label=f'$z={z}$', color=color)

    # Plot formatting
    plt.xlabel(r'$\Delta \rho$', fontsize=14)
    plt.ylabel(r'$\frac{S_{\text{wl}}}{Qm}$', rotation=0, labelpad=10, fontsize=14)
    plt.grid(False)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=1)

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
