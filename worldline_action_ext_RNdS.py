"""
Numerical calculation of the worldline instanton action
in extremal RNdS spacetime with comparison to a near-horizon
AdS approximation for large r_E.

Dependencies:
    numpy
    scipy
    matplotlib

Usage:
    python worldline_action_ext_RNdS.py
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

L = 1  # dS length scale
R_E_VALUES = [0.01, 0.1, 0.2, 0.3]
R_E_COLORS = ['m', 'b', 'g', 'r']

Z_VALUES = [100]
Z_COLORS = ['k']

EPS = 1e-12      # tolerance for brentq root finding
QUAD_LIMIT = 200 # max subdivisions in integration


# -------------------------------------------------------------------------
# Physics functions
# -------------------------------------------------------------------------

def blackening_factor(rho: float, l: float, rE: float) -> float:
    """
    Blackening factor for extremal RNdS spacetime.

    Parameters
    ----------
    rho : float
        Radial coordinate.
    l : float
        Curvature radius.
    rE : float
        Extremal horizon radius.

    Returns
    -------
    float
        Value of blackening factor at rho.
    """
    return (1 - rho)**2 * (1 - (rE**2 / l**2) * (1 / rho**2 + 2 / rho + 3))


def effective_potential_coefficients(l: float, rE: float, Qd: float, z: float, rho0: float) -> list[float]:
    """
    Coefficients of polynomial h(ρ).

    Returns coefficients [A, B, C, D, E] such that
    h(ρ) = A ρ⁴ + B ρ³ + C ρ² + D ρ + E.
    """
    A = 1 - Qd**2 * z**2 / rE**2 - 3 * rE**2 / l**2
    B = -2 + 2 * Qd * rho0 * z**2 / rE + 4 * rE**2 / l**2
    C = 1 - z**2 * rho0**2
    D = 0
    E = rE**2 / l**2
    return [A, B, C, D, E]


def effective_potential_polynomial(rho: float, l: float, rE: float, Qd: float, z: float, rho0: float) -> float:
    """Evaluate h(ρ) from its polynomial form."""
    A, B, C, D, E = effective_potential_coefficients(l, rE, Qd, z, rho0)
    return A * rho**4 + B * rho**3 + C * rho**2 + D * rho + E


def euclidean_time_integrand(rho: float, l: float, rE: float, Qd: float, z: float, rho0: float, rc: float) -> float:
    """Integrand for derivative of Euclidean time τ with respect to ρ."""
    f_val = blackening_factor(rho, l, rE)
    h_val = effective_potential_polynomial(rho, l, rE, Qd, z, rho0)
    if f_val == 0 or h_val <= 0:
        return 0
    num = rho0 - (Qd / rE) * rho
    den = rho * f_val * np.sqrt(h_val)
    return num / den


def G_rhox(rhox: float, rho2: float, l: float, rE: float, Qd: float, z: float, rho0: float, rc: float) -> float:
    """
    Compute G(rhox) = ∫[rhox,rho2] (dt/dρ) dρ.

    Must vanish for consistency of the Euclidean loop.
    """
    try:
        val, _ = quad(euclidean_time_integrand, rhox, rho2,
                      args=(l, rE, Qd, z, rho0, rc),
                      limit=QUAD_LIMIT, points=[rhox, rho2, rE / rc])
        return val
    except Exception:
        return np.inf


def worldline_integrand_first(rho: float, l: float, rE: float, Qd: float, z: float, rho0: float) -> float:
    """Integrand for computing a (normalization parameter), first term of the worldline action."""
    h_val = effective_potential_polynomial(rho, l, rE, Qd, z, rho0)
    if h_val <= 0:
        return 0
    return 2 * rE / (rho * np.sqrt(h_val))


def worldline_integrand_second(rho: float, l: float, rE: float, Qd: float, z: float, rho0: float, rc: float) -> float:
    """Integrand for I, the second term of the worldline action."""
    f_val = blackening_factor(rho, l, rE)
    h_val = effective_potential_polynomial(rho, l, rE, Qd, z, rho0)
    if f_val == 0 or h_val <= 0:
        return 0
    num = rho0 - (Qd / rE) * rho
    den = f_val * np.sqrt(h_val)
    return num / den


def find_valid_rho0_and_rhox(l: float, rE: float, Qd: float, z: float, rho0_grid: np.ndarray, rc: float):
    """
    Identify valid values of ρ0 and rhox.

    A pair (ρ0, rhox) is valid if:
      - h has two real, distinct roots in (0, 1),
      - τ'(ρ) can vanish between the roots,
      - and rhox < ρcrit, where I changes sign.

    Returns
    -------
    list of tuples
        Each tuple: (rho0, rho1, rho2, rho_crit, rhox).
    """
    rho0_valid = []

    for rho0 in rho0_grid:
        coeffs = effective_potential_coefficients(l, rE, Qd, z, rho0)
        roots = np.roots(coeffs)
        real_roots = [r.real for r in roots if np.isreal(r) and 0 < r.real < 1]

        if len(real_roots) == 3: # sometimes,
            rho1, rho2 = sorted(real_roots)[1:]
        elif len(real_roots) == 2:
            rho1, rho2 = sorted(real_roots)
        else:
            continue

        rho_crit = rho0 * rE / Qd

        if rho1 < rho_crit < rho2:
            try:
                rhox = brentq(G_rhox, rho1 + EPS, rho2 - EPS,
                              args=(rho2, l, rE, Qd, z, rho0, rc))
                if rhox < rho_crit:
                    rho0_valid.append((rho0, rho1, rho2, rho_crit, rhox))
            except ValueError:
                continue

    return rho0_valid


# -------------------------------------------------------------------------
# Main computation & plotting
# -------------------------------------------------------------------------

def main():
    plt.figure(figsize=(8.27, 6))
    first_plot = True

    for rE, rE_color in zip(R_E_VALUES, R_E_COLORS):
        Qd = rE * np.sqrt(1 - 3 * rE**2 / L**2)
        rc = -rE + L * np.sqrt(1 - 2 * rE**2 / L**2)

        for z, z_color in zip(Z_VALUES, Z_COLORS):
            delta_rho_vals = []
            S_inst_vals = []
            S_AdS_vals = []

            rho0_grid = np.linspace(0.05, 1, 100)
            valid = find_valid_rho0_and_rhox(L, rE, Qd, z, rho0_grid, rc)

            for rho0, _, rho2, _, rhox in valid:
                a_val, _ = quad(worldline_integrand_first, rhox, rho2,
                                args=(L, rE, Qd, z, rho0),
                                limit=QUAD_LIMIT, points=[rhox, rho2, rE / rc])

                I_val, _ = quad(worldline_integrand_second, rhox, rho2,
                                args=(L, rE, Qd, z, rho0, rc),
                                limit=QUAD_LIMIT, points=[rhox, rho2, rE / rc])

                S_A_over_m = 2 * z**2 * Qd * I_val
                S_inst_over_Qdm = (a_val + S_A_over_m) / Qd
                S_AdS_over_Qdm = 2 * np.pi * (
                    (z**2 * rho0 - 1) / (z**2 * rho0**2 - 1)**1.5
                    + z
                    - (z**2 * rho0) / np.sqrt(z**2 * rho0**2 - 1)
                )

                delta_rho = 1 - rhox
                delta_rho_vals.append(delta_rho)
                S_inst_vals.append(S_inst_over_Qdm)
                S_AdS_vals.append(S_AdS_over_Qdm)

            if first_plot:
                plt.plot(delta_rho_vals, S_AdS_vals,
                         label=r"Analytical AdS$_2 \times $S$^2$ approximation",
                         linestyle='--', color=z_color)
                first_plot = False

            plt.plot(delta_rho_vals, S_inst_vals, 'o',
                     label=f"Numerical result, $r_E={rE}$",
                     color=rE_color, markersize=3)

    plt.xlabel(r'$\Delta \rho$', fontsize=14)
    plt.ylabel(r'$\frac{S_{\text{wl}}}{Q_E \tilde{m}}$', rotation=0,
               labelpad=20, fontsize=14)
    plt.grid(False)

    plt.text(0.05, 17, f'$l = {L}$\n$z = {Z_VALUES[0]}$',
             fontsize=10, color='k',
             bbox=dict(boxstyle='square', facecolor='none'))

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
