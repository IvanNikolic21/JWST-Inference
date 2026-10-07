"""Per-point likelihood terms for observed luminosity functions.

Kept free of heavy dependencies so they can be unit-tested on their own.
"""
from functools import lru_cache

import numpy as np


def asymmetric_gaussian_lnl(pred, obs, sig_plus, sig_minus):
    """ln L of a model prediction for a measurement with asymmetric errors.

    Variable-width Gaussian with a linear sigma, Barlow (2004,
    arXiv:physics/0406120, Eqs. 14-16):

        ln L = -1/2 [(pred - obs) / (sigma + sigma' (pred - obs))]^2,
        sigma = 2 s+ s- / (s+ + s-),   sigma' = (s+ - s-) / (s+ + s-).

    It peaks at obs and equals -1/2 at obs + s+ and obs - s-. There is
    deliberately no -ln(sigma) normalisation term: sigma depends on the
    prediction, and including it would shift the peak away from obs (Barlow,
    Sec. "Variable Gaussian (1)"). Constant terms are omitted; they cancel in
    posteriors and in Bayes factors computed on the same data.
    """
    sig_a = 2.0 * sig_plus * sig_minus / (sig_plus + sig_minus)
    sig_b = (sig_plus - sig_minus) / (sig_plus + sig_minus)
    sigma = max(abs(sig_a + sig_b * (pred - obs)), 1e-12)
    return -0.5 * ((pred - obs) / sigma) ** 2


@lru_cache(maxsize=8)
def rectangle_pair_separation_pdf(x_deg, y_deg, n_grid=1500, n_bins=4000):
    """Distribution of the separation of two independent uniform points in an
    x_deg x y_deg rectangle (flat sky). |dx| and |dy| are independent with
    triangular densities 2(a - u)/a^2, so the joint density is evaluated on an
    n_grid x n_grid midpoint grid and binned in separation.

    Returns (separation bin centres [rad], probability per bin), summing to 1.
    """
    a, b = x_deg, y_deg
    u = (np.arange(n_grid) + 0.5) / n_grid
    dx, dy = np.meshgrid(u * a, u * b, indexing="ij")
    weight = ((a - dx) * (b - dy)).ravel()
    r = np.hypot(dx, dy).ravel()
    edges = np.linspace(0.0, np.hypot(a, b), n_bins + 1)
    prob, _ = np.histogram(r, bins=edges, weights=weight)
    prob /= prob.sum()
    centres = 0.5 * (edges[1:] + edges[:-1])
    return np.deg2rad(centres), prob


def integral_constraint_rectangle(theta_rad, w_theta, x_deg, y_deg):
    """Integral constraint w_IC = (1/Omega^2) int int w dOmega_1 dOmega_2 for an
    x_deg x y_deg rectangular field: the average of the model w(theta) over all
    pairs of points in the field. Deterministic (replaces the Monte Carlo
    estimate in ulty.w_IC). theta_rad must be increasing."""
    r, prob = rectangle_pair_separation_pdf(float(x_deg), float(y_deg))
    return float(np.sum(prob * np.interp(r, theta_rad, w_theta)))


def integral_constraint_rr(theta_bins_deg, rr_pairs, theta_rad, w_theta):
    """Integral constraint from random-pair counts in angular bins (Roche &
    Eales 1999): sum_i w(theta_i) RR_i / sum_i RR_i. Only equal to the
    field-wide integral if the bins cover all separations in the field."""
    w_i = np.interp(np.deg2rad(theta_bins_deg), theta_rad, w_theta)
    return float(np.sum(w_i * rr_pairs) / np.sum(rr_pairs))
