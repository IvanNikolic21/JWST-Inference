"""Per-point likelihood terms for observed luminosity functions.

Kept free of heavy dependencies so they can be unit-tested on their own.
"""
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
