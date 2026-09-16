"""
Shared helpers for the BPASS-vs-blackbody comparison scripts
(compare_bpass_blackbody.py, compare_bpass_blackbody_galaxies.py, and the
metallicity/IMF variants). Factored out once the same fit code needed to be
kept consistent across four+ near-duplicate scripts -- in particular the
fit lower-wavelength bound, which must never creep back below the Lyman
limit (see FIT_LAMBDA_MIN_A below).

CANNOT BE RUN/TESTED LOCALLY beyond the fit-machinery itself (real BPASS
spectra are cluster-only) -- see the individual scripts' docstrings.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.optimize import curve_fit

LYMAN_LIMIT_A = 912.0
REST_1500_A = 1500.0

# Fit lower bound = the Lyman limit itself, not some arbitrary "close to
# zero" value. Below 912A the gas is optically thick to photoionization and
# the emergent spectrum has a sharp opacity edge -- a smooth blackbody
# cannot represent that by construction, so including those points in the
# fit only biases the fitted temperature toward whatever compromise best
# (mis-)fits an edge it has no way to reproduce. Fit only where a blackbody
# continuum is actually a sensible model.
FIT_LAMBDA_MIN_A = LYMAN_LIMIT_A
FIT_LAMBDA_MAX_A = 3000.0


def planck_lambda(wave_A, T, amplitude):
    """Planck function B_lambda(T), free overall amplitude. BPASS's native
    flux units don't match B_lambda's SI/cgs scale, so amplitude is always
    fit freely -- only T carries physical meaning here."""
    wave = (wave_A * u.AA).to(u.m).value
    h, c, k = const.h.value, const.c.value, const.k_B.value
    with np.errstate(over="ignore", divide="ignore"):
        x = h * c / (wave * k * T)
        bb = (2 * h * c**2 / wave**5) / np.expm1(x)
    return amplitude * bb


def fit_blackbody(wave_A, flux_raw, fit_lambda_min=FIT_LAMBDA_MIN_A,
                   fit_lambda_max=FIT_LAMBDA_MAX_A):
    """Fit a blackbody to flux_raw(wave_A) over [fit_lambda_min,
    fit_lambda_max], fitting in flux normalized to the peak of that window
    (fitting directly in BPASS's native units against Planck_lambda's SI
    scale badly scales the free parameters and the solver gets stuck at
    the initial guess -- caught by smoke-testing before this was ever run
    against real data).

    Returns (T_fit, amp_fit, flux_normalized, fit_mask) or (None, None,
    flux_normalized, fit_mask) if the fit fails to converge.
    """
    fit_mask = (
        (wave_A >= fit_lambda_min)
        & (wave_A <= fit_lambda_max)
        & (flux_raw > 0)
    )
    norm = flux_raw[fit_mask].max()
    flux = flux_raw / norm

    p0 = [4e4, flux[fit_mask].max() / planck_lambda(REST_1500_A, 4e4, 1.0)]
    try:
        popt, _ = curve_fit(
            planck_lambda, wave_A[fit_mask], flux[fit_mask],
            p0=p0, bounds=([1e3, 0], [3e5, np.inf]), maxfev=10000,
        )
        T_fit, amp_fit = popt
    except RuntimeError:
        return None, None, flux, fit_mask

    return T_fit, amp_fit, flux, fit_mask


def nearest_age_index(bp, target_age_yr):
    """SED column index (0-indexed) whose age (bp.ag[idx+1]) is closest to
    target_age_yr."""
    ages = bp.ag[1:]
    return int(np.argmin(np.abs(ages - target_age_yr)))