"""Scaling relations used when fitting simulations for the prior.

These mirror the functions in uvlf.py (ms_mh_flattening, SFMS, SFMS_new,
sigma_SFR_variable, sigma_SHMR_variable, sigma_linear_z) so that the
simulation fits live in exactly the same parameterization as the inference.
They are duplicated here, rather than imported, so the prior pipeline does
not pull in uvlf.py's numba/hmf dependencies; tests/test_prior_pipeline.py
checks that the two stay identical.
"""
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

F_BARYON = cosmo.Ob0 / cosmo.Om0
HIGH_MASS_SLOPE = 0.61  # fixed high-mass slope of the SHMR (uvlf.py)
M_PIVOT_SHMR = 1e10     # halo mass at which f_star = fstar_norm
M_PIVOT_SFMS = 1e10     # stellar mass above which sigma_SFMS is constant


def hubble_rate_per_yr(z):
    """H(z) in 1/yr."""
    return cosmo.H(z).to(u.yr ** (-1)).value


def ms_mh_flattening(mh, fstar_norm=1.0, alpha_star_low=0.5, M_knee=2.6e11):
    """Mean SHMR, M*(Mh), with a knee at M_knee (uvlf.ms_mh_flattening)."""
    mh = np.asarray(mh, dtype=np.float64)
    f_star = fstar_norm / ((mh / M_knee) ** (-alpha_star_low)
                           + (mh / M_knee) ** HIGH_MASS_SLOPE)
    f_star = f_star * ((M_PIVOT_SHMR / M_knee) ** (-alpha_star_low)
                       + (M_PIVOT_SHMR / M_knee) ** HIGH_MASS_SLOPE)
    return np.minimum(f_star, F_BARYON) * mh


def sfms(ms, t_star, z):
    """Mean SFMS, SFR = M* H(z) / t_star (uvlf.SFMS)."""
    return np.asarray(ms) * hubble_rate_per_yr(z) / t_star


def sfms_slope(ms, t_star, z, slope_SFR=1.0):
    """SFMS with a free slope, normalized at 10^9.5 Msun (uvlf.SFMS_new)."""
    b_sfr = -np.log10(t_star) + np.log10(hubble_rate_per_yr(z)) + 9.5
    return (np.asarray(ms) / 1e9) ** slope_SFR * 10 ** b_sfr


def sigma_sfms(ms, norm, a_sig_SFR):
    """Mass-dependent SFMS scatter, constant above 1e10 Msun
    (uvlf.sigma_SFR_variable)."""
    ms = np.asarray(ms, dtype=np.float64)
    sigma = a_sig_SFR * np.log10(ms / M_PIVOT_SFMS) + norm
    return np.where(ms > M_PIVOT_SFMS, norm, sigma)


def sigma_shmr_mass(mh, norm, a_sig_SHMR, M_char):
    """Mass-dependent SHMR scatter, constant above M_char
    (uvlf.sigma_SHMR_variable)."""
    mh = np.asarray(mh, dtype=np.float64)
    sigma = a_sig_SHMR * np.log10(mh / M_char) + norm
    return np.where(mh > M_char, norm, sigma)


def sigma_linear_z(sigma_0, alpha_z, z, z_ref=11.0):
    """Linear-in-redshift rescaling of a scatter (uvlf.sigma_linear_z)."""
    return sigma_0 * (1.0 + alpha_z * ((1.0 + z) / (1.0 + z_ref) - 1.0))
