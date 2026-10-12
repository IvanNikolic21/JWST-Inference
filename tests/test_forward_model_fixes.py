"""Forward-model fixes of 2026-10-12 (BPASS age bins, SFH, SFMS scatter, HOD units).

Run with:  python -m pytest tests/test_forward_model_fixes.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
uvlf = pytest.importorskip("uvlf")


def test_bpass_age_edges_follow_the_bpass_convention():
    """Spectrum at log(age) = a covers 10^(a-0.05)-10^(a+0.05); the first starts at 0."""
    e = uvlf.BPASS_AGE_EDGES
    assert e.size == 52 and e[0] == 0.0
    assert np.isclose(e[1], 10 ** 6.05) and np.isclose(e[-1], 10 ** 11.05)
    log_age = 6.0 + 0.1 * np.arange(51)
    assert np.all((10 ** log_age[1:] > e[1:-1]) & (10 ** log_age[1:] < e[2:]))


def test_burst_window_is_bins_up_to_log_age_7():
    e = uvlf.BPASS_AGE_EDGES
    n = np.sum(e[:-1] < uvlf.BURST_WINDOW_YR)
    assert n == 11 and np.isclose(e[n], 10 ** 7.05)       # spectra 10^6.0 ... 10^7.0


@pytest.fixture(scope="module")
def sampler():
    return uvlf.SFH_sampler(z=10.0)


def test_sfh_before_100_myr_is_not_compounded(sampler):
    ms, tstar = 1e9, 0.28
    sfr = ms * sampler.Hubble_now / tstar
    sfh, _ = sampler.get_SFH_const(ms, sfr)
    ages = sampler.ages_SFH
    for i in range(len(sfh)):
        if ages[i] < 1e8:
            assert sfh[i] == sfr
        else:
            want = sfr * np.exp(-sampler.hubb_diffs[i] / tstar) * sampler.Hubbles[i] / sampler.Hubble_100
            assert np.isclose(sfh[i], want, rtol=1e-12)


@pytest.mark.parametrize("a", [0.05, 0.2])
def test_mass_dependent_scatters_stay_positive(a):
    ms = np.logspace(2, 12, 50)
    assert uvlf.sigma_SFR_variable(ms, norm=0.2, a_sig_SFR=a).min() >= uvlf.SIGMA_FLOOR
    assert uvlf.sigma_SHMR_variable(ms * 100, norm=0.2, a_sig_SHMR=a).min() >= uvlf.SIGMA_FLOOR


def test_p_sfr_given_mstar_uses_scatter_at_mstar():
    """Each row is a normalised Gaussian with the row's scatter and mean on the SFMS."""
    sfr = np.linspace(-5, 5, 20001)
    mu = np.array([-1.0, 0.5, 2.0])
    sig = np.array([0.6, 0.35, 0.2])
    p = uvlf._gauss_sfr_mstar(sfr, mu, sig)
    dx = sfr[1] - sfr[0]
    assert np.allclose(p.sum(axis=1) * dx, 1.0, atol=1e-6)
    assert np.allclose((p * sfr).sum(axis=1) * dx, mu, atol=1e-6)
    sd = np.sqrt((p * (sfr - mu[:, None]) ** 2).sum(axis=1) * dx)
    assert np.allclose(sd, sig, rtol=1e-5)


def test_metallicity_interpolation_is_loglog_and_exact_at_nodes():
    grid = np.array([1e-5, 1e-4, 1e-3, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.014])
    vals = 1e-6 * grid ** -0.2
    for z, v in zip(grid, vals):
        assert np.isclose(uvlf.interp_metallicity(grid, vals, z), v, rtol=1e-12)
    zm = np.sqrt(1e-4 * 1e-3)
    assert np.isclose(uvlf.interp_metallicity(grid, vals, zm), 1e-6 * zm ** -0.2, rtol=1e-12)


def test_hod_uses_msun_not_msun_over_h():
    ulty = pytest.importorskip("ulty")      # needs halomod (cluster only)
    hod = object.__new__(ulty.My_HOD)
    hod.params = dict(ulty.My_HOD._defaults, stellar_mass_min=9.0, fstar_norm=10 ** -2.67,
                      alpha_star_low=0.53, M_knee=2e12)
    m_h = np.logspace(10, 13, 300)                       # halomod masses, Msun/h
    p = hod._p_above_threshold(m_h, 0.3)
    m_sun = m_h / uvlf.cosmo.h
    mu = np.log10(uvlf.ms_mh_flattening(m_sun, uvlf.cosmo, fstar_norm=10 ** -2.67,
                                        alpha_star_low=0.53, M_knee=2e12))
    # 50% point of N_cen is where the Msun-based SHMR reaches 1e9
    m50 = np.interp(0.5, p, m_sun)
    assert abs(np.log10(m50) - np.interp(9.0, mu, np.log10(m_sun))) < 0.02
