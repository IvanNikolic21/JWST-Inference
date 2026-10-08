"""Ionizing photon rate from BPASS spectra (bpass_loader.get_Qion_sfr10).

Uses a synthetic loader with the BPASS layout (1 A wavelength bins starting at
1 A, 51 age bins, 13 metallicities), since the spectra live on the cluster.

Run with:  python -m pytest tests/test_qion.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
uvlf = pytest.importorskip("uvlf")

N_WV = 2000
L_ION = 1e5          # Lsun/A per 1e6 Msun below the Lyman limit, youngest bin


class FakeSampler:
    """Constant SFR over the last 100 Myr, nothing before."""

    def get_SFH_const(self, Mstar, SFR):
        return np.full(20, float(SFR)), 20


def fake_loader(metal_scaling=None):
    bp = object.__new__(uvlf.bpass_loader)
    bp.metal_avail = np.array([1e-5, 1e-4, 1e-3, 0.002, 0.003, 0.004,
                               0.006, 0.008, 0.01, 0.014, 0.02, 0.03, 0.04])
    bp.wv = np.linspace(1, N_WV, N_WV)
    bp.ag = np.array([0] + [10 ** (6.05 + 0.1 * i) for i in range(1, 52)])
    bp.ages = 52
    sed = np.zeros((13, 51, N_WV))
    sed[:, 0, :911] = L_ION                       # only the youngest bin ionizes
    if metal_scaling is not None:
        sed *= metal_scaling[:, None, None]
    bp.SEDS = sed
    return bp


def oh_for_mass_fraction(zm):
    """12+log(O/H) that get_Qion_sfr10 maps onto mass fraction zm."""
    return np.log10(zm * 10 ** 0.42 / 0.02) + 8.69


def expected_q(sfr, scale=1.0):
    lam = np.arange(1, 912)
    q_per_1e6 = L_ION * scale * np.sum(lam) * uvlf.L_SUN_ERG_S / uvlf.HC_ERG_ANG
    return q_per_1e6 * sfr * 10 ** 6.15 / 1e6     # mass formed in bin 0, units 1e6 Msun


def test_lyman_limit_cut_and_units():
    bp = fake_loader()
    q = bp.get_Qion_sfr10(oh_for_mass_fraction(0.001), 1e8, 1.0, 10.0,
                          SFH_samp=FakeSampler(), sfr_10=1.0)
    assert np.isclose(q, expected_q(1.0), rtol=1e-10)


def test_photons_above_lyman_limit_do_not_count():
    bp = fake_loader()
    bp.SEDS[:, :, 911:] = 1e9
    q = bp.get_Qion_sfr10(oh_for_mass_fraction(0.001), 1e8, 1.0, 10.0,
                          SFH_samp=FakeSampler(), sfr_10=1.0)
    assert np.isclose(q, expected_q(1.0), rtol=1e-10)


def test_linear_in_sfr_and_burst_replaces_recent_sfr():
    bp = fake_loader()
    args = (oh_for_mass_fraction(0.001), 1e8)
    q1 = bp.get_Qion_sfr10(*args, 1.0, 10.0, SFH_samp=FakeSampler(), sfr_10=1.0)
    q3 = bp.get_Qion_sfr10(*args, 3.0, 10.0, SFH_samp=FakeSampler(), sfr_10=3.0)
    qb = bp.get_Qion_sfr10(*args, 1.0, 10.0, SFH_samp=FakeSampler(), sfr_10=3.0)
    assert np.isclose(q3, 3 * q1) and np.isclose(qb, 3 * q1)


def test_burst_window_is_last_10_myr():
    """sfr_10 sets the SFR of the BPASS bins younger than 11.2 Myr only."""
    bp = fake_loader()
    bp.SEDS[:, :, :911] = L_ION                   # all ages ionize
    args = (oh_for_mass_fraction(0.001), 1e8, 1.0, 10.0)
    base = bp.get_Qion_sfr10(*args, SFH_samp=FakeSampler(), sfr_10=1.0)
    off = bp.get_Qion_sfr10(*args, SFH_samp=FakeSampler(), sfr_10=0.0)
    assert uvlf.BURST_WINDOW_YR == 1e7
    dt = np.diff(bp.ag)
    young = dt[:10].sum() / dt[:20].sum()         # 0-11.2 Myr share of the 0-112 Myr mass
    assert np.isclose(off / base, 1 - young)
    q_100 = bp.get_Qion_sfr10(*args, SFH_samp=FakeSampler(), sfr_10=0.0, burst_window=1e8)
    assert q_100 == 0.0


def test_get_UV_sfr10_uses_the_same_window():
    bp = fake_loader()
    bp.SEDS[:, :, 1449:1549] = 1.0                # flat UV at all ages
    oh = oh_for_mass_fraction(0.001)

    class Sampler(FakeSampler):
        pass
    l_base = bp.get_UV_sfr10(oh, 1e8, 1.0, 10.0, SFH_samp=Sampler(), sfr_10=1.0)
    l_off = bp.get_UV_sfr10(oh, 1e8, 1.0, 10.0, SFH_samp=Sampler(), sfr_10=0.0)
    dt = np.diff(bp.ag)
    assert np.isclose(l_off / l_base, 1 - dt[:10].sum() / dt[:20].sum(), rtol=1e-6)


def test_metallicity_interpolation_is_exact_at_nodes_and_loglinear_between():
    scale = np.logspace(0, -1, 13)                # Q drops 10x across the grid
    bp = fake_loader(metal_scaling=scale)
    for k in (0, 4, 9):
        q = bp.get_Qion_sfr10(oh_for_mass_fraction(bp.metal_avail[k]), 1e8, 1.0, 10.0,
                              SFH_samp=FakeSampler(), sfr_10=1.0)
        assert np.isclose(q, expected_q(1.0, scale[k]), rtol=1e-8)
    zm = np.sqrt(0.002 * 0.003)
    q = bp.get_Qion_sfr10(oh_for_mass_fraction(zm), 1e8, 1.0, 10.0,
                          SFH_samp=FakeSampler(), sfr_10=1.0)
    assert np.isclose(q, expected_q(1.0, np.sqrt(scale[3] * scale[4])), rtol=1e-8)
