"""The Sec. 5.1 extensions leave the fiducial model unchanged and behave as defined.

Run with:  python -m pytest tests/test_model_extensions.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
uvlf = pytest.importorskip("uvlf")

MH = 10 ** np.linspace(8, 13, 200)
FSTAR, ALPHA, MKNEE, TSTAR = 10 ** -2.5, 0.5, 2e12, 0.15


def new_relations(z, **ext):
    """The two lines the forward models now use."""
    f, a = uvlf.shmr_params_at_z(FSTAR, ALPHA, z, **{k: v for k, v in ext.items()
                                                      if k != "slope_SFR"})
    ms = uvlf.ms_mh_flattening(MH, uvlf.cosmo, alpha_star_low=a, fstar_norm=f,
                               M_knee=MKNEE)
    return ms, uvlf.SFMS_slope(ms, SFR_norm=TSTAR, z=z,
                               slope_SFR=ext.get("slope_SFR", 1.0))


def old_relations(z):
    """The two lines the forward models used before the extensions."""
    ms = uvlf.ms_mh_flattening(MH, uvlf.cosmo, alpha_star_low=ALPHA,
                               fstar_norm=FSTAR, M_knee=MKNEE)
    return ms, uvlf.SFMS(ms, SFR_norm=TSTAR, z=z)


@pytest.mark.parametrize("z", [6.0, 8.0, 10.0, 12.5])
def test_fiducial_bitwise_unchanged(z):
    for new, old in zip(new_relations(z), old_relations(z)):
        assert np.array_equal(new, old)


def test_fiducial_bitwise_unchanged_with_explicit_null_extensions():
    for new, old in zip(new_relations(9.0, alpha_fstar_z=0.0, alpha_star_z=0.0,
                                      slope_SFR=1.0), old_relations(9.0)):
        assert np.array_equal(new, old)


def test_extensions_anchored_at_z10():
    """At the pivot redshift every extension reduces to the fiducial SHMR."""
    for new, old in zip(new_relations(10.0, alpha_fstar_z=1.3, alpha_star_z=-0.7),
                        old_relations(10.0)):
        np.testing.assert_allclose(new, old, rtol=1e-12)


def test_fstar_z_scaling():
    z, a = 6.0, 1.5
    f, alpha = uvlf.shmr_params_at_z(FSTAR, ALPHA, z, alpha_fstar_z=a)
    assert np.isclose(f, FSTAR * ((1 + z) / 11) ** a) and alpha == ALPHA


def test_alpha_star_z_pivot():
    z, a = 14.0, 0.4
    f, alpha = uvlf.shmr_params_at_z(FSTAR, ALPHA, z, alpha_star_z=a)
    assert f == FSTAR and np.isclose(alpha, ALPHA + a * ((1 + z) / 11 - 1))


def test_sfms_slope_pivots_at_10_9():
    ms = np.array([10 ** 8.0, 10 ** 9.0, 10 ** 10.0])
    base = uvlf.SFMS(ms, SFR_norm=TSTAR, z=8)
    sl = uvlf.SFMS_slope(ms, SFR_norm=TSTAR, z=8, slope_SFR=0.8)
    np.testing.assert_allclose(sl / base, [10 ** 0.2, 1.0, 10 ** -0.2])
