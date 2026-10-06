"""The asymmetric-error UV LF likelihood follows Barlow (2004).

Run with:  python -m pytest tests/test_likelihood_terms.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from likelihood_terms import asymmetric_gaussian_lnl as lnl

# Willott et al. (2024) points with asymmetric errors used in the fiducial runs
POINTS = [(10.0, 9.9, 5.7), (2.6, 3.4, 1.7), (4.8, 11.1, 4.1), (8.8, 7.1, 4.4)]


@pytest.mark.parametrize("obs,sp,sm", POINTS)
def test_peak_at_measurement(obs, sp, sm):
    grid = np.linspace(obs - 0.99 * sm, obs + 3 * sp, 20001)
    vals = [lnl(p, obs, sp, sm) for p in grid]
    assert lnl(obs, obs, sp, sm) == 0.0
    assert abs(grid[int(np.argmax(vals))] - obs) < 1e-3 * (sp + sm)


@pytest.mark.parametrize("obs,sp,sm", POINTS)
def test_minus_half_at_quoted_errors(obs, sp, sm):
    assert np.isclose(lnl(obs + sp, obs, sp, sm), -0.5)
    assert np.isclose(lnl(obs - sm, obs, sp, sm), -0.5)


def test_symmetric_errors_reduce_to_chi2():
    for pred in (0.5, 3.0, 7.7):
        assert np.isclose(lnl(pred, 4.0, 1.5, 1.5), -0.5 * ((pred - 4.0) / 1.5) ** 2)


def test_no_preference_for_larger_error_side():
    """The bug this replaces rewarded predictions on the larger-error side."""
    obs, sp, sm = 4.8, 11.1, 4.1
    assert np.isclose(lnl(obs + sp, obs, sp, sm), lnl(obs - sm, obs, sp, sm))
