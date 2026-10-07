"""Effective redshifts used to compare the model with binned data."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from effective_redshift import acf_redshift, uvlf_redshift, z_eff_volume


def test_volume_weighted_mean_lies_inside_bin_and_above_centre_for_narrow_bins():
    for z1, z2 in [(7.5, 8.5), (9.5, 11.0), (11.0, 12.5)]:
        z = z_eff_volume(z1, z2)
        assert z1 < z < z2
        assert abs(z - 0.5 * (z1 + z2)) < 0.05


def test_willott_and_acf_values():
    assert uvlf_redshift("UVLF_z10_Willot23", 10) == 10.23
    assert uvlf_redshift("UVLF_z12_Willot23", 12) == 11.73
    assert uvlf_redshift("UVLF_z11_McLeod23", 11) == 10.94
    assert uvlf_redshift("UVLF_z11_Finkelstein24", 11) == 11.27
    assert uvlf_redshift("UVLF_z9_8_Whitler25", 9.8) == 9.8   # median-z sample: nominal
    assert uvlf_redshift("UVLF_z7_Bouwens21", 7) == 7         # dropout sample: nominal
    assert acf_redshift("z9", 9.25) == 9.02
    assert acf_redshift("z7", 7.0) == 7.29
