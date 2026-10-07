"""Effective redshifts at which the model is compared to binned data.

UV LF: binned 1/V_a estimators (e.g. Willott et al. 2024) measure the LF
averaged over the comoving volume of the redshift bin (weighted by
completeness, which is not public), so the model is evaluated at the
volume-weighted mean redshift of the bin. Galaxy median redshifts are not the
right quantity: they are pulled to the low-z edge by the declining LF.

ACF: halomod evaluates the 3D clustering at one redshift and uses N(z) only
for the projection, so the model is evaluated at the Limber-weighted
redshift, weight N(z)^2 H(z).

Datasets not listed keep their nominal redshift.
"""
import os
from functools import lru_cache

import numpy as np
from astropy.cosmology import Planck18 as cosmo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Redshift-bin edges of the binned UV LFs. Dropout-selected samples without
# top-hat bins (Bouwens et al. 2021; Whitler et al. 2025, quoted at their median
# redshifts) are not listed and keep their nominal redshift.
UVLF_BINS = {
    # Willott et al. (2024), Table 3
    "UVLF_z8_Willot23": (7.5, 8.5),
    "UVLF_z9_Willot23": (8.5, 9.5),
    "UVLF_z10_Willot23": (9.5, 11.0),
    "UVLF_z12_Willot23": (11.0, 12.5),
    # McLeod et al. (2024), z ~ 11 bin
    "UVLF_z11_McLeod23": (9.5, 12.5),
    # Donnan et al. (2024), Sec. 5 / Table 3
    "UVLF_z9_Donnan24": (8.5, 9.5),
    "UVLF_z10_Donnan24": (9.5, 10.5),
    "UVLF_z11_Donnan24": (10.5, 11.5),
    "UVLF_z12_5_Donnan24": (11.5, 13.5),
    # Harikane et al. (2024a, 2024b), spectroscopic LFs, Table 6 of 2024b
    "UVLF_z7_Harikane24": (6.5, 7.5),
    "UVLF_z8_Harikane24": (7.5, 8.5),
    "UVLF_z9_Harikane24": (8.5, 9.5),
    "UVLF_z10_Harikane24": (9.5, 11.0),
    "UVLF_z12_Harikane24": (11.0, 13.5),
    "UVLF_z14_Harikane24": (13.5, 15.0),
    # Finkelstein et al. (2024), photometric-redshift samples (Sec. 3.4)
    "UVLF_z9_Finkelstein24": (8.5, 9.7),
    "UVLF_z11_Finkelstein24": (9.7, 13.0),
    "UVLF_z14_Finkelstein24": (13.0, 15.0),
}

# Redshift distributions of the ACF samples (as read by LikelihoodAngBase)
ACF_NZ = {
    "z9": "Nz_8_105_alt.csv",   # Paquereau+2025, 8 < z < 10.5
    "z7": "Nz_6_8_alt.csv",     # Paquereau+2025, 6 < z < 8
}


@lru_cache(maxsize=None)
def z_eff_volume(z1, z2, n=2001):
    """Comoving-volume-weighted mean redshift of the bin [z1, z2]."""
    z = np.linspace(z1, z2, n)
    dv = cosmo.differential_comoving_volume(z).value
    return float(np.trapezoid(z * dv, z) / np.trapezoid(dv, z))


@lru_cache(maxsize=None)
def z_eff_limber(nz_file, zmax=15.0, n=4000):
    """Limber-weighted redshift of an angular sample: weight N(z)^2 H(z)."""
    d = np.loadtxt(os.path.join(SCRIPT_DIR, nz_file), delimiter=",")
    order = np.argsort(d[:, 0])
    z = np.linspace(1e-2, zmax, n)
    nz = np.interp(z, d[order, 0], d[order, 1], left=0, right=0)
    w = nz ** 2 * cosmo.H(z).value
    return float(np.trapezoid(z * w, z) / np.trapezoid(w, z))


def uvlf_redshift(name, nominal):
    """Redshift at which to evaluate the model for UV LF dataset `name`."""
    if name in UVLF_BINS:
        return round(z_eff_volume(*UVLF_BINS[name]), 2)
    return nominal


def acf_redshift(key, nominal):
    """Redshift at which to evaluate the clustering model for ACF bin `key`."""
    if key in ACF_NZ:
        return round(z_eff_limber(ACF_NZ[key]), 2)
    return nominal


if __name__ == "__main__":
    for name, edges in UVLF_BINS.items():
        print(f"{name:20s} bin {edges}: z_eff = {uvlf_redshift(name, None)}")
    for key, f in ACF_NZ.items():
        print(f"ACF {key:3s} ({f}): z_eff = {acf_redshift(key, None)}")
