"""Cosmology used by every halo mass function and halo model in this project.

astropy's Planck18 is the Planck 2018 "TT,TE,EE+lowE+lensing+BAO" parameter
set (Planck Collaboration 2020, A&A 641, A6, Table 2, last column), whose
sigma_8 is 0.8102. hmf/halomod take sigma_8 separately (their default, 0.8159,
is the Planck 2015 value), so both are passed together via HMF_COSMO_KW.
"""
from astropy.cosmology import Planck18 as cosmo

SIGMA_8 = 0.8102
HMF_COSMO_KW = dict(cosmo_model=cosmo, sigma_8=SIGMA_8)
