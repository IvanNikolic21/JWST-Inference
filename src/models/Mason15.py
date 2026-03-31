"""
A module defining model predictions from Mason+15 (and Mason+23, Nikolic+26) model.

The module includes functions that process the input parameters and returns UVLF.

"""

import hmf
import numpy as np


def pMuv_Mh(Muv, logMh, Muv_Mh_dict, sigmaUV_a=-0.34, sigmaUV_b=0.42, Muv_add=0, return_med_sigma=False):
    """
    sigmaUV (Mh) = a (log Mh - 12) + b.
    a = −0.34 and b = 0.42,
    """
    sigmaUV = sigmaUV_a * (logMh - 12) + sigmaUV_b
    # Ensure sigmaUV is strictly positive to define a valid Gaussian PDF
    sigmaUV_floor = 1e-3
    if np.isscalar(sigmaUV):
        sigmaUV = max(sigmaUV, sigmaUV_floor)
    else:
        sigmaUV = np.maximum(sigmaUV, sigmaUV_floor)

    # Get the Muv-Mh relation for the given redshift
    Muv_Mh_med = np.array(
        [Muv_from_logMh(np.round(np.float64(lM), 1), Muv_Mh_dict, use_scatter=False, dust=True) for lM in
         logMh]) + Muv_add

    if return_med_sigma:
        return Muv_Mh_med, sigmaUV

    else:
        # Calculate the probability density function (PDF) of Muv given logMh
        pdf = (1 / (np.sqrt(2 * np.pi) * sigmaUV[:, None])) * np.exp(
            -0.5 * ((Muv - Muv_Mh_med[:, None]) / sigmaUV[:, None]) ** 2)

        return pdf

def Muv_from_logMh(logMh, Muv_Mh_dict, use_scatter=True, sigmaUV=0.3, dust=False):
    if use_scatter:
        scatter = np.random.normal(0, sigmaUV)
    else:
        scatter = 0
    if dust:
        return Muv_Mh_dict[logMh][1] + scatter
    return Muv_Mh_dict[logMh][0] + scatter

def calculate_uvlf(Muv_shift, sigma_UV_a, sigma_UV_b , mf = None, Muv_grid = None, z = None, Muv_Mh_dict = None):
    if z is None:
        z = 10.0
    if mf is None:
        mf = hmf.MassFunction(
            z=z,
            Mmin=8,
            Mmax=14,
            dlog10m=0.05,
        )
    if Muv_grid is None:
        Muv_grid = np.linspace(-25, -13, 100)

    if z==10.0 and Muv_Mh_dict is None :
        Muv_Mh_file = '/groups/astro/ivannik/notebooks/clustering_project/Muv_Mh_z=10.txt'
        Muv_Mh = np.genfromtxt(Muv_Mh_file, dtype=None, names=True)
        Muv_Mh_dict = {Muv_Mh['logMh'][i]: [Muv_Mh['Muv'][i], Muv_Mh['Muv_dust'][i]] for i in range(len(Muv_Mh))}


    pmuvmh = pMuv_Mh(Muv_grid, np.log10(mf.m), Muv_Mh_dict, sigmaUV_a=sigma_UV_a, sigmaUV_b=sigma_UV_b, Muv_add=Muv_shift)
    UVLF_stochier = np.trapezoid(pmuvmh.T * mf.dndm, mf.m, axis=1)
    return UVLF_stochier

class Mason15(object):
    def __init__(self, z=10.0, hm_inst = None):
        self.z = z

        if z==10.0:
            Muv_Mh_file = '/groups/astro/ivannik/notebooks/clustering_project/Muv_Mh_z=10.txt'
            Muv_Mh = np.genfromtxt(Muv_Mh_file, dtype=None, names=True)
            self.Muv_Mh_dict = {Muv_Mh['logMh'][i]: [Muv_Mh['Muv'][i], Muv_Mh['Muv_dust'][i]] for i in range(len(Muv_Mh))}

    def calculate_UVLF(self, Muv_shift, sigma_UV_a, sigma_UV_b, Muv_grid = np.linspace(-25, -13, 100)):
        UVLF_pred = calculate_uvlf(
            Muv_shift, sigma_UV_a, sigma_UV_b, Muv_grid = Muv_grid, z=self.z, Muv_Mh_dict = self.Muv_Mh_dict
        )
        return UVLF_pred