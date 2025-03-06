import numpy as np
import halomod as hm
import hmf as hmf
import matplotlib.pyplot as plt
from hmf import cached_quantity, parameter, get_mdl
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.special import erfc
import pymultinest
from astropy.cosmology import Planck18 as cosmo

from numba import njit, prange
import ultranest


hmf_loc = hmf.MassFunction(z=11)
def ms_mh_flattening(mh, fstar_scale=1, alpha_star_low=0.5):
    """
        Get scaling relations for SHMR based on Davies+in prep.
        Parameters
        ----------
        mh: float,
            halo mass at which we're evaluating the relation.
        Returns
        ----------
        ms_mean: floats; optional,
            a and b coefficient of the relation.
    """
    f_star_mean = fstar_scale * 0.0076 * (2.6e11 / 1e10) ** alpha_star_low
    f_star_mean /= (mh / 2.6e11) ** (-alpha_star_low) + (mh / 2.6e11) ** 0.61
    return f_star_mean * mh

def ms_mh(ms, fstar_scale=1):
    mhs = np.logspace(5,15,500)
    mss = ms_mh_flattening(mhs, fstar_scale=fstar_scale)
    return 10**np.interp(np.log10(ms), np.log10(mss), np.log10(mhs))

def SFMS(Mstar, SFR_norm=1, z=9.25):
    """
        the functon returns SFR from Main sequence
    """
    b_SFR = -np.log10(0.43) + np.log10(cosmo.H(z).to(u.yr ** (-1)).value)

    return Mstar * 10 ** b_SFR * SFR_norm


def kUV(SFR):
    """
        Simplest transformation between SFR and Luv
    """
    return SFR / (1.15 * 1e-28)


def Muv_Luv(Luv):
    """
        Luv to Muv
    """
    return -2.5 * np.log10(Luv) + 51.6



@njit(parallel=True)
def uv_calc(
    Muv,
    masses_hmf,
    dndm,
    sigma_SFMS=0.1,
    sigma_SHMR=0.1,
    ms_obs_log = None,
    sfr_obs_log = None,
    msss=None,
    sfrs=None,
    muvs=None,
):
    #print(x_deg, y_deg)
    N_samples = int(1e5)
    log_mhs_int = np.random.uniform(
        7.0,
        14.0,
        N_samples,
    )
#     log_ms_int = np.random.uniform(
#         6.0,
#         12.0,
#         N_samples,
#     )
    ppred = 1 / np.sqrt(sigma_SFMS**2 + sigma_SHMR**2) * np.sqrt(2)
    #msss = np.interp(log_mhs_int, masses_hmf, np.log10(msss))

    log_ms_int = np.interp(log_mhs_int, masses_hmf, np.log10(msss))
    muvs_int = np.interp(log_mhs_int, masses_hmf, muvs)
    integral_sum = 0.0
    for i in prange(N_samples):  # Parallel loop
        dnd = np.interp(log_mhs_int[i], masses_hmf, dndm)
#         print(dnd)
        #exp = np.exp(-(Muv-muvs_int[i])**2/2/(sigma_SFMS**2 + sigma_SHMR**2))
        exp = np.exp(-(log_ms_int[i]-ms_obs_log)**2/2/(sigma_SFMS**2 + sigma_SHMR**2))
        integral_sum += dnd * ppred * exp
    return integral_sum / N_samples * 7


def UV_calc(
        Muv,
        masses_hmf,
        dndm,
        f_star_norm=1.0,
        alpha_star=0.5,
        sigma_SHMR=0.3,
        sigma_SFMS=0.3,
        t_star=0.5,
):
    msss = ms_mh_flattening(10 ** masses_hmf, alpha_star_low=alpha_star,
                            fstar_scale=f_star_norm)
    sfrs = SFMS(msss, SFR_norm=0.43 / t_star, z=11)
    muvs = Muv_Luv(kUV(sfrs))

    sfr_obs_log = np.interp(Muv, np.flip(muvs), np.flip(np.log10(sfrs)))
    ms_obs_log = np.interp(sfr_obs_log, np.log10(sfrs), np.log10(msss))

    uvlf = [uv_calc(
        muvi,
        masses_hmf,
        dndm,
        sigma_SFMS=sigma_SFMS,
        sigma_SHMR=sigma_SHMR,
        sfr_obs_log=sfr_obs_log[index],
        ms_obs_log=ms_obs_log[index],
        msss=msss,
        sfrs=sfrs,
        muvs=muvs,
    ) for index, muvi in enumerate(Muv)]

    return uvlf

def like_UV(fi, asi, s_sfri,s_shmri, tsti):
    McL_Muvs = np.array(
        [-22.57,-21.80,-20.80,-20.05,-19.55,-18.85,-18.23]
    )
    McL_uvlf = np.array(
        [0.012,0.128,1.251,3.951,9.713,23.490,63.080]
    )*1e-5
    Mcl_sig = np.array(
        [0.010,0.128,0.424,1.319, 4.170,9.190,28.650]
    )*1e-5
    lnL = 0
    preds = UV_calc(
        McL_Muvs,
        np.log10(hmf_loc.m),
        hmf_loc.dndlnm,
        f_star_norm=10**fi,
        alpha_star=asi,
        sigma_SHMR = s_shmri,
        sigma_SFMS = s_sfri,
        t_star = tsti,
    )
    for index, muvi in enumerate(McL_Muvs):
        lnL += -0.5 * (((preds[index] - McL_uvlf[index]) / Mcl_sig[
                index]) ** 2)
    return lnL