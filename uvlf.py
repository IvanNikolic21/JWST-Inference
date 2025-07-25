import numpy as np
import halomod as hm
import hmf as hmf
import matplotlib.pyplot as plt
from hmf import cached_quantity, parameter, get_mdl
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.special import erfc
import pymultinest
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from numba import njit, prange
import ultranest
from astropy import constants as const
from astropy.cosmology import z_at_value
from multiprocessing import Pool
from scipy.interpolate import splrep, BSpline
import scipy.integrate as intg

#hmf_loc = hmf.MassFunction(z=11)
def ms_mh_flattening(mh, fstar_norm = 1.0, alpha_star_low = 0.5, M_knee=2.6e11):
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

    f_star_mean = fstar_norm#fstar_scale * 0.0076 * (2.6e11 / 1e10) ** alpha_star_low
    f_star_mean /= (mh / M_knee) ** (-alpha_star_low) + (mh / M_knee) ** 0.61
    if M_knee != 2.6e11:
        f_star_mean *= (1e10 / M_knee) ** (-alpha_star_low) + (1e10 / M_knee) ** 0.61
    return f_star_mean * mh

def ms_mh(ms, fstar_norm=1, alpha_star_low=0.5, M_knee=2.6e11):
    """
        Get inverse of the SHMR relation
        Parameters
        ----------
        ms: float,
            stellar mass at which we're evaluating the relation.
        Returns
        ----------
        mh_mean: floats; optional,
            mh of the relation
    """
    mhs = np.logspace(5,18,1000)
    mss = ms_mh_flattening(mhs, fstar_norm=fstar_norm, alpha_star_low=alpha_star_low, M_knee=M_knee)
    return 10**np.interp(np.log10(ms), np.log10(mss), np.log10(mhs))

def SFMS(Mstar, SFR_norm = 1., z=9.25):
    """
        the functon returns SFR from Main sequence
    """
    b_SFR = -np.log10(SFR_norm) + np.log10(cosmo.H(z).to(u.yr ** (-1)).value)

    return Mstar * 10 ** b_SFR #* SFR_norm


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


def sigma_SFR_variable(Mstar, norm=0.18740570999999995, a_sig_SFR=-0.11654893):
    """
        Variable scatter of SFR-Mstar relation.
        It's based on FirstLight database.
    Parameters
    ----------
    Mstar: stellar mass at which the relation is taken

    Returns
    -------
    sigma: sigma of the relation
    """
    # a_sig_SFR = -0.11654893
    #b_sig_SFR = 1.35289501
    #     sigma = a_sig_SFR * np.log10(Mstar) + b_sig_SFR

    Mstar = np.asarray(Mstar)  # Convert input to a numpy array if not already

    sigma = a_sig_SFR * np.log10(Mstar/1e10) + norm
    sigma[Mstar > 10 ** 10] = norm

    return sigma


#     if Mstar > 10**10:
#         return 0.18740570999999995 - norm
#     else:
#         return sigma - norm

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
    N_samples = int(1e4)
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
    #msss = np.interp(log_mhs_int, masses_hmf, np.log10(msss))

    log_ms_int = np.interp(log_mhs_int, masses_hmf, np.log10(msss))
    sig_int = np.interp(log_ms_int, np.log10(msss), sigma_SFMS)
    muvs_int = np.interp(log_mhs_int, masses_hmf, muvs)
    integral_sum = 0.0
    for i in prange(N_samples):  # Parallel loop
        dnd = np.interp(log_mhs_int[i], masses_hmf, dndm)
        ppred = 1 / np.sqrt(sig_int[i]**2 + sigma_SHMR**2) * np.sqrt(2)

#         print(dnd)
        #exp = np.exp(-(Muv-muvs_int[i])**2/2/(sigma_SFMS**2 + sigma_SHMR**2))
        exp = np.exp(-(log_ms_int[i]-ms_obs_log)**2/2/(sig_int[i]**2 + sigma_SHMR**2))
        integral_sum += dnd * ppred * exp
    return integral_sum / N_samples * 7


def UV_calc(
        Muv,
        masses_hmf,
        dndm,
        f_star_norm=1.0,
        alpha_star=0.5,
        sigma_SHMR=0.3,
        sigma_SFMS_norm=0.0,
        t_star=0.5,
        a_sig_SFR = -0.11654893,
        z=11,
        M_knee=2.6e11,
):
    msss = ms_mh_flattening(10**masses_hmf, alpha_star_low=alpha_star,
                            fstar_norm=f_star_norm, M_knee=M_knee)
    sfrs = SFMS(msss, SFR_norm=t_star, z=z)
    muvs = Muv_Luv(kUV(sfrs))
    sfr_obs_log = np.interp(Muv, np.flip(muvs), np.flip(np.log10(sfrs)))
    ms_obs_log = np.interp(sfr_obs_log, np.log10(sfrs), np.log10(msss))
    sigma_SFMS_var = sigma_SFR_variable(msss, norm = sigma_SFMS_norm, a_sig_SFR=a_sig_SFR)
    uvlf = [uv_calc(
        muvi,
        masses_hmf,
        dndm,
        sigma_SFMS=sigma_SFMS_var,
        sigma_SHMR=sigma_SHMR,
        sfr_obs_log=sfr_obs_log[index],
        ms_obs_log=ms_obs_log[index],
        msss=msss,
        sfrs=sfrs,
        muvs=muvs,
    ) for index, muvi in enumerate(Muv)]

    return uvlf


def metalicity_from_FMR(M_star, SFR):
    """
    metalicity from Curti+19

    -----

    Function takes in the stellar mass and SFR and outputs the metallicity 12+log(O/H)
    """
    Z_0 = 8.779
    gamma = 0.31
    beta = 2.1
    m_0 = 10.11
    m_1 = 0.56
    M_0 = 10 ** (m_0 + m_1 * np.log10(SFR))
    return Z_0 - gamma / beta * np.log10(1 + (M_star / M_0) ** (-beta))


def OH_to_mass_fraction(Z_OH):
    """
    Convert 12+log(O/H) metalicty to mass fraction one.
    Very important note! So far I haven't accounted for solar metallicity being
    0.02!
    """
    return 10 ** (Z_OH - 8.69) * 0.02


def DeltaZ_z(z):
    """
        Evolution of the normalization of FMR. Based on Curti+23.
    Parameters
    ----------
    z: redshift

    Returns
    -------
    Delta Z: offset from FMR
    """
    a_d = -0.0553952
    b_d = 0.0635493
    return a_d * z + b_d
ang_to_hz = 1/const.c.cgs.value * 1500**2 * 1e-8

class SFH_sampler:
    """
        Class that contains Hubble integrals and derivations necessary for SFH
        calculation. The only reason this is a class is the speed-up.
    """
    def __init__(self,z):
        self.Hubble_now = cosmo.H(z).to(u.yr**-1).value
        self.ages_SFH = np.array([0] + [10**(6.05 + 0.1 * i) for i in range(1,52)])
        self.maximum_time = cosmo.lookback_time(30).to(u.yr).value - cosmo.lookback_time(z).to(u.yr).value
        self.lookback_time_100 = cosmo.lookback_time(z).value + 1e8
        self.Hubble_100 = cosmo.H(
                            z_at_value(
                                cosmo.lookback_time,
                                cosmo.lookback_time(z) + 100 * u.Myr
                            )
                        ).to(
                            u.yr ** -1
                        ).value
        self.Hubbles = []
        self.z_ages = []
        self.hubb_diffs = []
        for self.index_age, age in enumerate(self.ages_SFH):
            if age>self.maximum_time:
                self.index_age -= 1
                break
            z_age = z_at_value(cosmo.lookback_time, cosmo.lookback_time(z) + age * u.yr)
            Hubble_age = cosmo.H(z_age).to(u.yr**-1).value
            self.Hubbles.append(Hubble_age)
            self.z_ages.append(z_age)
            self.hubb_diffs.append(
                intg.quad(
                    lambda x: (
                        cosmo.H(
                            z_at_value(
                                cosmo.lookback_time,
                                cosmo.lookback_time(z) + x * u.Myr
                            )
                        ).to(
                            u.Myr ** -1
                        ).value
                    ),
                    a=100,
                    b=age / 10 ** 6
                )[0]
            )
        self.Hubbles = np.array(self.Hubbles)

    def get_SFH_exp(self, Mstar, SFR):
        """
            Generate SFH using Mstar and SFR.
        """
        t_STAR = Mstar / (SFR * self.Hubble_now**-1)
        SFR_now = SFR
        SFH = [SFR]
        for index,Hub in enumerate(self.Hubbles):
            SFH.append(SFR_now * np.exp(-(self.ages_SFH[index+1]/t_STAR) * Hub))
            SFR_now = SFH[index]
            Mstar -= SFH[-1] * (self.ages_SFH[index+1] - self.ages_SFH[index])
        return np.array(SFH), self.index_age

    def get_SFH_const(self, Mstar, SFR):
        """
            Generate SFH using Mstar and SFR.
        """
        t_STAR = Mstar / (SFR * self.Hubble_now**-1)
        SFR_now = SFR
        SFH = [SFR] * len(self.ages_SFH[self.ages_SFH<1e8]) #all SFH will be below 100Myr
        #stochasticity below 100Myr will be explicitly accounted for
        mass_formed = SFR * 1e8
        mass_remaining = Mstar - mass_formed
        for index,Hub in enumerate(self.Hubbles):
            if self.ages_SFH[index] < 1e8:
                continue
            #print(index, self.hubb_diffs[index], Hub/self.Hubble_100)
            #SFH.append(SFR_now * np.exp(-(self.ages_SFH[index]/t_STAR) * Hub))
            SFH.append(SFR_now * np.exp(-(1 / t_STAR) * self.hubb_diffs[index]) * Hub/self.Hubble_100)
            SFR_now = SFH[index]
            mass_remaining -= SFH[-1] * (
                self.ages_SFH[index] - self.ages_SFH[index-1]
            )
            if mass_remaining < 0:
                break
            #print(Mstar/(SFH[-1]* Hub**-1))
        return np.array(SFH), self.index_age

def wv_to_freq(wvs):
    """
    Converts a wavelength to frequency.
    Input
    ----------
        wvs: scalar or ndarray-like.
            wavelengths in Angstroms.
    Output:
        freq: scalar of ndarray-like.
            frequencies in Hertz.
    """

    return const.c.cgs.value / (wvs * 1e-8)


def reader(name):
    """
    Function that reads and splits a string into separate strings.
    """
    return open(name).read().split('\n')


def splitter(fp):
    """Function that splits SEDs. Useful for parallelling."""
    wv_b = int(1e5)
    return [
        [float(fp[i].split()[j]) for i in range(wv_b)] for j in range(1, 52)
    ]


def get_SFH_exp(Mstar, SFR, z):
    """
    Get SFH is based on the t_STAR parameter and exponental SFH.
    Single instance version.
    """
    Hubble_now = cosmo.H(z).to(u.yr ** -1).value
    t_STAR = Mstar / (SFR * Hubble_now ** -1)
    # setting maximum time for BPASS
    ages = np.array([0] + [10 ** (6.05 + 0.1 * i) for i in range(1, 52)])
    maximum_time = cosmo.lookback_time(30).to(
        u.yr).value - cosmo.lookback_time(z).to(u.yr).value
    SFH = []

    for index_age, age in enumerate(ages):
        z_age = z_at_value(cosmo.lookback_time,
                           cosmo.lookback_time(z) + age * u.yr)
        Hubble_age = cosmo.H(z_age).to(u.yr ** -1).value
        Hubble_integral = intg.quad(lambda x: (cosmo.H(
            z_at_value(cosmo.lookback_time,
                       cosmo.lookback_time(z) + x * u.Myr)).to(
            u.Myr ** -1).value), 0, age / 10 ** 6)[0]
        exp_term = np.exp(-(1 / t_STAR) * Hubble_integral)
        SFH.append(SFR * exp_term * Hubble_age / Hubble_now)
        if age > maximum_time:
            index_age -= 1
            break
    return SFH, index_age

class bpass_loader:
    """
    This Class contains all of the properties calculated using BPASS. Class
    structure is used to improve the speed.
    """

    def __init__(self, parallel=None,
                 filename='/home/inikolic/projects/stochasticity/stoc_sampler/BPASS/spectra-bin-imf135_300.a+00.'):
        """
        Input
        ----------
        parallel : boolean,
            Whether first processing is parallelized.
        filename : string,
            Which BPASS file is used.
        """
        self.metal_avail = np.array([1e-5, 1e-4, 1e-3, 0.002, 0.003, 0.004,
                                     0.006, 0.008, 0.01, 0.014, 0.02, 0.03,
                                     0.04])
        self.metal_avail_names = ['zem5', 'zem4', 'z001', 'z002', 'z003',
                                  'z004', 'z006', 'z008', 'z010', 'z014',
                                  'z020', 'z030', 'z040']

        self.SEDS_raw = []

        names = []
        for index, metal_name in enumerate(self.metal_avail_names):
            names.append(filename + metal_name + '.dat')

        if parallel:
            pool = Pool(parallel)
            self.SEDS_raw = pool.map(reader, names)
        else:
            for index, name in enumerate(self.metal_avail_names):
                self.SEDS_raw.append(
                    open(filename + name + '.dat').read().split('\n'))

        self.wv_b = len(self.SEDS_raw[0]) - 1
        self.wv = np.linspace(1, 1e5 + 1, self.wv_b + 1)
        if parallel:
            pool = Pool(parallel)
            self.SEDS = pool.map(splitter, self.SEDS_raw)
            self.SEDS = np.array(self.SEDS)
        else:
            self.SEDS = np.array([[[float(fp[i].split()[j]) for i in
                                    range(self.wv_b)] for j in range(1, 52)] for
                                  fp in self.SEDS_raw])
        self.ages = 52
        self.ag = np.array([0] + [10 ** (6.05 + 0.1 * i) for i in range(1, 52)])

        self.t_star = 0.36

    def get_UV(self, metal, Mstar, SFR, z, SFH_samp=None, sigma_uv=True):
        """
        Function returs the specific luminosity at 1500 angstroms averaged over
        100 angstroms.
        Input
        ----------
            metal : float,
                Metallicity of the galaxy.
            Mstar : float,
                Stellar mass of the galaxy.
            SFR : float,
                Star formation rate of the galaxy.
            z : float,
                redshift of observation.
            SFH_samp : boolean,
                whether SFH is sampled or it's given by previous properties.
                So far sampling does nothing so it's all the same.
        Output
        ----------
            UV_final : float,
                UV luminosity in ergs Hz^-1
        """
        metal = OH_to_mass_fraction(metal)

        # to get solar metalicity need to take 0.42 according to Strom+18

        metal = metal / 10 ** 0.42
        for i, met_cur in enumerate(self.metal_avail):
            if metal < met_cur:
                break
        met_prev = None
        if i != 0:
            met_prev = self.metal_avail[i - 1]
        met_next = self.metal_avail[i]

        # SEDp = self.SEDS[i-1]
        # SEDn = self.SEDS[i]

        try:
            self.SFH = np.zeros(self.ages - 1)
            self.SFH[:len(SFH_samp)] = np.array(SFH_samp)
            self.SFH /= 1e6

        except TypeError:  # not even set-up

            if SFH_samp is None:
                SFH_short, self.index_age = get_SFH_exp(Mstar, SFR, z)
            else:
                if sigma_uv:
                    SFH_short, self.index_age = SFH_samp.get_SFH_const(Mstar, SFR)
                else:
                    SFH_short, self.index_age = SFH_samp.get_SFH_exp(Mstar, SFR)
            self.SFH = np.zeros(self.ages - 1)
            self.SFH[:len(SFH_short)] = np.array(SFH_short)
            self.SFH /= 1e6

        # wv_UV = self.wv[1450:1550]
        # UV_p = np.zeros(self.ages-1)
        # UV_n = np.zeros(self.ages-1)

        UVs_all = np.sum(self.SEDS[0:10, :, 1449:1549],
                         axis=2) / 100 * self.SFH * (
                          self.ag[1:] - self.ag[:-1]) * ang_to_hz
        FUVs = np.sum(UVs_all, axis=1)
        s = splrep(self.metal_avail[:10], FUVs, k=5, s=5)
        UV_final = float(BSpline(*s)(metal))
        return UV_final


def UV_calc_BPASS(
        Muv,
        masses_hmf,
        dndm,
        f_star_norm=1.0,
        alpha_star=0.5,
        sigma_SHMR=0.3,
        sigma_SFMS_norm=0.0,
        t_star=0.5,
        a_sig_SFR=-0.11654893,
        z=11,
        vect_func = None,
        bpass_read = None,
        SFH_samp = None,
        M_knee=2.6e11,
):
    msss = ms_mh_flattening(10 ** masses_hmf, alpha_star_low=alpha_star,
                            fstar_norm=f_star_norm, M_knee=M_knee)
    sfrs = SFMS(msss, SFR_norm=t_star, z=z)

    Zs = metalicity_from_FMR(msss, sfrs)
    Zs += DeltaZ_z(z)
    F_UV = vect_func(Zs, msss, sfrs, z=z, SFH_samp=SFH_samp)
    muvs = Muv_Luv(F_UV * 3.846 * 1e33)
    sfr_obs_log = np.interp(Muv, np.flip(muvs), np.flip(np.log10(sfrs)))
    ms_obs_log = np.interp(sfr_obs_log, np.log10(sfrs), np.log10(msss))
    sigma_SFMS_var = sigma_SFR_variable(msss, norm=sigma_SFMS_norm,
                                        a_sig_SFR=a_sig_SFR)
    uvlf = [uv_calc(
        muvi,
        masses_hmf,
        dndm,
        sigma_SFMS=sigma_SFMS_var,
        sigma_SHMR=sigma_SHMR,
        sfr_obs_log=sfr_obs_log[index],
        ms_obs_log=ms_obs_log[index],
        msss=msss,
        sfrs=sfrs,
        muvs=muvs,
    ) for index, muvi in enumerate(Muv)]

    return uvlf

@njit
def uv_calc_op(
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
    sigma_kuv = 0.1,
):
    N_samples = int(3e4)
    log_mhs_int = np.random.uniform(7.0, 16.0, N_samples)
    log_ms_int = np.interp(log_mhs_int, masses_hmf, np.log10(msss))
    sig_int = np.interp(log_ms_int, np.log10(msss), sigma_SFMS)
    dnd_interp = np.interp(log_mhs_int, masses_hmf, dndm)

    denom = np.sqrt(sig_int**2 + sigma_SHMR**2 + sigma_kuv**2 + 0.054**2)
    prefactor = 1.0 / (denom * np.sqrt(2 * np.pi))
    exp_term = np.exp(-(log_ms_int - ms_obs_log)**2 / (2 * denom**2))

    return np.sum(dnd_interp * prefactor * exp_term) / N_samples * 9


def linear_model_kuv(X, sigma_kuv):
    a,b,c = (0.05041177782984782, -0.029117831879005154, -0.04726733615202826)
    M, z = X
    sigmas = a * (M-9) + b * (z-6) + c * (z-6) * (M-9) + sigma_kuv
    sigmas = np.clip(sigmas, 0.0, 0.5)
    return sigmas

def UV_calc_BPASS_op(
        Muv,
        masses_hmf,
        dndm,
        f_star_norm=1.0,
        alpha_star=0.5,
        sigma_SHMR=0.3,
        sigma_SFMS_norm=0.0,
        t_star=0.5,
        a_sig_SFR=-0.11654893,
        z=11,
        vect_func = None,
        bpass_read = None,
        SFH_samp = None,
        M_knee=2.6e11,
        sigma_kuv = 0.1,
        mass_dependent_sigma_uv=False,
):
    msss = ms_mh_flattening(10 ** masses_hmf, alpha_star_low=alpha_star,
                            fstar_norm=f_star_norm, M_knee=M_knee)
    sfrs = SFMS(msss, SFR_norm=t_star, z=z)

    Zs = metalicity_from_FMR(msss, sfrs)
    Zs += DeltaZ_z(z)
    F_UV = vect_func(Zs, msss, sfrs, z=z, SFH_samp=SFH_samp, sigma_uv = sigma_kuv)
    muvs = Muv_Luv(F_UV * 3.846 * 1e33)
    sfr_obs_log = np.interp(Muv, np.flip(muvs), np.flip(np.log10(sfrs)))
    ms_obs_log = np.interp(sfr_obs_log, np.log10(sfrs), np.log10(msss))

    if mass_dependent_sigma_uv:
        sigma_kuv_var = linear_model_kuv((ms_obs_log, z), sigma_kuv)
    else:
        sigma_kuv_var = sigma_kuv
    sigma_SFMS_var = sigma_SFR_variable(msss, norm=sigma_SFMS_norm,
                                        a_sig_SFR=a_sig_SFR)
    uvlf = [uv_calc_op(
        muvi,
        masses_hmf,
        dndm,
        sigma_SFMS=sigma_SFMS_var,
        sigma_SHMR=sigma_SHMR,
        sfr_obs_log=sfr_obs_log[index],
        ms_obs_log=ms_obs_log[index],
        msss=msss,
        sfrs=sfrs,
        muvs=muvs,
        sigma_kuv=sigma_kuv_var[index],
    ) for index, muvi in enumerate(Muv)]

    return uvlf