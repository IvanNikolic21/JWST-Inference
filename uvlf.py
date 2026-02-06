import numpy as np
import hmf as hmf
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from numba import njit, prange
from astropy import constants as const
from astropy.cosmology import z_at_value
from multiprocessing import Pool
from scipy.interpolate import splrep, BSpline
import scipy.integrate as intg
from timeit import default_timer as timer
import math
from numpy.polynomial.hermite import hermgauss

#hmf_loc = hmf.MassFunction(z=11)
def ms_mh_flattening(mh, cosmo, fstar_norm = 1.0, alpha_star_low = 0.5, M_knee=2.6e11):
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

    f_star_mean = fstar_norm
    f_star_mean /= (mh / M_knee) ** (-alpha_star_low) + (mh / M_knee) ** 0.61 #knee denominator
    f_star_mean *= (1e10 / M_knee) ** (-alpha_star_low) + (1e10 / M_knee) ** 0.61 #knee numerator
    return np.minimum(f_star_mean, cosmo.Ob0 / cosmo.Om0) * mh

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
    mss = ms_mh_flattening(mhs, cosmo = cosmo, fstar_norm=fstar_norm, alpha_star_low=alpha_star_low, M_knee=M_knee)
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
    msss = ms_mh_flattening(10**masses_hmf, cosmo=cosmo, alpha_star_low=alpha_star,
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

    def get_UV_sfr10(self, metal, Mstar, SFR, z, SFH_samp=None, sfr_10=True):
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
                if sfr_10 is not None:
                    SFH_short, self.index_age = SFH_samp.get_SFH_const(Mstar, SFR)
                    SFH_short[self.ag[:len(SFH_short)] < 1e8] +=  sfr_10
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
    msss = ms_mh_flattening(10 ** masses_hmf, cosmo=cosmo, alpha_star_low=alpha_star,
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
    muv_int = np.interp(log_mhs_int, masses_hmf, muvs)

    denom = np.sqrt(sig_int**2 + sigma_SHMR**2 + sigma_kuv**2 + 0.054**2)
    prefactor = 1.0 / (denom * np.sqrt(2 * np.pi))
    exp_term = np.exp(-(log_ms_int - ms_obs_log)**2 / (2 * denom**2))

    return np.sum(dnd_interp * prefactor * exp_term) / N_samples * 9


def linear_model_kuv(X, sigma_kuv):
    a,b,c = (0.05041177782984782, -0.029117831879005154, -0.04726733615202826)
    M, z = X
    sigmas = a * (M-9) + b * (z-6) - c * (z-6) * (M-9) + sigma_kuv
    sigmas = np.clip(sigmas, 0.0, 0.5)
    return sigmas

def SFMS_new(Mstar, SFR_norm = 1., z=9.25, slope_SFR=1.0):
    """
        the functon returns SFR from Main sequence
    """
    b_SFR = -np.log10(SFR_norm) + np.log10(cosmo.H(z).to(u.yr ** (-1)).value) + 9.5

    return (Mstar/1e9)**(slope_SFR) * 10 ** b_SFR #* SFR_norm

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
        slope_SFR=1.0,
):
    msss = ms_mh_flattening(10 ** masses_hmf,cosmo=cosmo, alpha_star_low=alpha_star,
                            fstar_norm=f_star_norm, M_knee=M_knee)
    if slope_SFR != 1.0:
        sfrs = SFMS_new(msss, SFR_norm = t_star, z=z, slope_SFR = slope_SFR)
    else:
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
        sigma_kuv_var = sigma_kuv * np.ones(np.shape(ms_obs_log))
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

@njit(fastmath=True, parallel=True)
def _outer_loop_parallel(
    muv_grid,
    p_muv_sfr,
    p_sfr_mstar,
    p_mstar_mh,
    dndlnm_on_mh,
):
    out = np.empty_like(muv_grid, dtype=np.float64)

    # p(Muv | M*)
    p_muv_mstar = p_sfr_mstar @ p_muv_sfr  # (Nmstar, Nmuv)

    # p(Muv | Mh)
    p_muv_given_mh = p_mstar_mh @ p_muv_mstar  # (Nmh, Nmuv)

    # numba doesn't do axis sums or broadcasting in parallel, so we have to do this loop by hand
    for i in prange(muv_grid.size):
        out[i] = np.sum(dndlnm_on_mh * p_muv_given_mh[:,i])  # (Nmh,)

    return out

@njit(parallel=True, fastmath=True)
def setup_sample_probabilities(
    muv_grid,              # array of Muv values (e.g. np.linspace(-23, -15, 50))
    sigma_UV,              # scalar (dispersion of Muv|SFR)
    muuv_of_sfr_grid,      # mu_UV(SFR) tabulated on
    sfr_grid,              # SFR grid (log10)
    mstar_grid,            # M* grid (log10) for interpolation of mu_SFR(M*)
    sigma_sfr_grid,        # sigma_SFR(SFR) tabulated on sfr_grid
    mh_grid,               # Mh grid (log10)
    sigma_SHMR,            # scalar (dispersion of log10 M* | Mh)
    dndlnm_grid,           # d n / d ln M on mh_grid
    seed,
    Nsfr,
    Nmstar,
    Nmh,
):
    # --- RNG & samples (fixed across all Muv) ---
    np.random.seed(seed)
    sfr_range = [-5.0, 5.0]
    mstar_range = [2.0, 12.0]
    mh_range = [5.0, 15.0]
    sfr_samples   = np.random.uniform(sfr_range[0], sfr_range[1],  Nsfr).astype(np.float64)   # (Nsfr,)
    mstar_samples = np.random.uniform(mstar_range[0], mstar_range[1], Nmstar).astype(np.float64)  # (Nmstar,)
    mh_samples    = np.random.uniform(mh_range[0], mh_range[1], Nmh).astype(np.float64)     # (Nmh,)

    scale_sfr   = (sfr_range[1] - sfr_range[0]) / float(Nsfr)    # domain length [-5,5]
    scale_mstar = (mstar_range[1] - mstar_range[0]) / float(Nmstar)  # domain length [2,12]
    scale_mh    = (mh_range[1] - mh_range[0]) / float(Nmh)     # domain length [5,15]
    total_scale = scale_sfr * scale_mstar * scale_mh

    # --- Interpolations to sampled points ---
    muuv_of_sfr      = np.interp(sfr_samples, sfr_grid,      muuv_of_sfr_grid).astype(np.float64)      # (Nsfr,)
    sigma_sfr_of_sfr = np.interp(sfr_samples, sfr_grid,      sigma_sfr_grid).astype(np.float64)        # (Nsfr,)
    sfr_target_of_ms = np.interp(mstar_samples, mstar_grid,  sfr_grid).astype(np.float64)  # (Nmstar,)
    dndlnm_on_mh     = np.interp(mh_samples,   mh_grid,      dndlnm_grid).astype(np.float64)           # (Nmh,)
    mstar_tgt_of_mh  = np.interp(mh_samples,   mh_grid,      mstar_grid).astype(np.float64) # (Nmh,)
    sigma_uv_of_sfr  = np.interp(sfr_samples, sfr_grid, sigma_UV).astype(np.float64)  # (Nmstar,)

    inv_sqrt2pi = 1.0 / np.sqrt(2.0*np.pi)

    # --- Conditional Relations ----
    p_muv_sfr = np.empty((Nsfr, muv_grid.size), dtype=np.float64)
    for i in prange(muv_grid.size):
        diff_muv = muv_grid[i] - muuv_of_sfr  # (Nmuv,)
        # Shape: (Nsfr, Nmuv)
        p_muv_sfr[:,i] = (inv_sqrt2pi / sigma_uv_of_sfr) * np.exp(
            -0.5 * ((diff_muv) / sigma_uv_of_sfr)**2
        )

    # Shape: (Nmstar, Nsfr)
    p_sfr_mstar = np.empty((Nmstar, Nsfr), dtype=np.float64)
    for i in prange(Nmstar):
        diff_sfr = sfr_samples - sfr_target_of_ms[i]
        p_sfr_mstar[i,:] = (inv_sqrt2pi / sigma_sfr_of_sfr) * np.exp(
            -0.5 * (diff_sfr / sigma_sfr_of_sfr)**2
        )

    # Shape: (Nmstar, Nmh)
    p_mstar_mh = np.empty((Nmh, Nmstar), dtype=np.float64)
    for i in prange(Nmh):
        diff_star = mstar_samples - mstar_tgt_of_mh[i]
        p_mstar_mh[i,:] = inv_sqrt2pi / sigma_SHMR * np.exp(
            -0.5 * (diff_star / sigma_SHMR)**2
        )

    return p_muv_sfr, p_sfr_mstar, p_mstar_mh, dndlnm_on_mh, total_scale

# ---------- Vectorized UVLF with Numba on the Mh reduction ----------
def uvlf_numba_vectorized(
    muv_grid,              # array of Muv values (e.g. np.linspace(-23, -15, 50))
    sigma_UV,              # scalar (dispersion of Muv|SFR)
    muuv_of_sfr_grid,      # mu_UV(SFR) tabulated on sfr_grid
    sfr_grid,              # SFR grid (log10)
    mstar_grid,            # M* grid (log10) for interpolation of mu_SFR(M*)
    sigma_sfr_grid,        # sigma_SFR(SFR) tabulated on sfr_grid
    mh_grid,               # Mh grid (log10)
    sigma_SHMR,            # scalar (dispersion of log10 M* | Mh)
    dndlnm_grid,           # d n / d ln M on mh_grid
    *,
    Nsfr,
    Nmstar,
    Nmh,
    seed=0
):
    """
    Computes UVLF(Muv) via nested Monte-Carlo with heavy vectorization and a Numba
    kernel for the Mh-reduction.

    Integral structure:
      UVLF(Muv) = ∫ dlnMh [ dndlnm(Mh) * p(Muv | Mh) ]
      p(Muv | Mh) = ∫ dM* N(M*; <M*>(Mh), sigma_SHMR)
                        * ∫ dSFR N(Muv; mu_UV(SFR), sigma_UV)
                                  * N(SFR; <SFR>(M*), sigma_SFR(SFR))

    All quantities are in log10 where appropriate. Monte-Carlo domains:
      SFR ∈ [-5, 5],  M* ∈ [2, 12],  Mh ∈ [5, 15]   (same as your code)
    """

    p_muv_sfr, p_sfr_mstar, p_mstar_mh, dndlnm_on_mh, total_scale = setup_sample_probabilities(
        muv_grid,
        sigma_UV,
        muuv_of_sfr_grid,
        sfr_grid,
        mstar_grid,
        sigma_sfr_grid,
        mh_grid,
        sigma_SHMR,
        dndlnm_grid,
        Nsfr=Nsfr,
        Nmstar=Nmstar,
        Nmh=Nmh,
        seed=seed,
    )

    out = _outer_loop_parallel(
        muv_grid,
        p_muv_sfr,
        p_sfr_mstar,
        p_mstar_mh,
        dndlnm_on_mh,
    ) * total_scale
    return out

INV_SQRT2PI = 0.3989422804014327

# ---------- tiny kernels: cache-friendly row-major writes ----------
@njit(parallel=True, fastmath=True)
def _gauss_muv_sfr(muv_grid, muuv_of_sfr, sigma_uv_of_sfr):
    Nmuv  = muv_grid.size
    Nsfr  = muuv_of_sfr.size
    outT  = np.empty((Nmuv, Nsfr), dtype=np.float64)  # row-major writes
    for k in prange(Nsfr):
        mu   = muuv_of_sfr[k]
        sig  = sigma_uv_of_sfr[k]
        invs = 1.0 / sig
        norm = INV_SQRT2PI * invs
        c    = -0.5 * invs * invs
        for j in range(Nmuv):
            dx = muv_grid[j] - mu
            outT[j, k] = norm * math.exp(c * dx * dx)
    return outT.T  # (Nsfr, Nmuv)

@njit(parallel=True, fastmath=True)
def _gauss_sfr_mstar(sfr_samples, sfr_target_of_ms, sigma_sfr_of_sfr):
    Nsfr      = sfr_samples.size
    Nmstar    = sfr_target_of_ms.size
    out = np.empty((Nmstar, Nsfr), dtype=np.float64)
    for i in prange(Nmstar):
        mu = sfr_target_of_ms[i]
        for j in range(Nsfr):
            sig  = sigma_sfr_of_sfr[j]
            invs = 1.0 / sig
            norm = INV_SQRT2PI * invs
            c    = -0.5 * invs * invs
            dx   = (sfr_samples[j] - mu)
            out[i, j] = norm * math.exp(c * dx * dx)
    return out  # (Nmstar, Nsfr)

@njit(parallel=True, fastmath=True)
def _gauss_mstar_mh(mstar_samples, mstar_tgt_of_mh, sigma_SHMR):
    Nmstar = mstar_samples.size
    Nmh    = mstar_tgt_of_mh.size
    invs = 1.0 / sigma_SHMR
    norm = INV_SQRT2PI * invs
    c    = -0.5 * invs * invs
    out  = np.empty((Nmh, Nmstar), dtype=np.float64)
    for i in prange(Nmh):
        mu = mstar_tgt_of_mh[i]
        for j in range(Nmstar):
            dx = mstar_samples[j] - mu
            out[i, j] = norm * math.exp(c * dx * dx)
    return out  # (Nmh, Nmstar)

# ---------- your setup(), but leaner & faster ----------
def setup_sample_probabilities_fast(
    muv_grid, sigma_UV, muuv_of_sfr_grid, sfr_grid,
    mstar_grid, sigma_sfr_grid, mh_grid, sigma_SHMR,
    dndlnm_grid, *, Nsfr, Nmstar, Nmh, seed=0, use_float32=False
):
    rng = np.random.default_rng(seed)
    sfr_range   = (-5.0, 5.0)
    mstar_range = ( 2.0,12.0)
    mh_range    = ( 5.0,15.0)

    sfr_samples   = rng.uniform(*sfr_range,   Nsfr).astype(np.float64)
    mstar_samples = rng.uniform(*mstar_range, Nmstar).astype(np.float64)
    mh_samples    = rng.uniform(*mh_range,    Nmh).astype(np.float64)

    scale_sfr   = (sfr_range[1]   - sfr_range[0])   / float(Nsfr)
    scale_mstar = (mstar_range[1] - mstar_range[0]) / float(Nmstar)
    scale_mh    = (mh_range[1]    - mh_range[0])    / float(Nmh)
    total_scale = scale_sfr * scale_mstar * scale_mh

    # Interpolations
    muuv_of_sfr      = np.interp(sfr_samples, sfr_grid,      muuv_of_sfr_grid).astype(np.float64)
    sigma_sfr_of_sfr = np.interp(sfr_samples, sfr_grid,      sigma_sfr_grid   ).astype(np.float64)
    sfr_target_of_ms = np.interp(mstar_samples, mstar_grid,  sfr_grid         ).astype(np.float64)
    dndlnm_on_mh     = np.interp(mh_samples,   mh_grid,      dndlnm_grid      ).astype(np.float64)
    mstar_tgt_of_mh  = np.interp(mh_samples,   mh_grid,      mstar_grid       ).astype(np.float64)

    # sigma_UV can be scalar or array on sfr_grid; handle both:
    if np.ndim(sigma_UV) == 0:
        sigma_uv_of_sfr = np.full_like(sfr_samples, float(sigma_UV), dtype=np.float64)
    else:
        sigma_uv_of_sfr = np.interp(sfr_samples, sfr_grid, sigma_UV).astype(np.float64)

    # Build Gaussian tables with Numba
    p_muv_sfr   = _gauss_muv_sfr(muv_grid, muuv_of_sfr, sigma_uv_of_sfr)      # (Nsfr,   Nmuv)
    p_sfr_mstar = _gauss_sfr_mstar(sfr_samples, sfr_target_of_ms, sigma_sfr_of_sfr)  # (Nmstar, Nsfr)
    p_mstar_mh  = _gauss_mstar_mh(mstar_samples, mstar_tgt_of_mh, sigma_SHMR) # (Nmh,    Nmstar)

    if use_float32:
        p_muv_sfr   = np.ascontiguousarray(p_muv_sfr,   dtype=np.float32)
        p_sfr_mstar = np.ascontiguousarray(p_sfr_mstar, dtype=np.float32)
        p_mstar_mh  = np.ascontiguousarray(p_mstar_mh,  dtype=np.float32)
        dndlnm_on_mh= np.ascontiguousarray(dndlnm_on_mh,dtype=np.float32)

    return p_muv_sfr, p_sfr_mstar, p_mstar_mh, dndlnm_on_mh, total_scale

def uvlf_fast_einsum(
    muv_grid, sigma_UV, muuv_of_sfr_grid, sfr_grid,
    mstar_grid, sigma_sfr_grid, mh_grid, sigma_SHMR, dndlnm_grid,
    *, Nsfr, Nmstar, Nmh, seed=0, use_float32=False
):
    p_muv_sfr, p_sfr_mstar, p_mstar_mh, dndlnm_on_mh, total_scale = setup_sample_probabilities_fast(
        muv_grid, sigma_UV, muuv_of_sfr_grid, sfr_grid,
        mstar_grid, sigma_sfr_grid, mh_grid, sigma_SHMR, dndlnm_grid,
        Nsfr=Nsfr, Nmstar=Nmstar, Nmh=Nmh, seed=seed, use_float32=use_float32
    )

    # Contract: dnd * B(mh,m*) * H(m*,sfr) * G(sfr,muv)  -> (Nmuv,)
    out = np.einsum('m,ma,ar,ru->u',
                    dndlnm_on_mh,
                    p_mstar_mh,
                    p_sfr_mstar,
                    p_muv_sfr,
                    optimize='greedy')
    # Note: by index naming, result shape is 's' == Nmuv.
    return (out * total_scale).astype(np.float64)


def UV_calc_numba(
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
        seed=0,
        **kw,
):
    msss = ms_mh_flattening(10 ** masses_hmf, cosmo, alpha_star_low=alpha_star,
                            fstar_norm=f_star_norm, M_knee=M_knee)
    sfrs = SFMS(msss, SFR_norm=t_star, z=z)

    Zs = metalicity_from_FMR(msss, sfrs)
    Zs += DeltaZ_z(z)
    F_UV = vect_func(Zs, msss, sfrs, z=z, SFH_samp=SFH_samp, sigma_uv = sigma_kuv)
    muvs = Muv_Luv(F_UV * 3.846 * 1e33)
    sfr_obs_log = np.interp(Muv, np.flip(muvs), np.flip(np.log10(sfrs)))
    ms_obs_log = np.interp(sfr_obs_log, np.log10(sfrs), np.log10(msss))

    if mass_dependent_sigma_uv:
        sigma_kuv_var = linear_model_kuv((msss, z), sigma_kuv)
    else:
        sigma_kuv_var = sigma_kuv * np.ones(np.shape(msss))
    sigma_SFMS_var = sigma_SFR_variable(msss, norm=sigma_SFMS_norm,
                                        a_sig_SFR=a_sig_SFR)
    uvlf = uvlf_fast_einsum(
        Muv,
        sigma_kuv_var,              # scalar (dispersion of Muv|SFR)
        muvs,      # mu_UV(SFR) tabulated on sfr_grid
        np.log10(sfrs),              # SFR grid (log10)
        np.log10(msss),            # M* grid (log10) for interpolation of mu_SFR(M*)
        sigma_SFMS_var,        # sigma_SFR(SFR) tabulated on sfr_grid
        masses_hmf,               # Mh grid (log10)
        sigma_SHMR,            # scalar (dispersion of log10 M* | Mh)
        dndm,           # d n / d ln M on mh_grid
        Nsfr=10_000,
        Nmstar=10_000,
        Nmh=10_000,
        seed=seed,
    )

    return uvlf

def gimme_dust(Muv):
    beta = -0.17 * Muv - 5.40
    Auv = 4.43 + 1.99 * beta
    return np.clip(Auv,0,5)

def apply_dust_to_uvlf(Mint, phi_int, gimme_dust, Mobs_grid=None):
    """
    Mint: 1D array of intrinsic magnitudes (ascending or descending is fine)
    phi_int: 1D array of intrinsic UVLF values (same length as Mint), in e.g. dex^-1 Mpc^-3 or mag^-1 Mpc^-3
    gimme_dust: function A(M) returning attenuation in mag (>=0)
    Mobs_grid: optional grid of observed magnitudes to interpolate onto (if None, use the mapped Mint grid)

    Returns
    -------
    Mobs: array of observed magnitudes
    phi_obs: array of dust-attenuated UVLF on Mobs grid
    """
    Mint = np.asarray(Mint)
    phi_int = np.asarray(phi_int)

    # 1) Map intrinsic magnitudes to observed ones
    A = gimme_dust(Mint)
    Mobs = Mint + A

    # Ensure monotonicity for interpolation/grid work (your law is monotonic, but numerics benefit from sorting)
    order = np.argsort(Mobs)
    Mobs_sorted = Mobs[order]
    Mint_sorted = Mint[order]
    phi_int_sorted = phi_int[order]

    # 2) Jacobian dMobs/dMint (numerical to handle clipping pieces)
    dMobs_dMint = np.gradient(Mobs_sorted, Mint_sorted)

    # Guard against any tiny/negative derivatives (can happen at clip boundaries); floor them
    dMobs_dMint = np.clip(dMobs_dMint, 1e-3, None)

    # 3) Number conservation: phi_obs(Mobs(Mint)) = phi_int(Mint) / (dMobs/dMint)
    phi_obs_at_Mint = phi_int_sorted / dMobs_dMint

    # 4) Put result on a clean Mobs grid
    if Mobs_grid is None:
        # return on the native mapped grid
        return Mobs_sorted, phi_obs_at_Mint
    else:
        # Interpolate φ to the requested observed-magnitude grid
        phi_obs_grid = np.interp(Mobs_grid, Mobs_sorted, phi_obs_at_Mint, left=0.0, right=0.0)
        return Mobs_grid, phi_obs_grid


INV_SQRT2PI = 0.3989422804014327
@njit(fastmath=True)
def _find_interval(xgrid, x):
    # returns i such that xgrid[i] <= x < xgrid[i+1], clamped
    n = xgrid.size
    if x <= xgrid[0]:
        return 0
    if x >= xgrid[n-2]:
        return n-2
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xgrid[mid] <= x:
            lo = mid
        else:
            hi = mid
    if lo > n-2:
        lo = n-2
    return lo
@njit(fastmath=True)
def bilinear_interp_regular(xgrid, ygrid, F, x, y):
    """
    xgrid: (Nx,), ygrid: (Ny,), F: (Nx, Ny)
    returns F(x,y) with bilinear interpolation (clamped).
    """
    i = _find_interval(xgrid, x)
    j = _find_interval(ygrid, y)
    x0 = xgrid[i]; x1 = xgrid[i+1]
    y0 = ygrid[j]; y1 = ygrid[j+1]
    # avoid divide by zero
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    f00 = F[i,   j  ]
    f10 = F[i+1, j  ]
    f01 = F[i,   j+1]
    f11 = F[i+1, j+1]
    f0 = f00 + tx * (f10 - f00)
    f1 = f01 + tx * (f11 - f01)
    return f0 + ty * (f1 - f0)

@njit(parallel=True, fastmath=True)
def build_p_muv_sfr_with_x(
    muv_grid,               # (Nmuv,)
    sfr_samples,            # (Nsfr,) samples (log10 SFR)
    mu_x_of_sfr,            # (Nsfr,) mean of x|SFR (dex)
    sigma_x_of_sfr,         # (Nsfr,) sigma of x|SFR (dex)
    sigma_muv_const,        # scalar, e.g. 0.1
    sfr_map_grid,           # (Nsfr_map,) grid for the mapping table (log10 SFR)
    x_map_grid,             # (Nx_map,) grid for the mapping table (dex)
    muuv_map,               # (Nsfr_map, Nx_map) -> mean Muv from mapping
    gh_t, gh_w              # (Nq,) nodes+weights for N(0,1) expectation
):
    """
    Returns p_muv_sfr: (Nsfr, Nmuv)
    p(Muv | sfr_sample) = E_{x~N(mu_x, sigma_x)} [ N(Muv; muuv_map(sfr,x), sigma_muv_const) ]
    where expectation is computed by Gauss-Hermite: x = mu_x + sqrt(2)*sigma_x*t_q
    """
    Nsfr = sfr_samples.size
    Nmuv = muv_grid.size
    Nq   = gh_t.size

    out = np.empty((Nsfr, Nmuv), dtype=np.float64)

    inv_sig_muv = 1.0 / sigma_muv_const
    norm_muv    = INV_SQRT2PI * inv_sig_muv
    c_muv       = -0.5 * inv_sig_muv * inv_sig_muv

    for i in prange(Nsfr):
        sfr = sfr_samples[i]
        mux = mu_x_of_sfr[i]
        sigx = sigma_x_of_sfr[i]

        # accumulate mixture over quadrature nodes
        # init row
        for j in range(Nmuv):
            out[i, j] = 0.0

        # if sigx is zero, just one evaluation
        if sigx <= 0.0:
            mu = bilinear_interp_regular(sfr_map_grid, x_map_grid, muuv_map, sfr, mux)
            for j in range(Nmuv):
                dx = muv_grid[j] - mu
                out[i, j] += norm_muv * math.exp(c_muv * dx * dx)
        else:
            s = math.sqrt(2.0) * sigx
            for q in range(Nq):
                xq = mux + s * gh_t[q]
                wq = gh_w[q]
                mu = bilinear_interp_regular(sfr_map_grid, x_map_grid, muuv_map, sfr, xq)
                for j in range(Nmuv):
                    dx = muv_grid[j] - mu
                    out[i, j] += wq * (norm_muv * math.exp(c_muv * dx * dx))

    return out

def setup_sample_probabilities_fast_with_sfr10(
    muv_grid,
    sfr_grid, mstar_grid, mh_grid,
    sigma_sfr_grid,
    sigma_SHMR,
    dndlnm_grid,
    # NEW: mapping inputs for μ_uv(SFR,x)
    sfr_map_grid, x_map_grid, muuv_map,
    # NEW: x|SFR model
    sigma_x_of_sfr_fn,       # callable: sigma_x_of_sfr_fn(sfr_samples)->array
    mu_x_of_sfr_fn=None,     # callable; if None, use mu_x=sfr (centered)
    sigma_muv_const=0.1,
    *,
    Nsfr, Nmstar, Nmh, seed=0,
    gh_n=24,
):
    rng = np.random.default_rng(seed)
    sfr_range   = (-5.0, 5.0)
    mstar_range = ( 2.0,12.0)
    mh_range    = ( 5.0,15.0)

    sfr_samples   = rng.uniform(*sfr_range,   Nsfr).astype(np.float64)
    mstar_samples = rng.uniform(*mstar_range, Nmstar).astype(np.float64)
    mh_samples    = rng.uniform(*mh_range,    Nmh).astype(np.float64)

    scale_sfr   = (sfr_range[1]   - sfr_range[0])   / float(Nsfr)
    scale_mstar = (mstar_range[1] - mstar_range[0]) / float(Nmstar)
    scale_mh    = (mh_range[1]    - mh_range[0])    / float(Nmh)
    total_scale = scale_sfr * scale_mstar * scale_mh

    # Interpolations needed for the other factors
    sigma_sfr_of_sfr = np.interp(sfr_samples, sfr_grid, sigma_sfr_grid).astype(np.float64)
    sfr_target_of_ms = np.interp(mstar_samples, mstar_grid, sfr_grid).astype(np.float64)
    dndlnm_on_mh     = np.interp(mh_samples, mh_grid, dndlnm_grid).astype(np.float64)
    mstar_tgt_of_mh  = np.interp(mh_samples, mh_grid, mstar_grid).astype(np.float64)

    # Build p_sfr_mstar and p_mstar_mh the same way you already do
    p_sfr_mstar = _gauss_sfr_mstar(sfr_samples, sfr_target_of_ms, sigma_sfr_of_sfr)  # (Nmstar, Nsfr)
    p_mstar_mh  = _gauss_mstar_mh(mstar_samples, mstar_tgt_of_mh, sigma_SHMR)        # (Nmh, Nmstar)

    # --- NEW: build p_muv_sfr by integrating over x ---
    gh_t, gh_w = hermgauss(gh_n)
    gh_t = gh_t.astype(np.float64)
    gh_w = (gh_w / np.sqrt(np.pi)).astype(np.float64)

    if mu_x_of_sfr_fn is None:
        mu_x = sfr_samples.copy()
    else:
        mu_x = np.asarray(mu_x_of_sfr_fn(sfr_samples), dtype=np.float64)
    sig_x = np.asarray(sigma_x_of_sfr_fn(sfr_samples), dtype=np.float64)
    # :white_check_mark: FIX: accept scalar-returning functions
    if mu_x.ndim == 0:
        mu_x = np.full_like(sfr_samples, float(mu_x), dtype=np.float64)
    if sig_x.ndim == 0:
        sig_x = np.full_like(sfr_samples, float(sig_x), dtype=np.float64)

    p_muv_sfr = build_p_muv_sfr_with_x(
        np.asarray(muv_grid, dtype=np.float64),
        sfr_samples,
        mu_x,
        sig_x,
        float(sigma_muv_const),
        np.asarray(sfr_map_grid, dtype=np.float64),
        np.asarray(x_map_grid, dtype=np.float64),
        np.asarray(muuv_map, dtype=np.float64),
        gh_t, gh_w
    )  # (Nsfr, Nmuv)

    return p_muv_sfr, p_sfr_mstar, p_mstar_mh, dndlnm_on_mh, total_scale

def uvlf_fast_einsum_sfr10version(
    muv_grid,
    sfr_grid, mstar_grid, mh_grid,
    sigma_sfr_grid, sigma_SHMR, dndlnm_grid,
    sfr_map_grid, x_map_grid, muuv_map,
    sigma_x_of_sfr_fn,
    mu_x_of_sfr_fn=None,
    sigma_muv_const=0.1,
    *,
    Nsfr, Nmstar, Nmh, seed=0, gh_n=24
):
    p_muv_sfr, p_sfr_mstar, p_mstar_mh, dndlnm_on_mh, total_scale = (
        setup_sample_probabilities_fast_with_sfr10(
            muv_grid,
            sfr_grid, mstar_grid, mh_grid,
            sigma_sfr_grid,
            sigma_SHMR,
            dndlnm_grid,
            sfr_map_grid, x_map_grid, muuv_map,
            sigma_x_of_sfr_fn,
            mu_x_of_sfr_fn=mu_x_of_sfr_fn,
            sigma_muv_const=sigma_muv_const,
            Nsfr=Nsfr, Nmstar=Nmstar, Nmh=Nmh, seed=seed, gh_n=gh_n
        )
    )
    out = np.einsum('m,ma,ar,ru->u',
                    dndlnm_on_mh,
                    p_mstar_mh,
                    p_sfr_mstar,
                    p_muv_sfr,
                    optimize='greedy')
    return out * total_scale

def linear_model_sfr10(X, sigma_sfr_10_norm):
    a,b,c = (0.05041177782984782, -0.029117831879005154, -0.04726733615202826)
    M, z = X
    sigmas = a * (M-9) + b * (z-6) - c * (z-6) * (M-9) + sigma_sfr_10_norm
    sigmas = np.clip(sigmas, 0.0, 2 * sigma_sfr_10_norm)
    return sigmas


def UV_calc_numba_sfr10(
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
        vect_func=None,
        SFH_samp=None,
        M_knee=2.6e11,
        sigma_sfr10=0.1,
        mass_dependent_sfr10=False,
        seed=0,
        **kw,
):
    msss = ms_mh_flattening(10 ** masses_hmf, cosmo, alpha_star_low=alpha_star,
                            fstar_norm=f_star_norm, M_knee=M_knee)
    sfrs = SFMS(msss, SFR_norm=t_star, z=z)

    Zs = metalicity_from_FMR(msss, sfrs)
    Zs += DeltaZ_z(z)
    sfr_map_grid = np.log10(sfrs)  # (Nsfr_map,)
    x_map_grid = np.linspace(-3, 3, 1000)  # x = log10(SFR10/SFR)  (nice, centered)
    # Build SFR10 grid per SFR:
    sfr10_grid = (10 ** sfr_map_grid)[:, None] * (10 ** x_map_grid)[None, :]  # (Nsfr_map, Nx_map)
    # Evaluate mapping: muuv_map[sfr_i, x_j] = MUV_mean(SFR=sfr_i, SFR10=sfr10_ij)
    # You can compute luminosity then convert to Muv:
    F_UVs = np.array(
        [
            vect_func(Zs, msss, 10 ** sfr_map_grid, z=z, SFH_samp=SFH_samp, sfr_10=sfr10_grid[:, i]) for i in
            range(1000)
        ]
    ).T
    muuv_map = Muv_Luv(F_UVs * 3.846e33)  # (Nsfr_map, Nx_map)

    if mass_dependent_sfr10:
        sigma_sfr10_var = linear_model_sfr10((msss, z), mass_dependent_sfr10)
    else:
        sigma_sfr10_var = mass_dependent_sfr10 * np.ones(np.shape(msss))
    sigma_SFMS_var = sigma_SFR_variable(msss, norm=sigma_SFMS_norm,
                                        a_sig_SFR=a_sig_SFR)
    uvlf = uvlf_fast_einsum_sfr10version(
        Muv,
        np.log10(sfrs),  # SFR grid (log10)
        np.log10(msss),  # M* grid (log10) for interpolation of mu_SFR(M*)
        masses_hmf,  # Mh grid (log10)
        sigma_SFMS_var,  # sigma_SFR(SFR) tabulated on sfr_grid
        sigma_SHMR,  # scalar (dispersion of log10 M* | Mh)
        dndm,  # d n / d ln M on mh_grid
        sfr_map_grid,
        x_map_grid,
        muuv_map,
        sigma_x_of_sfr_fn=lambda x: sigma_sfr10,
        Nsfr=10_000,
        Nmstar=10_000,
        Nmh=10_000,
        seed=seed,
    )
    return uvlf