import numpy as np
import halomod as hm
import hmf as hmf
from hmf import cached_quantity, parameter, get_mdl
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.special import erfc
from astropy.cosmology import Planck18 as cosmo

from numba import njit, prange
import ultranest
hmf_loc_9 = hmf.MassFunction(z=9.25,             Mmin=1,
            Mmax=18,
            dlog10m=0.02,)
hmf_loc_7 = hmf.MassFunction(z=7.0,             Mmin=1,
            Mmax=18,
            dlog10m=0.02,)
hmf_loc_5 = hmf.MassFunction(z=5.5,             Mmin=1,
            Mmax=18,
            dlog10m=0.02,)
from uvlf import ms_mh, ms_mh_flattening


class Bias_nonlin(hm.bias.ScaleDepBias):
    def __init__(self, xi_dm: np.ndarray, nu: np.ndarray, z: float):
        self.xi_dm = xi_dm
        self.nu = nu
        self.z = z
        if z == 9.25:
            self.hmf_loc = hmf_loc_9
        elif z == 7.0:
            self.hmf_loc = hmf_loc_7
        elif z == 5.5:
            self.hmf_loc = hmf_loc_5
        else:
            self.hmf_loc = hmf_loc_9
        #self.hmf_loc = hmf.MassFunction(self.z)
        super().__init__(self.xi_dm)

    def Mcol(self):
        """The nonlinear mass, nu(Mstar) = 1."""

        nu = spline(
            np.sqrt(self.hmf_loc.nu),
            #self.hmf_loc.delta_c / self.hmf_loc.sigma,
            self.hmf_loc.m,
            k=5
        )
        return nu(1)

    def Mnl(self):
        nu = spline(
            np.sqrt(self.hmf_loc.nu),
            #self.hmf_loc.delta_c / self.hmf_loc.sigma,
            self.hmf_loc.m,
            k=5
        )
        return nu(self.hmf_loc.delta_c)

    def bias_scale(self):
        K0 = -0.0697
        k1 = 1.1682
        k2 = 4.7577
        k3 = -0.1561
        L0 = 5.1447
        l1 = 1.4023
        l2 = 0.5823
        l3 = -0.1030
        alphaM = np.log10(self.hmf_loc.delta_c) / np.log10( self.Mnl() / self.Mcol())
        #print(alphaM, np.log10( self.Mnl() / self.Mcol()))
        bias = (
            1 + K0 * np.log10(
                1+(self.xi_dm[:,np.newaxis])**k1
            ) * ((np.sqrt(self.nu)[np.newaxis,:]) ** k2) * (1 + k3 / alphaM)
        ) * (
            1 + L0 * np.log10(
                1+(self.xi_dm[:,np.newaxis])**l1
            ) * ((np.sqrt(self.nu)[np.newaxis,:]) ** l2) * (1 + l3 / alphaM)
        )
        # print(bias)
        # print(bias[100,100])
        return bias


class AngularCF_NL(hm.AngularCF):
    
    
    @parameter("model")
    def sd_bias_model(self, val):
        """Model of Scale Dependant Bias."""
        #print("Here we are")
        if val is None:
            return None
        else:
            return get_mdl(val, "ScaleDepBias")

    @parameter("param")
    def sd_bias_params(self, val):
        """Dictionary of parameters for Scale Dependant Bias."""
        return val

    @cached_quantity
    def sd_bias(self):
        """A class containing relevant methods to calculate scale-dependent bias corrections."""
        #print(self.sd_bias_model)
        if self.sd_bias_model is None:
            return None
        else:
            #print(issubclass(self.sd_bias_model,Bias_nonlin))
            if issubclass(self.sd_bias_model,Bias_nonlin):
                return self.sd_bias_model(
                    xi_dm=self.corr_halofit_mm_fnc(self._r_table), nu=self.nu, **self.sd_bias_params
                )
            return self.sd_bias_model(
                self.corr_halofit_mm_fnc(self._r_table), **self.sd_bias_params
            )

    @cached_quantity
    def sd_bias_correction(self):
        """Return the correction for scale dependancy of bias."""
        #print("Starting correction")
        if self.sd_bias is not None:
            return self.sd_bias.bias_scale()
        else:
            return None

    @cached_quantity
    def _tracer_exclusion(self):
        densityfunc = self.dndm[self._tm] * self.total_occupation[self._tm] / self.mean_tracer_den

        if self.sd_bias_model is not None:
            #print(self.sd_bias_model, issubclass(self.sd_bias_model,Bias_nonlin))
            if issubclass(self.sd_bias_model,Bias_nonlin):
                bias = (self.sd_bias_correction * self.halo_bias)[:, self._tm]
                
            else:
                bias = np.outer(self.sd_bias_correction, self.halo_bias)[:, self._tm]
        else:
            bias = self.halo_bias[self._tm]

        return self.exclusion_model(
            m=self.m[self._tm],
            density=densityfunc,
            power_integrand=densityfunc * self.tracer_profile_ukm[:, self._tm],
            bias=bias,
            r=self._r_table,
            halo_density=self.halo_overdensity_mean * self.mean_density0,
            **self.exclusion_params,
        )



@njit(parallel=True)
def w_IC(ang_theta, ang_func,x_deg, y_deg, angular_distance):
    #print(x_deg, y_deg)
    N_samples = int(1e5)
    x1 = np.random.uniform(
        -x_deg*angular_distance * 2*np.pi/360/ 2,
        x_deg*angular_distance * 2*np.pi/360 / 2,
        N_samples,
    )
    x2 = np.random.uniform(
        -x_deg*angular_distance * 2*np.pi/360/ 2,
        x_deg*angular_distance * 2*np.pi/360 / 2,
        N_samples,
    )
    y1 = np.random.uniform(
        -y_deg*angular_distance * 2*np.pi/360 / 2,
        y_deg*angular_distance * 2*np.pi/360 / 2,
        N_samples,
    )
    y2 = np.random.uniform(
        -y_deg*angular_distance * 2*np.pi/360 / 2,
        y_deg*angular_distance * 2*np.pi/360 / 2,
        N_samples,
    )

    integral_sum = 0.0
    for i in prange(N_samples):  # Parallel loop
        theta = np.sqrt(
            (x1[i]-x2[i])**2 + (y1[i] -y2[i])**2) / angular_distance
        integral_sum += np.interp(
            theta, ang_theta, ang_func)

    return integral_sum / N_samples



class My_HOD(hm.hod.Zheng05):
    """
    Five-parameter model of Zheng (2005).

    Parameters
    ----------
    stellar_mass_min : float, default = 8.75
        Minimum mass of halo that supports a central galaxy
    stellar_mass_sigma: float, default = 0.3
    fstar_scale: float, default=1
    others: same as Zheng+05

    References
    ----------
    .. [1] Zheng, Z. et al., "Theoretical Models of the Halo Occupation Distribution:
           Separating Central and Satellite Galaxies ",
           https://ui.adsabs.harvard.edu/abs/2005ApJ...633..791Z.

    """

    truncate_shmr = True  # set to False to disable baryon-fraction ceiling on scatter

    _defaults = {
        'stellar_mass_min':8.75,
        'stellar_mass_sigma':0.05,
        'fstar_norm':1,
        'fstar_norm_sat':1,
        'stellar_mass_sigma_sat':0.3,
        'M1': 11.5,
        'alpha':0.50,
        'M_min':10.0,
        'sig_logm':0.05,
        'M_0':12.0,
        'M_1':13.0,
        'alpha_star_low':0.5,
        'M_knee':2.6e11,
    }

    def _p_above_threshold(self, m, sigma):
        """
        P(M* > M*_min | Mh=m) using stellar-mass-space Gaussian with optional
        truncation at the baryon-fraction ceiling M*_max = (Ob0/Om0)*Mh.

        Both paths work in stellar mass space via the forward SHMR, so the only
        difference between truncate_shmr=True and False is the truncation itself.
        """
        mu = np.log10(ms_mh_flattening(
            m, cosmo,
            fstar_norm=self.params["fstar_norm"],
            alpha_star_low=self.params["alpha_star_low"],
            M_knee=self.params["M_knee"],
        ))
        ms_min = self.params["stellar_mass_min"]
        s2 = sigma * np.sqrt(2)

        if self.truncate_shmr:
            b = np.log10(cosmo.Ob0 / cosmo.Om0) + np.log10(m)  # log10 M*_max
            # P(M*_min < M* <= M*_max) / P(M* <= M*_max) for truncated Gaussian
            cdf_b   = 0.5 * erfc(-(b     - mu) / s2)   # Φ((b - μ)/σ)
            cdf_min = 0.5 * erfc(-(ms_min - mu) / s2)  # Φ((m*_min - μ)/σ)
            return np.maximum(0.0, (cdf_b - cdf_min) / np.maximum(cdf_b, 1e-300))
        else:
            return 0.5 * erfc((ms_min - mu) / s2)

    def _central_occupation(self, m):
        """Amplitude of central tracer at mass M."""
        return self._p_above_threshold(m, self.params["stellar_mass_sigma"])

    def _satellite_occupation(self, m):
        """Amplitude of satellite tracer at mass M."""
        ns = np.zeros_like(m)
        mask = m > 10 ** self.params["M_0"]
        ns[mask] = (
            self._p_above_threshold(m[mask], self.params["stellar_mass_sigma"])
            * ((m[mask] - 10 ** self.params["M_0"]) / 10 ** self.params["M1"]) ** self.params["alpha"]
        )
        return ns

    @property
    def mmin(self):
        """Minimum turnover mass for tracer."""
        if self.truncate_shmr:
            # N_cen is exactly zero below the halo mass where the baryon-fraction
            # ceiling first reaches M*_min: Mh = M*_min / (Ob0/Om0).
            # Subtract 1 dex as a buffer for the numerical integrator.
            return (
                self.params["stellar_mass_min"]
                - np.log10(cosmo.Ob0 / cosmo.Om0)
                - 1.0
            )
        else:
            return (
                np.log10(ms_mh(
                    10 ** self.params["stellar_mass_min"],
                    fstar_norm=self.params["fstar_norm"],
                    alpha_star_low=self.params["alpha_star_low"],
                    M_knee=self.params["M_knee"],
                ))
                - 5 * self.params["stellar_mass_sigma"]
            )

#FIDUCIALS
# fid_params = {
#     #'p1':lambda x: np.exp(-0.5*(x-9.25)**2/0.5**2),
#     'zmin':8,
#     'zmax':10.5,
#     'theta_max':10**-0.8,
#     'theta_min':10**-6.3,
#     'theta_num':50,
#     'theta_log':True,
#     'hod_model':My_HOD,
#     # 'hod_model': "Zheng05",
#     'tracer_concentration_model':hm.concentration.Duffy08,
#     'tracer_profile_model':hm.profiles.NFW,
#     'hmf_model':"Behroozi",
#     'bias_model':"Tinker10",
#     'sd_bias_model':Bias_nonlin,
#     'sd_bias_params':{'z':9.25},
# }

#param_names = ['fstar_norm', 'sigma_SHMR', 'alpha_star', 'sigma_SFMS', 't_star']

#
# def my_prior_transform(cube,ndim, nparams):
#     #params = cube
#     # transform location parameter: uniform prior
#     lo = -1
#     hi = 2
#
#     # if len(np.shape(cube)) > 1:
#     #     cube[:, 0] = cube[:, 0] * (hi - lo) + lo
#     # else:
#     cube[0] = cube[0] * (hi - lo) + lo
#
#     lo = 0.01
#     hi = 1.5
#     # if len(np.shape(cube)) > 1:
#     #     cube[:, 1] = cube[:, 1] * (hi - lo) + lo
#     # else:
#     cube[1] = cube[1] * (hi - lo) + lo
#
#     # lo = 11
#     # hi = 14
#     # # if len(np.shape(cube)) > 1:
#     # #     cube[:, 2] = cube[:, 2] * (hi - lo) + lo
#     # # else:
#     # cube[2] = cube[2] * (hi - lo) + lo
#
#     # alpha_star
#     lo = 0.0
#     hi = 1.0
#     cube[2] = cube[2] * (hi - lo) + lo
#
#     #SFMS sigma
#     lo = 0.01
#     hi = 1.5
#     cube[3] = cube[3] * (hi - lo) + lo
#
#     #t_star
#     lo = 0.01
#     hi = 0.99
#     cube[4] = cube[4] * (hi - lo) + lo
#
#     return cube

# dir_dat = "/home/inikolic/projects/UVLF_FMs/data/Paquereau_2025_clustering/GalClustering_COSMOS-Web_Paquereau2025/clustering_measurements/"
#
# cons_ndgal = [
#     i.strip().split() for i in open(
#         dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_ndgal.dat"
#     ).readlines()
# ]
# cons_theta = [
#     i.strip().split() for i in open(
#         dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_theta.dat"
#     ).readlines()
# ]
# cons_wtheta = [
#     i.strip().split() for i in open(
#         dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_wtheta.dat"
#     ).readlines()
# ]
# cons_wsig = [
#     i.strip().split() for i in open(
#         dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_wsig.dat"
#     ).readlines()
# ]
#
# thethats87_pos = [
#     float(i) for i,j in zip(cons_theta[1],cons_wtheta[1]) if float(j)>0
# ]
# wthethats87_pos = [
#     float(i) for i,j in zip(cons_wtheta[1],cons_wtheta[1]) if float(j)>0
# ]
# wsig87_pos = [
#     float(i) for i,j in zip(cons_wsig[1],cons_wtheta[1]) if float(j)>0
# ]
#
# thethats90_pos = [
#     float(i) for i,j in zip(cons_theta[2],cons_wtheta[2]) if float(j)>0
# ]
# wthethats90_pos = [
#     float(i) for i,j in zip(cons_wtheta[2],cons_wtheta[2]) if float(j)>0
# ]
# wsig90_pos = [
#     float(i) for i,j in zip(cons_wsig[2],cons_wtheta[2]) if float(j)>0
# ]
#
#
# cons_ndgal_z7 = [
#     i.strip().split() for i in open(
#         dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_ndgal.dat"
#     ).readlines()
# ]
# cons_theta_z7 = [
#     i.strip().split() for i in open(
#         dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_theta.dat"
#     ).readlines()
# ]
# cons_wtheta_z7 = [
#     i.strip().split() for i in open(
#         dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_wtheta.dat"
#     ).readlines()
# ]
# cons_wsig_z7 = [
#     i.strip().split() for i in open(
#         dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_wsig.dat"
#     ).readlines()
# ]
#
# thethats90_pos_z7 = [
#     float(i) for i,j in zip(cons_theta[2],cons_wtheta[2]) if float(j)>0
# ]
# wthethats90_pos_z7 = [
#     float(i) for i,j in zip(cons_wtheta[2],cons_wtheta[2]) if float(j)>0
# ]
# wsig90_pos_z7 = [
#     float(i) for i,j in zip(cons_wsig[2],cons_wtheta[2]) if float(j)>0
# ]




# def my_likelihood(cube, ndim, nparams, lnew):
#     params = cube
#     # if len(np.shape(params)) > 1:
#     #     fs_sc = params[:, 0]
#     #     sig_shmr = params[:, 1]
#     #     m1 = params[:, 2]
#     #
#     # else:
#     fs_sc = params[0]
#     sig_shmr = params[1]
#
#     # compute intensity at every x position according to the model
#     # print(
#     #     sig_shmr,
#     #     10**fs_sc,
#     #     m1,
#     # )
#
#     def angy(fi, si):
#         angular_gal.hod_params = {
#             'stellar_mass_min': 8.75,
#             'stellar_mass_sigma': si,
#             'fstar_scale': 10 ** fi,
#             'alpha': 1.0,
#         }
#         ang_th = angular_gal.theta
#         ang_ang = angular_gal.angular_corr_gal
#         w_IC_instance = w_IC(
#             ang_th,
#             ang_ang,
#             41.5 / 60, 46.6 / 60, 940.29997
#         )
#         like = 0
#
#         #    thethats87_pos,
#         # wthethats87_pos,
#         # wsig87_pos
#         for i_theta, ts in enumerate(thethats87_pos):
#             if ts>0.003:
#
#                 wi = np.interp(
#                     ts,
#                     ang_th / 2 / np.pi * 360,
#                     ang_ang - w_IC_instance
#                 )
#                 # compare model and data with gaussian likelihood:
#                 like += -0.5 * (((wi - wthethats87_pos[i_theta]) / wsig87_pos[
#                     i_theta]) ** 2)
#
#         angular_gal.hod_params = {
#             'stellar_mass_min': 9.0,
#             'stellar_mass_sigma': si,
#             'fstar_scale': 10 ** fi,
#             'alpha': 1.0,
#         }
#         ang_th = angular_gal.theta
#         ang_ang = angular_gal.angular_corr_gal
#         w_IC_instance = w_IC(
#             ang_th,
#             ang_ang,
#             41.5 / 60, 46.6 / 60, 940.29997
#         )
#         for i_theta, ts in enumerate(thethats90_pos):
#             if ts>0.003:
#
#                 wi = np.interp(
#                     ts,
#                     ang_th / 2 / np.pi * 360,
#                     ang_ang - w_IC_instance
#                 )
#                 # compare model and data with gaussian likelihood:
#                 like += -0.5 * (((wi - wthethats90_pos[i_theta]) / wsig90_pos[
#                     i_theta]) ** 2)
#         return like
#
#     vecy = np.vectorize(angy)
#     like = vecy(fs_sc, sig_shmr)
#
#     return like


# def like_calc(
#     cube, ndim, nparams, lnew
# ):
#     like_clust = my_likelihood(cube, ndim, nparams, lnew)
#     #fi, asi, s_sfri,s_shmri, tsti
#     like_uv = like_UV(
#         cube[0],
#         cube[2],
#         cube[1],
#         cube[3],
#         cube[4],
#     )
#     return like_clust + like_uv
#
# result = pymultinest.run(
#     LogLikelihood=like_calc,
#     Prior=my_prior_transform,
#     n_dims=5,
#     use_MPI=True,
#     outputfiles_basename="/home/inikolic/projects/UVLF_FMs/run_speed/run_mult_uvlf/",
#     importance_nested_sampling = False,
#     sampling_efficiency= 0.8,
#     evidence_tolerance= 0.5,
#     multimodal= False,
#     n_iter_before_update=20,
#     max_iter=1000,#just to get some results
# )



# sampler = ultranest.ReactiveNestedSampler(
#     param_names, my_likelihood, my_prior_transform, vectorized=True, log_dir='/home/inikolic/projects/UVLF_FMs/run_speed/',
# )
# result = sampler.run(dlogz=10,dKL=0.5, frac_remain=0.5)
#
# sampler.print_results()
