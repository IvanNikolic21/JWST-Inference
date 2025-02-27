import numpy as np
import halomod as hm
import hmf as hmf
import matplotlib.pyplot as plt
from hmf import cached_quantity, parameter, get_mdl
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.special import erfc

    
from numba import njit, prange
import ultranest

hmf_loc = hmf.MassFunction(z=9.25)

class Bias_nonlin(hm.bias.ScaleDepBias):
    def __init__(self, xi_dm: np.ndarray, nu: np.ndarray, z: float):
        self.xi_dm = xi_dm
        self.nu = nu
        self.z = z
        #self.hmf_loc = hmf.MassFunction(self.z)
        super().__init__(self.xi_dm)

    def Mcol(self):
        """The nonlinear mass, nu(Mstar) = 1."""

        nu = spline(
            np.sqrt(hmf_loc.nu),
            #self.hmf_loc.delta_c / self.hmf_loc.sigma,
            hmf_loc.m,
            k=5
        )
        return nu(1)

    def Mnl(self):
        nu = spline(
            np.sqrt(hmf_loc.nu),
            #self.hmf_loc.delta_c / self.hmf_loc.sigma,
            hmf_loc.m,
            k=5
        )
        return nu(hmf_loc.delta_c)

    def bias_scale(self):
        K0 = -0.0697
        k1 = 1.1682
        k2 = 4.7577
        k3 = -0.1561
        L0 = 5.1447
        l1 = 1.4023
        l2 = 0.5823
        l3 = -0.1030
        alphaM = np.log10(hmf_loc.delta_c) / np.log10( self.Mnl() / self.Mcol())
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


def ms_mh_flattening(mh, fstar_scale=1):
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
    f_star_mean = fstar_scale * 0.0076 * (2.6e11 / 1e10) ** 0.5
    f_star_mean /= (mh / 2.6e11) ** -0.5 + (mh / 2.6e11) ** 0.61
    return f_star_mean * mh
def ms_mh(ms, fstar_scale=1):
    mhs = np.logspace(5,15,500)
    mss = ms_mh_flattening(mhs, fstar_scale=fstar_scale)
    return 10**np.interp(np.log10(ms), np.log10(mss), np.log10(mhs))

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

    _defaults = {
        'stellar_mass_min':8.75,
        'stellar_mass_sigma':0.05,
        'fstar_scale':1,
        'fstar_scale_sat':1,
        'stellar_mass_sigma_sat':0.3,
        'M1': 11.5,
        'alpha':0.50,
        'M_min':10.0,
        'sig_logm':0.05,
        'M_0':12.0,
        'M_1':13.0,
    }    

    def _central_occupation(self, m):
        """Amplitude of central tracer at mass M."""
        #print(self.params["stellar_mass_sigma"])
        return 0.5 * erfc(
            -(
                np.log10(m)-np.log10(
                    ms_mh(10**self.params["stellar_mass_min"],
                          fstar_scale=self.params["fstar_scale"]
                         )
                )
            )/self.params["stellar_mass_sigma"] / np.sqrt(2)
        )
    
    def _satellite_occupation(self, m):
        """Amplitude of satellite tracer at mass M."""
        ns = np.zeros_like(m)
        ns_0 = np.zeros_like(m)
        
        ns_0[m > 10 ** self.params["M_0"]] = (
            (m[m > 10 ** self.params["M_0"]] - 10 ** self.params["M_0"]) / 10 ** self.params["M_1"]
        ) ** self.params["alpha"]
    
        ns[m > 10 ** self.params["M_0"]] = 0.5 * erfc(
                -(
                    np.log10(m[m > 10 ** self.params["M_0"]])-np.log10(
                        ms_mh(10**self.params["stellar_mass_min"],
                              fstar_scale=self.params["fstar_scale_sat"]
                             )
                    )
                )/self.params["stellar_mass_sigma_sat"] / np.sqrt(2)
            ) * (m[m > 10 ** self.params["M_0"]]/self.params["M1"])**self.params["alpha"]
            
        return ns

#FIDUCIALS
fid_params = {
    #'p1':lambda x: np.exp(-0.5*(x-9.25)**2/0.5**2),
    'zmin':8,
    'zmax':10.5,
    'theta_max':10**-0.8,
    'theta_min':10**-6.3,
    'theta_num':50,
    'theta_log':True,
    'hod_model':My_HOD,
    # 'hod_model': "Zheng05",
    'tracer_concentration_model':hm.concentration.Duffy08,
    'tracer_profile_model':hm.profiles.NFW,
    'hmf_model':"Behroozi",
    'bias_model':"Tinker10",
    'sd_bias_model':Bias_nonlin,
    'sd_bias_params':{'z':9.25},
}

param_names = ['fstar_norm', 'log_scat_SHMR', 'M1']


def my_prior_transform(cube):
    params = cube.copy()
    # transform location parameter: uniform prior
    lo = -1
    hi = 2

    if len(np.shape(cube)) > 1:
        params[:, 0] = cube[:, 0] * (hi - lo) + lo
    else:
        params[0] = cube[0] * (hi - lo) + lo

    lo = 0.01
    hi = 1.5
    if len(np.shape(cube)) > 1:
        params[:, 1] = cube[:, 1] * (hi - lo) + lo
    else:
        params[1] = cube[1] * (hi - lo) + lo

    lo = 11
    hi = 14
    if len(np.shape(cube)) > 1:
        params[:, 2] = cube[:, 2] * (hi - lo) + lo
    else:
        params[2] = cube[2] * (hi - lo) + lo

    return params

dir_dat = "/home/inikolic/projects/UVLF_FMs/data/Paquereau_2025_clustering/GalClustering_COSMOS-Web_Paquereau2025/clustering_measurements/"

cons_ndgal = [
    i.strip().split() for i in open(
        dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_ndgal.dat"
    ).readlines()
]
cons_theta = [
    i.strip().split() for i in open(
        dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_theta.dat"
    ).readlines()
]
cons_wtheta = [
    i.strip().split() for i in open(
        dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_wtheta.dat"
    ).readlines()
]
cons_wsig = [
    i.strip().split() for i in open(
        dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_wsig.dat"
    ).readlines()
]

thethats87_pos = [
    float(i) for i,j in zip(cons_theta[1],cons_wtheta[1]) if float(j)>0
]
wthethats87_pos = [
    float(i) for i,j in zip(cons_wtheta[1],cons_wtheta[1]) if float(j)>0
]
wsig87_pos = [
    float(i) for i,j in zip(cons_wsig[1],cons_wtheta[1]) if float(j)>0
]

thethats90_pos = [
    float(i) for i,j in zip(cons_theta[2],cons_wtheta[2]) if float(j)>0
]
wthethats90_pos = [
    float(i) for i,j in zip(cons_wtheta[2],cons_wtheta[2]) if float(j)>0
]
wsig90_pos = [
    float(i) for i,j in zip(cons_wsig[2],cons_wtheta[2]) if float(j)>0
]

angular_gal = AngularCF_NL(
    **fid_params,
    hod_params={
        'stellar_mass_min': 8.75,
        'stellar_mass_sigma': 0.3,
        'fstar_scale': 10 ** 0.0,
        'alpha': 1.0,
        'M1': 13.5,
        'fstar_scale_sat': 10 ** 0,
        'stellar_mass_sigma_sat': 0.3,
    }
)


def my_likelihood(params):
    if len(np.shape(params)) > 1:
        fs_sc = params[:, 0]
        sig_shmr = params[:, 1]
        m1 = params[:, 2]

    else:
        fs_sc, sig_shmr, m1 = params

    # compute intensity at every x position according to the model
    # print(
    #     sig_shmr,
    #     10**fs_sc,
    #     m1,
    # )

    def angy(fi, si, mi):
        angular_gal.hod_params = {
            'stellar_mass_min': 8.75,
            'stellar_mass_sigma': si,
            'fstar_scale': 10 ** fi,
            'alpha': 1.0,
            'M1': mi,
            'fstar_scale_sat': 10 ** fi,
            'stellar_mass_sigma_sat': si,
        }
        ang_th = angular_gal.theta
        ang_ang = angular_gal.angular_corr_gal
        w_IC_instance = w_IC(
            ang_th,
            ang_ang,
            41.5 / 60, 46.6 / 60, 940.29997
        )
        like = 0

        #    thethats87_pos,
        # wthethats87_pos,
        # wsig87_pos
        for i_theta, ts in enumerate(thethats87_pos):
            wi = np.interp(
                ts,
                ang_th / 2 / np.pi * 360,
                ang_ang - w_IC_instance
            )
            # compare model and data with gaussian likelihood:
            like += -0.5 * (((wi - wthethats87_pos[i_theta]) / wsig87_pos[
                i_theta]) ** 2)

        angular_gal.hod_params = {
            'stellar_mass_min': 9.0,
            'stellar_mass_sigma': si,
            'fstar_scale': 10 ** fi,
            'alpha': 1.0,
            'M1': mi,
            'fstar_scale_sat': 10 ** fi,
            'stellar_mass_sigma_sat': si,
        }
        ang_th = angular_gal.theta
        ang_ang = angular_gal.angular_corr_gal
        w_IC_instance = w_IC(
            ang_th,
            ang_ang,
            41.5 / 60, 46.6 / 60, 940.29997
        )
        for i_theta, ts in enumerate(thethats90_pos):
            wi = np.interp(
                ts,
                ang_th / 2 / np.pi * 360,
                ang_ang - w_IC_instance
            )
            # compare model and data with gaussian likelihood:
            like += -0.5 * (((wi - wthethats90_pos[i_theta]) / wsig90_pos[
                i_theta]) ** 2)
        return like

    vecy = np.vectorize(angy)
    like = vecy(fs_sc, sig_shmr, m1)

    return like

sampler = ultranest.ReactiveNestedSampler(
    param_names, my_likelihood, my_prior_transform, vectorized=True, log_dir='/home/inikolic/projects/UVLF_FMs/run_speed/', max_iters=1000,
)
result = sampler.run(log_interval=3, dlogz=1.0,dKL=1.0, frac_remain=0.2)

sampler.print_results()
