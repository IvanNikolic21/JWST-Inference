import numpy as np
import halomod as hm
import hmf as hmf
import matplotlib.pyplot as plt
from hmf import cached_quantity, parameter, get_mdl
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from numba import njit, prange


class Bias_nonlin(hm.bias.ScaleDepBias):
    def __init__(self, xi_dm: np.ndarray, nu: np.ndarray, z: float):
        self.xi_dm = xi_dm
        self.nu = nu
        self.z = z
        self.hmf_loc = hmf.MassFunction(self.z)
        super().__init__(self.xi_dm)

    def Mcol(self):
        """The nonlinear mass, nu(Mstar) = 1."""

        nu = spline(
            np.sqrt(self.hmf_loc.nu),
            self.hmf_loc.m,
            k=5
        )
        return nu(1)

    def Mnl(self):
        """The collapsed mass, nu(Mstar) = delta_c."""

        nu = spline(
            np.sqrt(self.hmf_loc.nu),
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

        bias = (
            1 + K0 * np.log10(
                1+(self.xi_dm[:,np.newaxis])**k1
            ) * ((np.sqrt(self.nu)[np.newaxis,:]) ** k2) * (1 + k3 / alphaM)
        ) * (
            1 + L0 * np.log10(
                1+(self.xi_dm[:,np.newaxis])**l1
            ) * ((np.sqrt(self.nu)[np.newaxis,:]) ** l2) * (1 + l3 / alphaM)
        )

        return bias

class AngularCF_NL(hm.AngularCF):
    
    
    @parameter("model")
    def sd_bias_model(self, val):
        """Model of Scale Dependant Bias."""
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
        if self.sd_bias_model is None:
            return None
        else:
            if issubclass(self.sd_bias_model,Bias_nonlin):
                return self.sd_bias_model(
                    xi_dm=self.corr_halofit_mm_fnc(self._r_table), nu=self.nu, z=self.z, **self.sd_bias_params
                )
            return self.sd_bias_model(
                self.corr_halofit_mm_fnc(self._r_table), **self.sd_bias_params
            )

    @cached_quantity
    def sd_bias_correction(self):
        """Return the correction for scale dependancy of bias."""

        if self.sd_bias is not None:
            return self.sd_bias.bias_scale()
        else:
            return None

    @cached_quantity
    def _tracer_exclusion(self):
        densityfunc = self.dndm[self._tm] * self.total_occupation[self._tm] / self.mean_tracer_den

        if self.sd_bias_model is not None:

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

    N_samples = int(1e8)
    x1 = np.random.uniform(
        -x_deg*angular_distance * 2*np.pi/360/ 2,
        x_deg*angular_distance * 2*np.pi/360 / 2,
        N_samples,
    )
    x2 = np.random.uniform(
        -x_deg*angular_distance * 2*np.pi/360 / 2,
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
            x1[i]**2 + x2[i]**2 + y1[i]**2 + y2[i]**2) / angular_distance

        integral_sum += np.interp(theta, ang_theta, ang_func)

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
        print(self.params["stellar_mass_sigma"])
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

