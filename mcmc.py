import numpy as np
import halomod as hm
import hmf as hmf
import time
import pymultinest
from astropy.cosmology import Planck18 as cosmo
import json
import ultranest
from scipy.special import erf, erfinv
from uvlf import UV_calc

from ulty import Bias_nonlin, AngularCF_NL, w_IC, My_HOD
from observations import Observations

class LikelihoodAngBase():
    """

    To be filled

    Returns
    -------

    """
    def __init__(self, params):
        fid_params = {
            'p1':lambda x: np.exp(-0.5*(x-9.25)**2/0.5**2),
            'zmin': 8,
            'zmax': 10.5,
            'theta_max': 10 ** -0.8,
            'theta_min': 10 ** -6.3,
            'theta_num': 50,
            'theta_log': True,
            'hod_model': My_HOD,
            'tracer_concentration_model': hm.concentration.Duffy08,
            'tracer_profile_model': hm.profiles.NFW,
            'hmf_model': "Behroozi",
            'bias_model': "Tinker10",
            'sd_bias_model': Bias_nonlin,
            'sd_bias_params': {'z': 9.25},
            'transfer_model': "EH",
            'rnum': 30,
            'rmin': 0.1,
            'rmax': 30,
            'dr_table': 0.1,
            'dlnk': 0.1,
            'dlog10m': 0.05,
        }
        self.params = params

        self.angular_gal = AngularCF_NL(
            **fid_params,
            hod_params={
                'stellar_mass_min': 8.75,
                'stellar_mass_sigma': 0.3,
                'fstar_scale': 10 ** 0.0,
                'alpha_star_low': 0.5,
                'alpha': 1.0,
                'M1': 13.5,
                'fstar_scale_sat': 10 ** 0,
                'stellar_mass_sigma_sat': 0.3,
            }
        )

    def call_likelihood(self, p, obs="Ang_z9_m87", thet = None, w = None, sig_w=None, savedir=False):

        dic_params = {}
        paramida = p
        for index, pary in enumerate(self.params):
            dic_params[pary] = paramida[index]

        if "alpha" in dic_params:
            alpha = dic_params["alpha"]
        else:
            alpha = 1.0

        if "M_0" in dic_params:
            M_0 = dic_params["M_0"]
        else:
            M_0 = 12.0

        if "M_1" in dic_params:
            M_1 = dic_params["M_1"]
        else:
            M_1 = 13.5

        if "fstar_scale" in dic_params:
            fstar_scale = dic_params["fstar_scale"]
        else:
            fstar_scale = 10 ** 0.0

        if "sigma_SHMR" in dic_params:
            sigma_SHMR = dic_params["sigma_SHMR"]
        else:
            sigma_SHMR = 0.3

        if "alpha_star_low" in dic_params:
            alpha_star_low = dic_params["alpha_star_low"]
        else:
            alpha_star_low = 0.5

        if obs=="Ang_z9_m87":
            M_thresh = 8.75
        else:
            M_thresh = 9.0

        self.angular_gal.hod_params = {
            'stellar_mass_min': M_thresh,
            'stellar_mass_sigma': sigma_SHMR,
            'fstar_scale': 10 ** fstar_scale,
            'alpha': alpha,
            'alpha_star_low': alpha_star_low,
            'M1': M_1,
            'M_0': M_0,
        }
        ang_th = self.angular_gal.theta
        ang_ang = self.angular_gal.angular_corr_gal
        w_IC_instance = w_IC(
            ang_th,
            ang_ang,
            41.5 / 60, 46.6 / 60, 940.29997
        )
        like= 0
        for i_theta, ts in enumerate(thet):
            if ts>0.003:

                wi = np.interp(
                    ts,
                    ang_th / 2 / np.pi * 360,
                    ang_ang - w_IC_instance
                )
                # compare model and data with gaussian likelihood:
                like += -0.5 * (((wi - w[i_theta]) / sig_w[
                    i_theta]) ** 2)
        if obs=="Ang_z9_m9" and savedir:
            fname = str(savedir) + 'fs' + str(np.round(fstar_scale,8)) + '_sig' + str(
                np.round(sigma_SHMR,8)) + '_al' + str(np.round(alpha_star_low,8)) + '.txt'
            np.savetxt(fname, ang_ang - w_IC_instance)
        return like


class LikelihoodUVLFBase:
    """
    To be filled
    Returns
    -------

    """

    def __init__(self, params):
        self.hmf_loc = hmf.MassFunction(z=11, Mmin=5,Mmax=15, dlog10m=0.05)
        self.params = params

    def call_likelihood(self, p, muvs_o=None, uvlf_o=None, sig_o=None):
        # dic_params = dict.fromkeys(self.params, p)
        dic_params = {}
        paramida = p
        for index, pary in enumerate(self.params):
            dic_params[pary] = paramida[index]

        if "fstar_scale" in dic_params:
            fstar_scale = dic_params["fstar_scale"]
        else:
            fstar_scale = 0.0

        if "sigma_SHMR" in dic_params:
            sigma_SHMR = dic_params["sigma_SHMR"]
        else:
            sigma_SHMR = 0.3

        if "sigma_SFMS_norm" in dic_params:
            sigma_SFMS_norm = dic_params["sigma_SFMS_norm"]
        else:
            sigma_SFMS_norm = 0.0

        if "t_star" in dic_params:
            t_star = dic_params["t_star"]
        else:
            t_star = 0.5

        if "alpha_star_low" in dic_params:
            alpha_star = dic_params["alpha_star_low"]
        else:
            alpha_star = 0.5

        if "a_sig_SFR" in dic_params:
            a_sig_SFR = dic_params["a_sig_SFR"]
        else:
            a_sig_SFR = -0.11654893

        lnL = 0
        preds = UV_calc(
            muvs_o,
            np.log10(self.hmf_loc.m),
            self.hmf_loc.dndlnm,
            f_star_norm=10 ** fstar_scale,
            alpha_star=alpha_star,
            sigma_SHMR=sigma_SHMR,
            sigma_SFMS_norm=sigma_SFMS_norm,
            t_star=t_star,
            a_sig_SFR=a_sig_SFR,
        )

        for index, muvi in enumerate(muvs_o):
            lnL += -0.5 * (((preds[index] - uvlf_o[index]) / sig_o[
                index]) ** 2)
        return lnL

def run_mcmc(
        likelihoods,
        params,
        mult_params=None,
        priors=None,
        covariance=False,
        diagonal=False,
):

    if priors is None:
        priors = [(-1.0,1.0),(0.0,1.0), (0.05,0.9), (0.01,1.0), (0.01,1.0)]
    #initialize likelihoods
    output_filename = "/home/inikolic/projects/UVLF_FMs/run_speed/runs_260326/ang_only_90_diagcov/"
    #if initialized
    mult_params_fid = {
        "use_MPI": True,
        "outputfiles_basename": output_filename,
        "importance_nested_sampling": False,
        "sampling_efficiency": 0.8,
        "evidence_tolerance": 0.5,
        "multimodal": False,
        "n_iter_before_update": 20,
        'n_live_points': 1000,
    }

    prior_pars = dict(zip(params, priors))
    with open(
            output_filename + 'priors_params.json',
            'w') as f:
        json.dump(prior_pars, f)

    if mult_params is None:
        mult_params = mult_params_fid
    else:
        for key in mult_params_fid:
            if key not in mult_params:
                mult_params[key] = mult_params_fid[key]

    if "Ang_z9_m87" in likelihoods or "Ang_z9_m9" in likelihoods:
        ang = True
        AngBase = LikelihoodAngBase(params)
    else:
        ang = False
    if "UVLF_z11_McLeod23" in likelihoods:
        uvlf = True
        UVLFBase = LikelihoodUVLFBase(params)
    else:
        uvlf = False

    observations_inst = Observations(ang, uvlf)

    def likelihood(p, ndim, nparams, lnew):
        lnL = 0
        p_new = np.zeros((ndim))
        for i in range(ndim):
            p_new[i] = p[i]
        for li in likelihoods:
            if li == "Ang_z9_m87":
                thet, w, wsig = observations_inst.get_obs_z9_m87()
                lnL+=AngBase.call_likelihood(
                    p_new,
                    obs="Ang_z9_m87",
                    thet=thet,
                    w=w,
                    sig_w=wsig
                )
            elif li == "Ang_z9_m9":
                thet, w, wsig = observations_inst.get_obs_z9_m90()
                lnL+=AngBase.call_likelihood(
                    p_new,
                    obs="Ang_z9_m9",
                    thet=thet,
                    w=w,
                    sig_w=wsig,
                    savedir=output_filename,
                )
            elif li == "UVLF_z11_McLeod23":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z11_McLeod23()
                lnL+=UVLFBase.call_likelihood(
                    p_new,
                    muvs_o=muvs_o,
                    uvlf_o=uvlf_o,
                    sig_o=sig_o
                )
            else:
                lnL+=0 #an option for testing
        return lnL

    def phi(x):
        """Integral of the unit-variance gaussian."""
        return 0.5 * (1 + erf(x / np.sqrt(2)))

    def phiinv(x):
        """Inverse of the integral of the unit-variance gaussian."""
        return np.sqrt(2) * erfinv(2 * x - 1)


    def prior(cube, ndim, nparams):
        if covariance:
            cov_mat = 2 * np.loadtxt(
                '/home/inikolic/projects/UVLF_FMs/priors/cov_matr.txt'
            ) #my default is twice the covariance.
            mu = np.loadtxt(
                '/home/inikolic/projects/UVLF_FMs/priors/means.txt'
            )
            if diagonal:
                cov_mat=np.diag(cov_mat.diagonal())
            x = np.zeros(len(mu))  # vector of picked prior values

            gp = cube
            limits = np.copy(priors)
            mu_i = np.copy(mu)
            cov_i = np.copy(np.diag(cov_mat))
            # calculating the inverse of cond. probs
            for i in range(len(mu)):
                if i > 0:
                    mu_i[i] += (cov_mat[:i, i] @ np.linalg.inv(
                        cov_mat[:i, :i])) @ (
                                       x[:i] - mu[:i]
                               )
                    cov_i[i] = cov_i[i] - (
                            cov_mat[:i, i] @ np.linalg.inv(cov_mat[:i, :i])
                    ) @ (cov_mat[i, :i])

                y_min = phi((limits[i][0] - mu_i[i]) / np.sqrt(cov_i[i]))
                y_max = phi((limits[i][1] - mu_i[i]) / np.sqrt(cov_i[i]))
                gp[i] = mu_i[i] + np.sqrt(cov_i[i]) * phiinv(
                    y_min + gp[i] * (y_max - y_min)
                )
                x[i] = gp[i]

            for i, k in enumerate(limits):
                # j = params.index(k)
                # saving p's for prior params
                cube[i] = gp[i]

            return cube

        else:
            # params = []
            # for i in range(ndim):
            #     params.append(
            #         cube[i] * (priors[i][1] - priors[i][0]) + priors[i][0])
            # #cube = np.array(params).copy()
            # return cube
            for i in range(ndim):
                cube[i] = priors[i][0] + cube[i] * (priors[i][1] - priors[i][0])

    result = pymultinest.run(
        LogLikelihood=likelihood,
        Prior=prior,
        n_dims=len(params),
        **mult_params
    )


if __name__ == "__main__":
    #initialize likelihoods
    #likelihoods = ["UVLF_z11_McLeod23"]
    likelihoods = ["Ang_z9_m9"]
    #likelihoods = []
    #likelihoods = ["UVLF_z11_McLeod23"]
    params = ["fstar_scale", "sigma_SHMR", "t_star", "alpha_star_low",
              "sigma_SFMS_norm", "a_sig_SFR"]
    priors = [(-3.0, 1.0), (0.001, 2.0), (0.001, 1.0), (0.0, 2.0), (0.001, 1.2),
              (-1.0, 0.5)]

    #priors = [(-1.0,1.0),(0.01,1.0), (0.0,1.0)]
    #more possibilities: "M_1", "M_0", "alpha" -> relating to satellite params.
    #new possibility: "a_sig_SFR" -> relating to sigma_SFMS scaling with stellar mass.
    #"write a list of all possible parameters"

    run_mcmc(likelihoods, params, priors=priors, covariance=True, diagonal=True)
