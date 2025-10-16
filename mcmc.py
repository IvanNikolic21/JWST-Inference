import numpy as np
import halomod as hm
import hmf as hmf
import time
import pymultinest
from astropy.cosmology import Planck18 as cosmo
import json
import ultranest

import scipy.integrate as intg

from scipy.special import erf, erfinv
from uvlf import UV_calc
import csv
import os
from ulty import Bias_nonlin, AngularCF_NL, w_IC, My_HOD
from observations import Observations
from uvlf import bpass_loader, UV_calc_BPASS, SFH_sampler, get_SFH_exp, UV_calc_BPASS_op
from uvlf import uvlf_numba_vectorized, UV_calc_numba
import argparse

class LikelihoodAngBase():
    """

    To be filled

    Returns
    -------

    """
    def __init__(self, params, realistic_Nz=False, hmf_choice="Tinker08", z=9.25, exact_specs=True):
        self.exact_specs = exact_specs
        if realistic_Nz:
            print("this is redshift in Angular likelihood base", z)
            with open(
                    '/home/inikolic/projects/UVLF_FMs/github_code/JWST-Inference/Nz_8_105_alt.csv',
                    newline='') as csvfile:
                Nz_8_10 = list(
                    csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                )
            with open(
                '/home/inikolic/projects/UVLF_FMs/github_code/JWST-Inference/Nz_6_8_alt.csv',
                newline='') as csvfile:
                Nz_6_8 = list(
                    csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                )
            with open(
                '/home/inikolic/projects/UVLF_FMs/github_code/JWST-Inference/Nz_5_6_alt.csv',
                newline='') as csvfile:
                Nz_5_6 = list(
                    csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                )

            z_8 = np.array(Nz_8_10)[:,0]#.argsort()
            z_6 = np.array(Nz_6_8)[:,0]#.argsort()
            z_5 = np.array(Nz_5_6)[:,0]#.argsort()

            z_8_ind = z_8.argsort()
            z_6_ind = z_6.argsort()
            z_5_ind = z_5.argsort()

            z_8_s = z_8
            Nz_8_s = np.array(Nz_8_10)[:,1]
            z_6_s = z_6
            Nz_6_s = np.array(Nz_6_8)[:,1]#[z_6_ind[::-1]]
            z_5_s = z_5
            Nz_5_s = np.array(Nz_5_6)[:,1]#[z_5_ind[::-1]]

            p1 = lambda x: np.interp(x,z_8_s, Nz_8_s, left=0, right=0 )
            p1_z7 = lambda x: np.interp(x,z_6_s, Nz_6_s , left=0, right=0)
            p1_z5_5 = lambda x:np.interp(x, z_5_s, Nz_5_s, left=0, right=0)
            #p1_z5_5 = lambda x: np.exp(-0.5 * (x-5.5)**2/0.25**2)

            self.p1 = p1
            self.p1_z7 = p1_z7
            self.p1_z5_5 = p1_z5_5
        else:
            p1 = lambda x: np.exp(-0.5 * (x - 9.25) ** 2 / 0.5 ** 2)
            p1_z7 = lambda x: np.exp(-0.5 * (x - 7.0) ** 2 / 0.5 ** 2)
            p1_z5_5 = lambda x: np.exp(-0.5 * (x-5.5)**2/0.25**2)
            self.p1 = p1
            self.p1_z7 = p1_z7
            self.p1_z5_5 = p1_z5_5

        fid_params = {
            'p1':p1,
            'zmin': 0,
            'zmax': 15,
            'theta_max': 10 ** -0.8,
            'theta_min': 10 ** -6.3,
            'theta_num': 50,
            'theta_log': True,
            'hod_model': My_HOD,
            'tracer_concentration_model': hm.concentration.Duffy08,
            'tracer_profile_model': hm.profiles.NFW,
            'hmf_model': hmf_choice,
            'bias_model': "Tinker10",
            'transfer_model': "EH",
            'exclusion_model': "Sphere",
            'rnum': 50,
            'rmin': 0.05,
            'rmax': 50,
            'dr_table': 0.05,
            'dlnk': 0.05,
            'dlog10m': 0.02,
            'z':z,
        }
        if not exact_specs:
            fid_params['sd_bias_model'] =  Bias_nonlin
            fid_params['sd_bias_params'] = {'z': z}
        self.params = params
        self.angular_gal = AngularCF_NL(
            **fid_params,
            hod_params={
                'stellar_mass_min': 8.75,
                'stellar_mass_sigma': 0.3,
                'fstar_norm': 10 ** 0.0,
                'alpha_star_low': 0.5,
                'alpha': 1.0,
                'M1': 12.0,
                'fstar_norm_sat': 10 ** 0,
                'stellar_mass_sigma_sat': 0.3,
            }
        )

    def call_likelihood(self, p, obs="Ang_z9_m87", thet = None, w = None, sig_w=None, savedir=False, no_call=False):

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
            M_0 = 11.65

        if "M_1" in dic_params:
            M_1 = dic_params["M_1"]
        else:
            M_1 = 12.3

        if "fstar_norm" in dic_params:
            fstar_norm = dic_params["fstar_norm"]
        else:
            fstar_norm = 10 ** 0.0

        if "sigma_SHMR" in dic_params:
            sigma_SHMR = dic_params["sigma_SHMR"]
        else:
            sigma_SHMR = 0.3

        if "alpha_star_low" in dic_params:
            alpha_star_low = dic_params["alpha_star_low"]
        else:
            alpha_star_low = 0.5

        if "M_knee" in dic_params:
            M_knee = 10**dic_params["M_knee"]
        else:
            M_knee = 2.6e11

        if obs == "Ang_z9_m87":
            M_thresh = 8.75
        elif obs == "Ang_z7_m87":
            M_thresh = 8.7
        elif obs in ["Ang_z9_m9", "Ang_z7_m9", "Ang_z5_5_m9"]:
            M_thresh = 9.0
        elif obs == "Ang_z5_5_m9_25":
            M_thresh = 9.25
        elif obs == "Ang_z5_5_m9_5":
            M_thresh = 9.5
        elif obs == "Ang_z5_5_m8_5":
            M_thresh = 8.5
        else:
            M_thresh = 9.3

        if obs=="Ang_z7_m9" or obs=="Ang_z7_m93":
            p1_chosen = self.p1_z7
        elif obs=="Ang_z5_5_m9" or obs=="Ang_z5_5_m9_25" or obs=="Ang_z5_5_m9_5":
            p1_chosen = self.p1_z5_5
        else:
            p1_chosen = self.p1
        self.angular_gal.hod_params = {
            'stellar_mass_min': M_thresh,
            'stellar_mass_sigma': sigma_SHMR,
            'fstar_norm': 10 ** fstar_norm,
            'alpha': alpha,
            'alpha_star_low': alpha_star_low,
            'M1': M_1,
            'M_0': M_0,
            'M_knee': M_knee,
        }
        self.angular_gal.update(p1=p1_chosen)
        ang_th = self.angular_gal.theta
        ang_ang = self.angular_gal.angular_corr_gal
        # if self.exact_specs:
        #     w_IC_instance = w_IC(
        #         ang_th,
        #         ang_ang,
        #         18.5 / 60, 16.6 / 60, 940.29997
        #     )
        # else:
        w_IC_instance = w_IC(
            ang_th,
            ang_ang,
            41.5 / 60, 46.6 / 60, 940.29997
        )

        like= 0
        for i_theta, ts in enumerate(thet):
            #print(ts, w[i_theta], sig_w[i_theta])
            if ts>0.005:

                wi = np.interp(
                    ts,
                    ang_th / 2 / np.pi * 360,
                    ang_ang - w_IC_instance
                )
                # compare model and data with gaussian likelihood:
                like += -0.5 * (((wi - w[i_theta]) / sig_w[
                    i_theta]) ** 2)


        if obs=="Ang_z9_m9" and savedir:
            fname = str(savedir) + 'fs' + str(np.round(fstar_norm,8)) + '_sig' + str(
                np.round(sigma_SHMR,8)) + '_al' + str(np.round(alpha_star_low,8)) + '.txt'
            np.savetxt(fname, ang_ang - w_IC_instance)
        elif obs=="Ang_z5_5_m9" and savedir:
            fname = str(savedir) + 'ang_z5_5_fs' + str(np.round(fstar_norm,8)) + '_sig' + str(
                np.round(sigma_SHMR,8)) + '_al' + str(np.round(alpha_star_low,8)) + '.txt'
            np.savetxt(fname, ang_ang - w_IC_instance)
        elif obs=="Ang_z7_m9" and savedir:
            fname = str(savedir) + 'ang_z7_fs' + str(np.round(fstar_norm,8)) + '_sig' + str(np.round(sigma_SHMR,8)) + '_al' + str(np.round(alpha_star_low,8)) + '.txt'
            np.savetxt(fname, ang_ang - w_IC_instance)
        if no_call:
            return 0
        return like


class LikelihoodUVLFBase:
    """
    To be filled
    Returns
    -------

    """

    def __init__(self, params, z, hmf_choice="Tinker08", sigma_uv=True, mass_dependent_sigma_uv=False):
        self.z = z
        self.hmf_loc = hmf.MassFunction(
            z=z,
            Mmin=5,
            Mmax=19,
            dlog10m=0.05,
            hmf_model=hmf_choice
        )
        self.params = params
        self.sigma_uv = sigma_uv
        self.mass_dependent_sigma_uv = mass_dependent_sigma_uv

    def call_likelihood(
            self,
            p,
            muvs_o=None,
            uvlf_o=None,
            sig_o=None,
            use_BPASS=True,
            vect_func=None,
            bpass_read=None,
            sfr_samp_inst=None,
):
        # dic_params = dict.fromkeys(self.params, p)
        dic_params = {}
        paramida = p
        for index, pary in enumerate(self.params):
            dic_params[pary] = paramida[index]

        if "fstar_norm" in dic_params:
            fstar_norm = dic_params["fstar_norm"]
        else:
            fstar_norm = 0.0

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

        if "M_knee" in dic_params:
            M_knee = 10**dic_params["M_knee"]
        else:
            M_knee = 2.6e11

        if "sigma_UV" in dic_params:
            sigma_UV = dic_params["sigma_UV"]
        else:
            sigma_UV = 0.2

        lnL = 0
        if use_BPASS:
            if self.sigma_uv:
                preds = UV_calc_numba(
                    muvs_o,
                    np.log10(self.hmf_loc.m),
                    self.hmf_loc.dndlog10m,
                    f_star_norm=10 ** fstar_norm,
                    alpha_star=alpha_star,
                    sigma_SHMR=sigma_SHMR,
                    sigma_SFMS_norm=sigma_SFMS_norm,
                    t_star=t_star,
                    a_sig_SFR=a_sig_SFR,
                    z=self.z,
                    vect_func=vect_func,
                    bpass_read=bpass_read,
                    SFH_samp=sfr_samp_inst,
                    M_knee=M_knee,
                    sigma_kuv=sigma_UV,
                    mass_dependent_sigma_uv=self.mass_dependent_sigma_uv,
                )
            else:
                preds = UV_calc_BPASS(
                    muvs_o,
                    np.log10(self.hmf_loc.m),
                    self.hmf_loc.dndlog10m,
                    f_star_norm=10 ** fstar_norm,
                    alpha_star=alpha_star,
                    sigma_SHMR=sigma_SHMR,
                    sigma_SFMS_norm=sigma_SFMS_norm,
                    t_star=t_star,
                    a_sig_SFR=a_sig_SFR,
                    z=self.z,
                    vect_func=vect_func,
                    bpass_read=bpass_read,
                    SFH_samp=sfr_samp_inst,
                    M_knee = M_knee
                )
        else:
            preds = UV_calc(
                muvs_o,
                np.log10(self.hmf_loc.m),
                self.hmf_loc.dndlog10m,
                f_star_norm=10 ** fstar_norm,
                alpha_star=alpha_star,
                sigma_SHMR=sigma_SHMR,
                sigma_SFMS_norm=sigma_SFMS_norm,
                t_star=t_star,
                a_sig_SFR=a_sig_SFR,
                z=self.z,
                M_knee = M_knee,
            )

        for index, muvi in enumerate(muvs_o):
            if isinstance(sig_o, tuple):
                if sig_o[0][index] < 0.0:
                    if muvi==-23.5:
                        erf_modifier = -1
                    else:
                        erf_modifier = +1
                    #trick for lower limits for spectroscopic estimates
                    sig_a = - 2 * (sig_o[0][index] * sig_o[1][index]) / (
                                sig_o[0][index] + sig_o[1][index])
                    sig_b = (sig_o[0][index] - sig_o[1][index]) / (
                                sig_o[0][index] + sig_o[1][index])
                    lnL += np.log(
                        0.5*(
                            1+erf_modifier*erf(
                                (
                                    (
                                        (preds[index] - uvlf_o[index]) / np.sqrt(
                                        2
                                    )/abs(
                                            np.sqrt((sig_a + sig_b * (preds[index] - uvlf_o[index]))**2)
                                        )
                                    )
                                )
                            )
                        )
                    )
                else:
                    pred_x = np.linspace(-13, -1.0, 100000)

                    sig_a = 2 * (sig_o[0][index] * sig_o[1][index])/(sig_o[0][index] + sig_o[1][index])
                    sig_b = (sig_o[0][index] - sig_o[1][index])/(sig_o[0][index] + sig_o[1][index])
                    sig_this = sig_a + sig_b * (preds[index] - uvlf_o[index])
                    # L = intg.trapezoid(
                    #     y=np.exp(
                    #         -0.5 * ((10 ** pred_x - uvlf_o[index]) ** 2 / (
                    #                     sig_this ** 2))
                    #     ) / 2 / np.pi / sig_this / 0.5 * np.exp(
                    #         -0.5 * ((np.log10(
                    #             preds[index]) - pred_x) ** 2 / 0.5 ** 2)
                    #     ),
                    #     x=pred_x
                    # )
                    #lnL += np.log(L)
                    lnL += -0.5 * ((preds[index] - uvlf_o[index])**2 / ((sig_a + sig_b * (preds[index] - uvlf_o[index])) ** 2))
            else:
                pred_x = np.linspace(-13,-1.0, 100000)
                # L = intg.trapezoid(
                #     y = np.exp(
                #         -0.5 * (( 10**pred_x - uvlf_o[index])**2 / (sig_o[index] ** 2))
                #     ) / 2 / np.pi / sig_o[index] / 0.5 * np.exp(
                #         -0.5 * ((np.log10(preds[index]) - pred_x) ** 2 / 0.5**2)
                #     ),
                #     x = pred_x
                # )
                #lnL += np.log(L)
                lnL += -0.5 * ((preds[index] - uvlf_o[index])**2 / (sig_o[
                    index] ** 2))
        return lnL

def run_mcmc(
        likelihoods,
        params,
        mult_params=None,
        priors=None,
        covariance=False,
        diagonal=False,
        realistic_Nz=False,
        use_BPASS=True,
        M_knee=False,
        output_dir="/home/user/Documents/projects/UVLF_clust/",
        hmf_choice="Tinker08",
        exact_specs=True,
        sigma_uv=True,
        mass_dependent_sigma_uv=False,
):

    if priors is None:
        if M_knee and sigma_uv:
            priors = [(-5.0, 1.0), (0.0, 1.0), (0.05, 0.9), (0.01, 1.0),
                      (0.01, 1.0), (-1.0, 0.5), (11.5, 16.0), (0.0,0.5)]
        elif M_knee and not sigma_uv:
            priors = [(-5.0,1.0),(0.0,1.0), (0.05,0.9), (0.01,1.0), (0.01,1.0), (-1.0, 0.5), (11.5,16.0)]
        else:
            priors = [(-3.0,1.0),(0.0,1.0), (0.05,0.9), (0.01,1.0), (0.01,1.0), (-1.0, 0.5)]
    #initialize likelihoods
    #print(likelihoods)
    output_filename = output_dir#"/home/inikolic/projects/UVLF_FMs/run_speed/runs_may/all_ang_prior/"
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
    uvlf = False
    ang = False
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

    if any({"Ang_z9_m87", "Ang_z9_m9"}.intersection(set(likelihoods))
           ):
        ang = True
        AngBase_z9 = LikelihoodAngBase(params, realistic_Nz=realistic_Nz, hmf_choice=hmf_choice, z=9.25, exact_specs=exact_specs)
    if any({"Ang_z7_m87",
            "Ang_z7_m93", "Ang_z7_m9"}.intersection(set(likelihoods))
           ):
        ang = True
        AngBase_z7 = LikelihoodAngBase(params, realistic_Nz=realistic_Nz,
                                       hmf_choice=hmf_choice, z=7, exact_specs=exact_specs)
    if any({"Ang_z5_5_m85", "Ang_z5_5_m9", "Ang_z5_5_m92_5",
            "Ang_z5_5_m9_5"}.intersection(set(likelihoods))
           ):# or "Ang_z9_m9" in likelihoods or "Ang_z7_m9" in likelihoods:
        ang = True
        AngBase_z5 = LikelihoodAngBase(params, realistic_Nz=realistic_Nz, hmf_choice=hmf_choice, z=5.5, exact_specs=exact_specs)
    if not ang:
        AngBase = LikelihoodAngBase(params, realistic_Nz=realistic_Nz, hmf_choice=hmf_choice, exact_specs=exact_specs)

    SFR_samp_11 = None
    SFR_samp_10 = None
    SFR_samp_9 = None
    SFR_samp_8 = None
    SFR_samp_7 = None
    SFR_samp_9_8 = None
    SFR_samp_12_5 = None

    if "UVLF_z11_McLeod23" in likelihoods:
        uvlf = True
        UVLFBase_Mc11 = LikelihoodUVLFBase(params, z=11, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_11 = SFH_sampler(z=11)
    if "UVLF_z9_Donnan24" in likelihoods:
        uvlf = True
        UVLFBase_Don24 = LikelihoodUVLFBase(params, z=9, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_9 = SFH_sampler(z=9)

    if "UVLF_z10_Donnan24" in likelihoods:
        uvlf = True
        UVLFBase_Don24_10 = LikelihoodUVLFBase(params, z=10, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_10 = SFH_sampler(z=10)

    if "UVLF_z11_Donnan24" in likelihoods:
        uvlf = True
        UVLFBase_Don24_11 = LikelihoodUVLFBase(params, z=11, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_11 = SFH_sampler(z=11)


    if "UVLF_z12_5_Donnan24" in likelihoods:
        uvlf = True
        UVLFBase_Don24_12_5 = LikelihoodUVLFBase(params, z=12.5, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_12_5 = SFH_sampler(z=12.5)

    if "UVLF_z7_Harikane24" in likelihoods:
        uvlf = True
        UVLFBase_Har24_7 = LikelihoodUVLFBase(params, z=7, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_7 = SFH_sampler(z=7)

    if "UVLF_z8_Harikane24" in likelihoods:
        uvlf = True
        UVLFBase_Har24_8 = LikelihoodUVLFBase(params, z=8, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_8 = SFH_sampler(z=8)

    if "UVLF_z9_Harikane24" in likelihoods:
        uvlf = True
        UVLFBase_Har24_9 = LikelihoodUVLFBase(params, z=9, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_9 = SFH_sampler(z=9)

    if "UVLF_z10_Harikane24" in likelihoods:
        uvlf = True
        UVLFBase_Har24_10 = LikelihoodUVLFBase(params, z=10, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_10 = SFH_sampler(z=10)

    if "UVLF_z12_Harikane24" in likelihoods:
        uvlf = True
        UVLFBase_Har24_12 = LikelihoodUVLFBase(params, z=12, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_12 = SFH_sampler(z=12)

    if "UVLF_z14_Harikane24" in likelihoods:
        uvlf = True
        UVLFBase_Har24_14 = LikelihoodUVLFBase(params, z=14, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_14 = SFH_sampler(z=14)

    if "UVLF_z8_Willot23" in likelihoods:
        uvlf = True
        UVLFBase_Wil23_8 = LikelihoodUVLFBase(params, z=8, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_8 = SFH_sampler(z=8)
    if "UVLF_z9_Willot23" in likelihoods:
        uvlf = True
        UVLFBase_Wil23_9 = LikelihoodUVLFBase(params, z=9, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_9 = SFH_sampler(z=9)
    if "UVLF_z10_Willot23" in likelihoods:
        uvlf = True
        UVLFBase_Wil23_10 = LikelihoodUVLFBase(params, z=10, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_10 = SFH_sampler(z=10)
    if "UVLF_z12_Willot23" in likelihoods:
        uvlf = True
        UVLFBase_Wil23_12 = LikelihoodUVLFBase(params, z=12, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_12 = SFH_sampler(z=12)

    if "UVLF_z9_8_Whitler25" in likelihoods:
        uvlf = True
        UVLFBase_Whitler25_9_8 = LikelihoodUVLFBase(params, z=9.8, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_9_8 = SFH_sampler(z=9.8)
    if "UVLF_z12_8_Whitler25" in likelihoods:
        uvlf = True
        UVLFBase_Whitler25_12_8 = LikelihoodUVLFBase(params, z=12.8, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_12_8 = SFH_sampler(z=12.8)
    if "UVLF_z14_3_Whitler25" in likelihoods:
        uvlf = True
        UVLFBase_Whitler25_14_3 = LikelihoodUVLFBase(params, z=14.3, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_14_3 = SFH_sampler(z=14.3)

    if "UVLF_z9_Finkelstein24" in likelihoods:
        uvlf = True
        UVLFBase_Fin24_9 = LikelihoodUVLFBase(params, z=9, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_9 = SFH_sampler(z=9)
    if "UVLF_z11_Finkelstein24" in likelihoods:
        uvlf = True
        UVLFBase_Fin24_11 = LikelihoodUVLFBase(params, z=11, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_11 = SFH_sampler(z=11)
    if "UVLF_z14_Finkelstein24" in likelihoods:
        uvlf = True
        UVLFBase_Fin24_14 = LikelihoodUVLFBase(params, z=14, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_14 = SFH_sampler(z=14)

    if "UVLF_z5_Bouwens21" in likelihoods:
        uvlf = True
        UVLFBase_Bouwens21_5 = LikelihoodUVLFBase(params, z=5, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_5 = SFH_sampler(z=5)
    if "UVLF_z6_Bouwens21" in likelihoods:
        uvlf = True
        UVLFBase_Bouwens21_6 = LikelihoodUVLFBase(params, z=6, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_6 = SFH_sampler(z=6)
    if "UVLF_z7_Bouwens21" in likelihoods:
        uvlf = True
        UVLFBase_Bouwens21_7 = LikelihoodUVLFBase(params, z=7, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_7 = SFH_sampler(z=7)
    if "UVLF_z8_Bouwens21" in likelihoods:
        uvlf = True
        UVLFBase_Bouwens21_8 = LikelihoodUVLFBase(params, z=8, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_8 = SFH_sampler(z=8)
    if "UVLF_z9_Bouwens21" in likelihoods:
        uvlf = True
        UVLFBase_Bouwens21_9 = LikelihoodUVLFBase(params, z=9, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_9 = SFH_sampler(z=9)
    if "UVLF_z10_Bouwens21" in likelihoods:
        uvlf = True
        UVLFBase_Bouwens21_10 = LikelihoodUVLFBase(params, z=10, hmf_choice=hmf_choice, sigma_uv=sigma_uv, mass_dependent_sigma_uv=mass_dependent_sigma_uv)
        SFR_samp_10 = SFH_sampler(z=10)

    if uvlf and use_BPASS:
        bpass_read = bpass_loader()
        vect_func = np.vectorize(bpass_read.get_UV)
    else:
        bpass_read = None
        vect_func = None


    observations_inst = Observations(True, True)

    def likelihood(p, ndim, nparams, lnew):
        lnL = 0
        p_new = np.zeros((ndim))
        for i in range(ndim):
            p_new[i] = p[i]
        for li in likelihoods:
            if li == "Ang_z9_m87":
                thet, w, wsig = observations_inst.get_obs_z9_m87()
                lnL+=AngBase_z9.call_likelihood(
                    p_new,
                    obs="Ang_z9_m87",
                    thet=thet,
                    w=w,
                    sig_w=wsig
                )
            elif li == "Ang_z9_m9":
                thet, w, wsig = observations_inst.get_obs_z9_m90()
                lnL+=AngBase_z9.call_likelihood(
                    p_new,
                    obs="Ang_z9_m9",
                    thet=thet,
                    w=w,
                    sig_w=wsig,
                    savedir=output_filename,
                )
            elif li == "Ang_z7_m87":
                thet, w, wsig = observations_inst.get_obs_z7_m87()
                lnL+=AngBase_z7.call_likelihood(
                    p_new,
                    obs="Ang_z7_m87",
                    thet=thet,
                    w=w,
                    sig_w=wsig,
                    savedir=output_filename,
                )
            elif li == "Ang_z7_m9":
                thet, w, wsig = observations_inst.get_obs_z7_m90()
                lnL+=AngBase_z7.call_likelihood(
                    p_new,
                    obs="Ang_z7_m9",
                    thet=thet,
                    w=w,
                    sig_w=wsig,
                    savedir=output_filename,
                )
            elif li == "Ang_z7_m93":
                thet, w, wsig = observations_inst.get_obs_z7_m93()
                lnL+=AngBase_z7.call_likelihood(
                    p_new,
                    obs="Ang_z7_m93",
                    thet=thet,
                    w=w,
                    sig_w=wsig,
                    savedir=output_filename,
                )
            elif li == "Ang_z5_5_m85":
                thet, w, wsig = observations_inst.get_obs_z5_5_m85()
                lnL+=AngBase_z5.call_likelihood(
                    p_new,
                    obs="Ang_z5_5_m85",
                    thet=thet,
                    w=w,
                    sig_w=wsig,
                    savedir=output_filename,
                )
            elif li == "Ang_z5_5_m9":
                thet, w, wsig = observations_inst.get_obs_z5_5_m90()
                lnL+=AngBase_z5.call_likelihood(
                    p_new,
                    obs="Ang_z5_5_m9",
                    thet=thet,
                    w=w,
                    sig_w=wsig,
                    savedir=output_filename,
                )
            elif li == "Ang_z5_5_m92_5":
                thet, w, wsig = observations_inst.get_obs_z5_5_m92_5()
                lnL+=AngBase_z5.call_likelihood(
                    p_new,
                    obs="Ang_z5_5_m92_5",
                    thet=thet,
                    w=w,
                    sig_w=wsig,
                    savedir=output_filename,
                )
            elif li == "Ang_z5_5_m9_5":
                thet, w, wsig = observations_inst.get_obs_z5_5_m95()
                lnL+=AngBase_z5.call_likelihood(
                    p_new,
                    obs="Ang_z5_5_m9_5",
                    thet=thet,
                    w=w,
                    sig_w=wsig,
                    savedir=output_filename,
                )
            elif li == "UVLF_z11_McLeod23":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z11_McLeod23()
                muvs_mask = [muvs_o >= -20.0]
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = sig_o[muvs_mask]
                lnL+=UVLFBase_Mc11.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst = SFR_samp_11,
                    bpass_read = bpass_read,
                    vect_func = vect_func,
                )
                if ang==False:
                    thet, w, wsig = observations_inst.get_obs_z9_m90()
                    _ = AngBase.call_likelihood(
                        p_new,
                        obs="Ang_z9_m9",
                        thet=thet,
                        w=w,
                        sig_w=wsig,
                        savedir=output_filename,
                        no_call=True
                    )
            elif li == "UVLF_z9_Donnan24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z9_Donnan24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])
                lnL+=UVLFBase_Don24.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_9,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
                if ang==False and len(likelihoods) == 1:
                    thet, w, wsig = observations_inst.get_obs_z9_m90()
                    _ = AngBase.call_likelihood(
                        p_new,
                        obs="Ang_z9_m9",
                        thet=thet,
                        w=w,
                        sig_w=wsig,
                        savedir=output_filename,
                        no_call=True
                    )
                if ang==False:
                    thet, w, wsig = observations_inst.get_obs_z7_m90()
                    _ = AngBase.call_likelihood(
                        p_new,
                        obs="Ang_z7_m9",
                        thet=thet,
                        w=w,
                        sig_w=wsig,
                        savedir=output_filename,
                        no_call=True
                    )
            elif li == "UVLF_z10_Donnan24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z10_Donnan24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])
                lnL+=UVLFBase_Don24_10.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst = SFR_samp_10,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li == "UVLF_z11_Donnan24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z11_Donnan24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])
                lnL+=UVLFBase_Don24_11.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_11,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li == "UVLF_z12_5_Donnan24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z12_5_Donnan24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])
                lnL+=UVLFBase_Don24_12_5.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_12_5,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li == "UVLF_z7_Harikane24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z7_Harikane24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Har24_7.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_7,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li == "UVLF_z8_Harikane24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z8_Harikane24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Har24_8.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_8,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li == "UVLF_z9_Harikane24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z9_Harikane24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Har24_9.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_9,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
                if not ang:
                    thet, w, wsig = observations_inst.get_obs_z9_m90()
                    _ = AngBase.call_likelihood(
                        p_new,
                        obs="Ang_z9_m9",
                        thet=thet,
                        w=w,
                        sig_w=wsig,
                        savedir=output_filename,
                        no_call=True
                    )
                    thet, w, wsig = observations_inst.get_obs_z7_m90()
                    _ = AngBase.call_likelihood(
                        p_new,
                        obs="Ang_z7_m9",
                        thet=thet,
                        w=w,
                        sig_w=wsig,
                        savedir=output_filename,
                        no_call=True
                    )
            elif li == "UVLF_z10_Harikane24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z10_Harikane24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Har24_10.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_10,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )

            elif li == "UVLF_z12_Harikane24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z12_Harikane24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Har24_12.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_12,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )

            elif li == "UVLF_z14_Harikane24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z14_Harikane24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Har24_14.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_14,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li == "UVLF_z8_Willot23":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z8_Willot23()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Wil23_8.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_8,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li == "UVLF_z9_Willot23":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z9_Willot23()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Wil23_9.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_9,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li == "UVLF_z10_Willot23":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z10_Willot23()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Wil23_10.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_10,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li == "UVLF_z12_Willot23":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z12_Willot23()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Wil23_12.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_12,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li=="UVLF_z9_8_Whitler25":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z9_8_Whitler25()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Whitler25_9_8.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_9_8,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li=="UVLF_z12_8_Whitler25":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z12_8_Whitler25()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Whitler25_12_8.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_12_8,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li=="UVLF_z14_3_Whitler25":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z14_3_Whitler25()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Whitler25_14_3.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_14_3,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li=="UVLF_z9_Finkelstein24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z9_Finkelstein24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Fin24_9.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_9,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li=="UVLF_z11_Finkelstein24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z11_Finkelstein24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Fin24_11.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_11,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li=="UVLF_z14_Finkelstein24":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z14_Finkelstein24()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_c = (sig_o[0][muvs_mask], sig_o[1][muvs_mask])

                lnL+=UVLFBase_Fin24_14.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_c,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_14,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li=="UVLF_z5_Bouwens21":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z5_Bouwens21()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_o_arr = np.array(sig_o)
                sig_c = sig_o_arr[muvs_mask]
                sig_c = [tuple(i) for i in sig_c]
                lnL+=UVLFBase_Bouwens21_5.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_o,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_5,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )

            elif li=="UVLF_z6_Bouwens21":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z6_Bouwens21()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_o_arr = np.array(sig_o)
                sig_c = sig_o_arr[muvs_mask]
                sig_c = [tuple(i) for i in sig_c]
                lnL+=UVLFBase_Bouwens21_6.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_o,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_6,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li=="UVLF_z7_Bouwens21":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z7_Bouwens21()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_o_arr = np.array(sig_o)
                sig_c = sig_o_arr[muvs_mask]
                sig_c = [tuple(i) for i in sig_c]
                lnL+=UVLFBase_Bouwens21_7.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_o,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_7,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li=="UVLF_z8_Bouwens21":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z8_Bouwens21()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_o_arr = np.array(sig_o)
                sig_c = sig_o_arr[muvs_mask]
                sig_c = [tuple(i) for i in sig_c]
                lnL+=UVLFBase_Bouwens21_8.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_o,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_8,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li=="UVLF_z9_Bouwens21":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z9_Bouwens21()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_o_arr = np.array(sig_o)
                sig_c = sig_o_arr[muvs_mask]
                sig_c = [tuple(i) for i in sig_c]
                lnL+=UVLFBase_Bouwens21_9.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_o,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_9,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
                )
            elif li == "UVLF_z10_Bouwens21":
                muvs_o, uvlf_o, sig_o = observations_inst.get_obs_uvlf_z10_Bouwens21()
                muvs_mask = muvs_o >= -20.0
                muvs_c = muvs_o[muvs_mask]
                uvlf_c = uvlf_o[muvs_mask]
                sig_o_arr = np.array(sig_o)
                sig_c = sig_o_arr[muvs_mask]
                sig_c = [tuple(i) for i in sig_c]
                lnL += UVLFBase_Bouwens21_10.call_likelihood(
                    p_new,
                    muvs_o=muvs_c,
                    uvlf_o=uvlf_c,
                    sig_o=sig_o,
                    use_BPASS=use_BPASS,
                    sfr_samp_inst=SFR_samp_10,
                    bpass_read=bpass_read,
                    vect_func=vect_func,
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

            if M_knee and not sigma_uv:
                # cov_mat = np.loadtxt(
                #     '/home/inikolic/projects/UVLF_FMs/priors/cov_matr_Mknee.txt'
                # ) * 4.0
                # mu = np.loadtxt(
                #     '/home/inikolic/projects/UVLF_FMs/priors/means_Mknee.txt'
                # )
                cov_mat = np.loadtxt(
                    '/home/inikolic/projects/UVLF_FMs/angular_clustering_debug/new_prior_analysis/cov_matr_Mknee_wide.txt'
                )
                mu = np.loadtxt(
                    '/home/inikolic/projects/UVLF_FMs/angular_clustering_debug/new_prior_analysis/means_Mknee_wide.txt'
                )
            elif M_knee and sigma_uv:
                cov_mat = np.loadtxt(
                    '/home/inikolic/projects/UVLF_FMs/angular_clustering_debug/new_prior_analysis/cov_matr_uv.txt'
                )
                mu = np.loadtxt(
                    '/home/inikolic/projects/UVLF_FMs/angular_clustering_debug/new_prior_analysis/means_uv.txt'
                )
            else:
                cov_mat = 2 * np.loadtxt(
                    '/home/inikolic/projects/UVLF_FMs/priors/cov_matr_SMHM.txt'
                ) #my default is twice the covariance.
                mu = np.loadtxt(
                    '/home/inikolic/projects/UVLF_FMs/priors/means_goodSMHM.txt'
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
            return cube

    result = pymultinest.run(
        LogLikelihood=likelihood,
        Prior=prior,
        n_dims=len(params),
        **mult_params
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        type=str,
        default="/home/user/Documents/projects/UVLF_clust/analysis_post/Harikane_only/"
    )
    parser.add_argument(
        "-p",
        "--params-list",
        nargs='+',
        default=[
            "fstar_norm",
            "sigma_SHMR",
            "t_star",
            "alpha_star_low",
            "sigma_SFMS_norm",
            "a_sig_SFR"
        ]
    )
    parser.add_argument("-l", "--names-list", nargs='+', default=[])
    parser.add_argument("--covariance", action="store_false")
    parser.add_argument("--diagonal", action="store_false")
    parser.add_argument("--realistic_Nz", action="store_false")
    parser.add_argument("--use_Mknee", action="store_true")
    parser.add_argument("--hmf", type=str, default="Tinker08")
    parser.add_argument(
        '--exact_specs',
        action="store_false",
        help="Use exact specifications from Paquereau+25 in computing\n integral constraint and use only linear bias",
    )
    parser.add_argument("--sigma_uv", action="store_false")
    parser.add_argument("--mass_dependent_sigma_uv", action="store_true")
    inputs = parser.parse_args()
    likelihoods = inputs.names_list

    # likelihoods = [
    #     # "UVLF_z11_McLeod23",
    #     # "UVLF_z9_Donnan24",
    #     # "UVLF_z10_Donnan24",
    #     # "UVLF_z11_Donnan24",
    #     # "UVLF_z12_5_Donnan24",
    #     # "UVLF_z9_Harikane24",
    #     # "UVLF_z10_Harikane24",
    #     # "UVLF_z12_Harikane24",
    #     # "UVLF_z14_Harikane24",
    #     # "UVLF_z8_Willot23",
    #     # "UVLF_z9_Willot23",
    #     # "UVLF_z10_Willot23",
    #     # "UVLF_z12_Willot23",
    #     # "UVLF_z9_8_Whitler25",
    #     # "UVLF_z12_8_Whitler25",
    #     # "UVLF_z14_3_Whitler25",
    #     # "UVLF_z9_Finkelstein24",
    #     # "UVLF_z11_Finkelstein24",
    #     # "UVLF_z14_Finkelstein24",
    #     "Ang_z9_m9",
    #     "Ang_z9_m87",
    #     "Ang_z7_m9",
    #     "Ang_z7_m87"
    #     "Ang_z7_m93",
    #     "Ang_z5_5_m85",
    #     "Ang_z5_5_m9",
    #     "Ang_z5_5_m9_25",
    #     "Ang_z5_5_m9_5"
    # ]
    params = inputs.params_list

    if not os.path.exists(inputs.output_directory):
        os.makedirs(inputs.output_directory, exist_ok=True)
    if params == ["fstar_norm", "sigma_SHMR", "t_star", "alpha_star_low", "sigma_SFMS_norm", "a_sig_SFR",]:
        priors = [(-3.0, 1.0), (0.001, 2.0), (0.001, 1.0), (0.0, 2.0),
                  (0.001, 1.2), (-1.0, 0.5)]
    elif params == ["fstar_norm", "sigma_SHMR", "t_star", "alpha_star_low", "sigma_SFMS_norm", "a_sig_SFR", "M_knee"]:
        priors = [(-6.0, 1.0), (0.001, 2.0), (0.001, 1.0), (0.0, 2.0),
                  (0.001, 1.5), (-1.0, 0.5), (11.5,16.0)]
    elif params == ["fstar_norm", "sigma_SHMR", "t_star", "alpha_star_low", "sigma_SFMS_norm", "a_sig_SFR", "M_knee", "sigma_UV"]:
        priors = [(-6.0, 1.0), (0.001, 2.0), (0.001, 1.0), (0.0, 2.0),
                  (0.001, 1.5), (-1.0, 0.5), (11.5,16.0), (0.001,0.5)]
        if not inputs.sigma_uv:
            raise ValueError("You need to set --sigma_uv to use sigma_UV parameter.")
    elif params == ["fstar_norm", "sigma_SHMR", "alpha_star_low"]:
        priors = [(-3.0,1.0), (0.001,2.0), (0.0,2.0)]
    elif params == ["fstar_norm", "sigma_SHMR", "t_star", "alpha_star_low", "sigma_SFMS_norm", "a_sig_SFR", "M_knee", "alpha_z_SHMR"]:
        priors = [(-5.0, 1.0), (0.001, 2.0), (0.001, 1.0), (0.0, 2.0),
                  (0.001, 1.2), (-1.0, 0.5), (11.5,16.0), (-1.0,2.0)]
    else:
        raise ValueError("Invalid parameter list provided.")

    if inputs.mass_dependent_sigma_uv:
        if "sigma_UV" not in params or not inputs.sigma_uv:
            raise ValueError("You need to include 'sigma_UV' in the params list to use mass-dependent sigma_UV.")

    #, "M_knee"]
    #params = ["fstar_norm", "sigma_SHMR", "alpha_star_low",]
    #priors = [(-3.0,0.0), (0.001,2.0), (0.0,1.2)]
   # priors = [(-5.0, 1.0), (0.001, 2.0), (0.001, 1.0), (0.0, 2.0), (0.001, 1.2),
   #          (-1.0, 0.5)]#, (10.0,16.0)]

    #priors = [(-1.0,1.0),(0.01,1.0), (0.0,1.0)]
    #more possibilities: "M_1", "M_0", "alpha" -> relating to satellite params.
    #new possibility: "a_sig_SFR" -> relating to sigma_SFMS scaling with stellar mass.
    #"write a list of all possible parameters"

    run_mcmc(
        likelihoods,
        params,
        priors=priors,
        covariance=inputs.covariance,
        diagonal=inputs.diagonal,
        realistic_Nz=inputs.realistic_Nz,
        M_knee=inputs.use_Mknee,
        output_dir = inputs.output_directory,
        hmf_choice = inputs.hmf,
        exact_specs=inputs.exact_specs,
        sigma_uv=inputs.sigma_uv,
        mass_dependent_sigma_uv=inputs.mass_dependent_sigma_uv,
    )
