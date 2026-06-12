import os
import json
import numpy as np
import hmf as hmf
from astropy.cosmology import Planck18 as cosmo
from uvlf import bpass_loader, SFH_sampler
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory_of_posteriors",
        type=str,
        required=True,
    )
    directory = parser.parse_args().directory_of_posteriors

    with open(os.path.join(directory, "run_config.json")) as f:
        run_config = json.load(f)

    func_name              = run_config["uvlf_func"]
    params                 = run_config["params"]
    sigma_sfr_10_explicit  = run_config["sigma_sfr_10_explicit"]
    sigma_uv               = run_config["sigma_uv"]
    mass_dependent_sigma_uv = run_config.get("mass_dependent_sigma_uv", False)
    mass_dependent_sfr10   = run_config.get("mass_dependent_sfr10", False)

    if func_name == "UV_calc_numba_sfr10":
        from uvlf import UV_calc_numba_sfr10 as uvlf_func
    elif func_name == "UV_calc_numba":
        from uvlf import UV_calc_numba as uvlf_func
    else:
        from uvlf import UV_calc_BPASS as uvlf_func

    z_s    = [6.0, 8.0, 10.0, 11.0, 12.5, 14.0]
    muvs_o = np.linspace(-25, -16, 20)

    posteriors = np.genfromtxt(os.path.join(directory, "post_equal_weights.dat"))
    hmf_locs = [
        hmf.MassFunction(z=z, Mmin=5, Mmax=19, dlog10m=0.05, hmf_model="Tinker08")
        for z in z_s
    ]
    SFR_samps = [SFH_sampler(z=z) for z in z_s]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir == "/groups/astro/ivannik/programs/JWST-Inference":
        bpass_read = bpass_loader(
            filename="/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00.",
        )
    else:
        bpass_read = bpass_loader()

    vect_func = np.vectorize(
        bpass_read.get_UV_sfr10 if sigma_sfr_10_explicit else bpass_read.get_UV
    )

    for post_sample in posteriors:
        dic = dict(zip(params, post_sample))

        kwargs = dict(
            f_star_norm     = 10 ** dic.get("fstar_norm", 0.0),
            alpha_star      = dic.get("alpha_star_low", 0.5),
            sigma_SHMR      = dic.get("sigma_SHMR", 0.3),
            sigma_SFMS_norm = dic.get("sigma_SFMS_norm", 0.0),
            t_star          = dic.get("t_star", 0.5),
            a_sig_SFR       = dic.get("a_sig_SFR", -0.11654893),
            M_knee          = 10 ** dic["M_knee"] if "M_knee" in dic else 2.6e11,
        )
        if sigma_sfr_10_explicit:
            kwargs["sigma_sfr10"]          = dic.get("sigma_sfr_10", dic.get("sigma_SFR_10", 0.2))
            kwargs["mass_dependent_sfr10"] = mass_dependent_sfr10
        elif sigma_uv:
            kwargs["sigma_kuv"]                = dic.get("sigma_UV", 0.2)
            kwargs["mass_dependent_sigma_uv"]  = mass_dependent_sigma_uv

        preds = np.zeros((len(z_s), len(muvs_o)))
        for index_z, z in enumerate(z_s):
            preds[index_z] = uvlf_func(
                muvs_o,
                np.log10(hmf_locs[index_z].m / cosmo.h),
                hmf_locs[index_z].dndlog10m * cosmo.h ** 3
                    * np.exp(-5e8 / (hmf_locs[index_z].m / cosmo.h)),
                z=z,
                vect_func=vect_func,
                bpass_read=bpass_read,
                SFH_samp=SFR_samps[index_z],
                **kwargs,
            )
        np.savetxt(
            os.path.join(directory, f"UVLFs_{post_sample[0]:.8f}.txt"),
            preds,
        )