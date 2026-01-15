import os
import numpy as np
import hmf as hmf
from astropy.cosmology import Planck18 as cosmo

try:
    from uvlf import UV_calc_numba as uvlf_func
except ImportError:
    from uvlf import UV_calc_BPASS_op as uvlf_func

from uvlf import bpass_loader, SFH_sampler

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory_of_posteriors",
        type=str,
        default="/home/user/Documents/projects/UVLF_clust/analysis_post/Harikane_only/"
    ) #this is also where database will be stored.
    z_s = [6.0,8.0,10.0,11.0,12.5,14.0]
    muvs_o = np.linspace(-25,-16,20)
    posteriors =  np.genfromtxt(parser.parse_args().directory_of_posteriors + "post_equal_weights.dat")
    hmf_locs = [
        hmf.MassFunction(
            z=z,
            Mmin=5,
            Mmax=19,
            dlog10m=0.05,
            hmf_model="Tinker08",
        ) for z in z_s
    ]
    SFR_samps = [
        SFH_sampler(z=z) for z in z_s
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir == "/groups/astro/ivannik/programs/JWST-Inference":
        bpass_read = bpass_loader(
            filename='/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00.',
        )
    else:
        bpass_read = bpass_loader()
    vect_func = np.vectorize(bpass_read.get_UV)
    for index_in, post_sample in enumerate(posteriors):
        preds = np.zeros((len(z_s), len(muvs_o)))
        for index_z, z in enumerate(z_s):
            preds[index_z] = uvlf_func(
                muvs_o,
                np.log10(hmf_locs[index_z].m/ cosmo.h),
                hmf_locs[index_z].dndlog10m * cosmo.h**3 * np.exp(- 5e8 / (hmf_locs[index_z].m / cosmo.h) ),
                f_star_norm=10 ** post_sample[0],
                alpha_star=post_sample[3],
                sigma_SHMR=post_sample[1],
                sigma_SFMS_norm=post_sample[4],
                t_star=post_sample[2],
                a_sig_SFR=post_sample[5],
                z=z,
                vect_func=vect_func,
                bpass_read=bpass_read,
                SFH_samp=SFR_samps[index_z],
                M_knee=10**post_sample[6],
                slope_SFR=post_sample[7],
                sigma_kuv=post_sample[8],
                mass_dependent_sigma_uv=True,
            )
        np.savetxt(parser.parse_args().directory_of_posteriors + f"UVLFs_{post_sample[0]:.8f}.txt", preds)
