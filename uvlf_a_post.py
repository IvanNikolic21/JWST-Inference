import numpy as np
import hmf as hmf

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
    parser.add_argument("--z_dependent_SHMR", action="store_true")
    parser.add_argument("--dependence_on_alpha_star", action="store_true")
    z_s = [6.0,8.0,10.0,11.0,12.5,14.0]
    muvs_o = np.linspace(-23,-16,20)
    posteriors =  np.loadtxt(parser.parse_args().directory_of_posteriors + "post_equal_weights.dat")
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
    bpass_read = bpass_loader()
    vect_func = np.vectorize(bpass_read.get_UV)
    for index_in, post_sample in enumerate(posteriors):
        preds = np.zeros((len(z_s), len(muvs_o)))
        for index_z, z in enumerate(z_s):
            preds[index_z] = uvlf_func(
                muvs_o,
                np.log10(hmf_locs[index_z].m),
                hmf_locs[index_z].dndlog10m,
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
                sigma_kuv=post_sample[8],
                #z_dependent_SHMR=parser.parse_args().z_dependent_SHMR,
                mass_dependent_sigma_uv=True,
                alpha_z_SHMR=post_sample[7],
                dependence_on_alpha_star=parser.parse_args().dependence_on_alpha_star,
            )
        np.savetxt(parser.parse_args().directory_of_posteriors + f"UVLFs_{post_sample[0]:.8f}.txt", preds)
