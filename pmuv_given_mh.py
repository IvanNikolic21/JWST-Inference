import os
import json
import numpy as np
import hmf as hmf
from astropy.cosmology import Planck18 as cosmo
from uvlf import bpass_loader, SFH_sampler, p_muv_given_mh_sfr10
from mpi4py import MPI
import argparse


def gaussian_moments(muv_grid, pdf):
    """
    Weighted mean/sigma of pdf (..., Nmuv) over muv_grid (Nmuv,), plus the
    fraction of probability mass actually captured by the grid.

    pdf is a density (not pre-normalized to sum to 1 over muv_grid), so
    captured_frac = sum(pdf) * dmuv estimates how much of the true p(Muv|Mh)
    falls inside [muv_grid[0], muv_grid[-1]]. If this is well below 1, the
    grid is clipping the distribution and mu/sigma below are moments of a
    truncated distribution, not the true ones.
    """
    dmuv = muv_grid[1] - muv_grid[0]
    raw_norm = np.sum(pdf, axis=-1)
    captured_frac = raw_norm * dmuv
    norm = np.where(raw_norm > 0, raw_norm, 1.0)
    mu = np.sum(pdf * muv_grid, axis=-1) / norm
    var = np.sum(pdf * (muv_grid - mu[..., None]) ** 2, axis=-1) / norm
    sigma = np.sqrt(np.maximum(var, 0.0))
    return mu, sigma, captured_frac


def build_kwargs(dic, fixed_Mknee, sigma_sfr_10_explicit, mass_dependent_sfr10):
    kwargs = dict(
        f_star_norm     = 10 ** dic.get("fstar_norm", 0.0),
        alpha_star      = dic.get("alpha_star_low", 0.5),
        sigma_SHMR      = dic.get("sigma_SHMR", 0.3),
        sigma_SFMS_norm = dic.get("sigma_SFMS_norm", 0.0),
        t_star          = dic.get("t_star", 0.5),
        a_sig_SFR       = dic.get("a_sig_SFR", -0.11654893),
        M_knee          = (
            2e12 if fixed_Mknee
            else 10 ** dic["M_knee"] if "M_knee" in dic else 2.6e11
        ),
    )
    if sigma_sfr_10_explicit:
        kwargs["sigma_sfr10"]          = dic.get("sigma_sfr_10", dic.get("sigma_SFR_10", 0.2))
        kwargs["mass_dependent_sfr10"] = mass_dependent_sfr10
    return kwargs


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory_of_posteriors",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute and overwrite Mu/Sigma/CapturedFrac files even if they "
             "already exist for a posterior sample. Default is to skip samples "
             "that already have output (e.g. from a previous run).",
    )
    args = parser.parse_args()
    directory = args.directory_of_posteriors
    recompute = args.recompute

    with open(os.path.join(directory, "run_config.json")) as f:
        run_config = json.load(f)

    params                 = run_config["params"]
    sigma_sfr_10_explicit  = run_config["sigma_sfr_10_explicit"]
    fixed_Mknee             = run_config.get("fixed_Mknee", False)
    mass_dependent_sfr10   = run_config.get("mass_dependent_sfr10", False)

    if not sigma_sfr_10_explicit:
        raise NotImplementedError(
            "p_muv_given_mh_sfr10 only supports runs with sigma_sfr_10_explicit=True."
        )

    z_s      = [6.0, 8.0, 10.0, 12.0]
    mh_eval  = np.arange(8.5, 12.0 + 1e-9, 0.1)   # log10 Mh grid
    # Wide enough to bound p(Muv|Mh) across the full mh_eval range: faint
    # halos can have most of their probability fainter than -16, and the
    # SFR10/SFR burst tail can push massive halos brighter than -25. See
    # CAPTURE_THRESHOLD check below for a per-sample diagnostic of this.
    muv_grid = np.arange(-30.0, -9.0 + 1e-9, 0.1)
    CAPTURE_THRESHOLD = 0.99

    posteriors = np.genfromtxt(os.path.join(directory, "post_equal_weights.dat"))
    my_posteriors = posteriors[rank::size]
    print(f"[rank {rank}/{size}] processing {len(my_posteriors)} of {len(posteriors)} samples", flush=True)

    hmf_locs = [
        hmf.MassFunction(z=z, Mmin=5, Mmax=19, dlog10m=0.05, hmf_model="Tinker08")
        for z in z_s
    ]
    masses_hmf_per_z = [np.log10(h.m / cosmo.h) for h in hmf_locs]
    SFR_samps = [SFH_sampler(z=z) for z in z_s]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir == "/groups/astro/ivannik/programs/JWST-Inference":
        bpass_read = bpass_loader(
            filename="/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00.",
        )
    else:
        bpass_read = bpass_loader()

    vect_func = np.vectorize(bpass_read.get_UV_sfr10)

    for post_sample in my_posteriors:
        mu_path      = os.path.join(directory, f"MuMuvMh_{post_sample[0]:.8f}.txt")
        sigma_path   = os.path.join(directory, f"SigmaMuvMh_{post_sample[0]:.8f}.txt")
        captured_path = os.path.join(directory, f"CapturedFracMuvMh_{post_sample[0]:.8f}.txt")
        if not recompute and os.path.exists(mu_path) and os.path.exists(sigma_path):
            continue

        dic = dict(zip(params, post_sample))
        kwargs = build_kwargs(dic, fixed_Mknee, sigma_sfr_10_explicit, mass_dependent_sfr10)

        mus           = np.zeros((len(z_s), len(mh_eval)))
        sigmas        = np.zeros((len(z_s), len(mh_eval)))
        captured_frac = np.zeros((len(z_s), len(mh_eval)))
        for index_z, z in enumerate(z_s):
            pdf = p_muv_given_mh_sfr10(
                muv_grid,
                masses_hmf_per_z[index_z],
                mh_eval,
                z=z,
                vect_func=vect_func,
                bpass_read=bpass_read,
                SFH_samp=SFR_samps[index_z],
                **kwargs,
            )
            mus[index_z], sigmas[index_z], captured_frac[index_z] = gaussian_moments(muv_grid, pdf)

        if np.any(captured_frac < CAPTURE_THRESHOLD):
            n_bad = np.sum(captured_frac < CAPTURE_THRESHOLD)
            worst = np.min(captured_frac)
            print(
                f"[rank {rank}] WARNING sample {post_sample[0]:.8f}: muv_grid only "
                f"captures {worst:.3f} of p(Muv|Mh) for {n_bad} (z, Mh) points "
                f"(threshold {CAPTURE_THRESHOLD}). mu/sigma there are biased toward "
                f"a truncated distribution; widen muv_grid further if this persists.",
                flush=True,
            )

        np.savetxt(mu_path, mus)
        np.savetxt(sigma_path, sigmas)
        np.savetxt(captured_path, captured_frac)

    comm.Barrier()
    if rank == 0:
        median_npz_path = os.path.join(directory, "PMuvMh_full_median.npz")
        if recompute or not os.path.exists(median_npz_path):
            median_row = np.median(posteriors, axis=0)
            dic_med = dict(zip(params, median_row))
            kwargs_med = build_kwargs(dic_med, fixed_Mknee, sigma_sfr_10_explicit, mass_dependent_sfr10)

            pdf_full = np.zeros((len(z_s), len(mh_eval), len(muv_grid)))
            captured_frac_full = np.zeros((len(z_s), len(mh_eval)))
            for index_z, z in enumerate(z_s):
                pdf_full[index_z] = p_muv_given_mh_sfr10(
                    muv_grid,
                    masses_hmf_per_z[index_z],
                    mh_eval,
                    z=z,
                    vect_func=vect_func,
                    bpass_read=bpass_read,
                    SFH_samp=SFR_samps[index_z],
                    **kwargs_med,
                )
                _, _, captured_frac_full[index_z] = gaussian_moments(muv_grid, pdf_full[index_z])

            if np.any(captured_frac_full < CAPTURE_THRESHOLD):
                worst = np.min(captured_frac_full)
                print(
                    f"[rank 0] WARNING median posterior: muv_grid only captures "
                    f"{worst:.3f} of p(Muv|Mh) for some (z, Mh) points "
                    f"(threshold {CAPTURE_THRESHOLD}); see captured_frac in the npz.",
                    flush=True,
                )

            np.savez(
                median_npz_path,
                z_s=np.asarray(z_s),
                mh_grid=mh_eval,
                muv_grid=muv_grid,
                pdf=pdf_full,
                median_params=median_row,
                captured_frac=captured_frac_full,
            )
            print(f"[rank 0] saved full p(Muv|Mh) for median posterior to PMuvMh_full_median.npz", flush=True)
        else:
            print(f"[rank 0] {median_npz_path} already exists, skipping (use --recompute to overwrite)", flush=True)