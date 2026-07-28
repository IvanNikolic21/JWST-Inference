"""
Quantify the SHMR-truncation bias across the full posterior, at fixed
Muv = -20, -19, -18, as a function of redshift.

For each posterior sample and redshift, computes phi_trunc (truncate_shmr=True,
what the fit actually uses) and phi_notrunc (truncate_shmr=False, the pre-fix
model) at the same seed (common random numbers -- the two share the same MC
draw, so their difference isn't swamped by independent MC noise). The bias is

    delta_log10_phi(sample, z, Muv) = log10(phi_trunc) - log10(phi_notrunc)

Aggregating delta_log10_phi over the posterior at each (z, Muv) gives the
median/16-84th-percentile bias -- i.e. how much of the truncated model's
predicted UVLF amplitude, at each redshift, is being suppressed relative to
the untruncated one, honestly propagated through the actual posterior rather
than one fiducial parameter point.

Usage (cluster):
    mpirun -n <N> python quantify_truncation_bias.py --directory_of_posteriors <dir> \\
        [--sample_start 0] [--n_samples 500]

Reads `post_equal_weights.dat` and `run_config.json` from <dir>. Optional:
    --muv_targets -20,-19,-18

Saves to <dir>/:
    truncation_bias.npz   -- delta_log10_phi[sample, z, muv], phi_trunc, phi_notrunc, z_s, muv_targets
    truncation_bias.pdf   -- median +/- 16-84th percentile bias vs z, one line per Muv
"""

import os
import json
import argparse
import numpy as np
import hmf as hmf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from mpi4py import MPI
from uvlf import bpass_loader, SFH_sampler

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory_of_posteriors", type=str, required=True)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--muv_targets", type=str, default="-20,-19,-18")
    args = parser.parse_args()
    directory = args.directory_of_posteriors
    muv_targets = np.array([float(m) for m in args.muv_targets.split(",")])

    with open(os.path.join(directory, "run_config.json")) as f:
        run_config = json.load(f)

    func_name              = run_config["uvlf_func"]
    param_names            = run_config["params"]
    sigma_sfr_10_explicit  = run_config["sigma_sfr_10_explicit"]
    sigma_uv               = run_config["sigma_uv"]
    mass_dependent_sigma_uv = run_config.get("mass_dependent_sigma_uv", False)
    fixed_Mknee             = run_config.get("fixed_Mknee", False)
    mass_dependent_sfr10   = run_config.get("mass_dependent_sfr10", False)

    if func_name == "UV_calc_numba_sfr10":
        from uvlf import UV_calc_numba_sfr10 as uvlf_func
    elif func_name == "UV_calc_numba":
        from uvlf import UV_calc_numba as uvlf_func
    else:
        raise ValueError(
            f"uvlf_func={func_name!r} does not expose truncate_shmr "
            f"(no truncation-bias diagnostic to run for it)."
        )

    z_s = [6.0, 8.0, 10.0, 11.0, 12.5, 14.0]

    posteriors_all = np.genfromtxt(os.path.join(directory, "post_equal_weights.dat"))
    posteriors = posteriors_all[args.sample_start: args.sample_start + args.n_samples]
    my_posteriors = posteriors[rank::size]

    if rank == 0:
        print(f"[truncation_bias] {len(posteriors)} samples total, {size} ranks, "
              f"{len(my_posteriors)} per rank (rank 0), z={z_s}, Muv={muv_targets}", flush=True)

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

    n_local = len(my_posteriors)
    n_z = len(z_s)
    n_muv = len(muv_targets)
    local_phi_trunc   = np.full((n_local, n_z, n_muv), np.nan)
    local_phi_notrunc = np.full((n_local, n_z, n_muv), np.nan)

    for i_local, post_sample in enumerate(my_posteriors):
        dic = dict(zip(param_names, post_sample))
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
        elif sigma_uv:
            kwargs["sigma_kuv"]               = dic.get("sigma_UV", 0.2)
            kwargs["mass_dependent_sigma_uv"] = mass_dependent_sigma_uv

        for i_z, z in enumerate(z_s):
            masses_hmf = np.log10(hmf_locs[i_z].m / cosmo.h)
            dndm = hmf_locs[i_z].dndlog10m * cosmo.h ** 3 * np.exp(-5e8 / (hmf_locs[i_z].m / cosmo.h))
            common = dict(
                z=z, vect_func=vect_func, SFH_samp=SFR_samps[i_z], seed=0, **kwargs,
            )
            if func_name == "UV_calc_numba":
                common["bpass_read"] = bpass_read
            local_phi_trunc[i_local, i_z]   = uvlf_func(muv_targets, masses_hmf, dndm,
                                                          truncate_shmr=True, **common)
            local_phi_notrunc[i_local, i_z] = uvlf_func(muv_targets, masses_hmf, dndm,
                                                          truncate_shmr=False, **common)

        if rank == 0 and (i_local + 1) % 10 == 0:
            print(f"  rank 0: {i_local + 1}/{n_local} done", flush=True)

    gathered_trunc   = comm.gather(local_phi_trunc, root=0)
    gathered_notrunc = comm.gather(local_phi_notrunc, root=0)

    if rank == 0:
        phi_trunc   = np.concatenate(gathered_trunc, axis=0)   # (Nsample, Nz, Nmuv)
        phi_notrunc = np.concatenate(gathered_notrunc, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            delta_log10_phi = np.log10(phi_trunc) - np.log10(phi_notrunc)

        out_path = os.path.join(directory, "truncation_bias.npz")
        np.savez(
            out_path,
            phi_trunc=phi_trunc, phi_notrunc=phi_notrunc,
            delta_log10_phi=delta_log10_phi,
            z_s=np.array(z_s), muv_targets=muv_targets,
        )
        print(f"Saved: {out_path}", flush=True)

        # ── plot: median +/- 16-84th percentile bias vs z, one line per Muv ──
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = plt.cm.viridis(np.linspace(0, 0.85, n_muv))
        for i_muv, muv in enumerate(muv_targets):
            d = delta_log10_phi[:, :, i_muv]  # (Nsample, Nz)
            med = np.nanmedian(d, axis=0)
            lo  = np.nanpercentile(d, 16, axis=0)
            hi  = np.nanpercentile(d, 84, axis=0)
            ax.plot(z_s, med, color=colors[i_muv], lw=2.5, label=rf"$M_{{\rm UV}}={muv:g}$")
            ax.fill_between(z_s, lo, hi, color=colors[i_muv], alpha=0.25, lw=0)

        ax.axhline(0.0, color="k", ls=":", lw=1)
        ax.set_xlabel(r"$z$", fontsize=16)
        ax.set_ylabel(r"$\log_{10}\phi_{\rm trunc} - \log_{10}\phi_{\rm no\,trunc}$", fontsize=14)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=12, frameon=False)
        plt.tight_layout()

        plot_path = os.path.join(directory, "truncation_bias.pdf")
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Saved: {plot_path}", flush=True)
