"""
Muv-selected effective galaxy bias b_eff(z; Muv_cut) from the posterior --
for comparison to the (mostly UV-magnitude-selected) literature, instead of
the stellar-mass-selected version in compute_effective_bias.py.

Unlike a stellar-mass cut, p(Muv|Mh) has no closed form (that's why
pmuv_given_mh.py fits it with curve_fit instead of a Gaussian-variance
shortcut), so N_c(Mh) here comes from numerically integrating the full
Monte Carlo p(Muv|Mh) pdf (uvlf.p_muv_given_mh_sfr10):

    N_c(Mh; z, Muv_cut) = int_{-inf}^{Muv_cut} p(Muv | Mh, z) dMuv

(more negative Muv = brighter, so this is "brighter than Muv_cut"). Then

    b_eff(z; Muv_cut) = [ sum_Mh dn/dlog10Mh(Mh,z) N_c(Mh) b_h(Mh,z) ]
                       / [ sum_Mh dn/dlog10Mh(Mh,z) N_c(Mh) ]

dn/dlog10Mh and b_h(Mh,z) (Tinker10, same bias_model as the ACF fit) are
read directly off a LikelihoodAngBase's angular_gal (mcmc.py) -- built once
per redshift, since neither depends on the posterior sample, only N_c does.
The grid is uniform in log10(Mh) (halomod's dlog10m), so a plain sum is a
correct Riemann-sum quadrature -- no need for per-point bin-width weights.

No reionization-feedback quenching factor (the ad hoc exp(-5e8/Mh) used in
the UVLF scripts) is applied here: the Muv cut itself already suppresses
the low-mass tail, and this is meant to mirror how b_eff is normally
defined in clustering work, not the UVLF fitting pipeline.

Only supports sigma_sfr_10_explicit=True runs (p_muv_given_mh_sfr10 is the
SFR10/burstiness forward model; there's no equivalent helper yet for the
plain sigma_uv model).

Usage (cluster):
    mpirun -n <N> python compute_effective_bias_muv.py --directory_of_posteriors <dir> \\
        [--z_list 6,7,8,9,10] [--muv_targets -20,-19,-18] [--sample_start 0] [--n_samples 500]

Saves to <dir>/:
    effective_bias_muv.npz -- b_eff[sample, z, muv], z_list, muv_targets
    effective_bias_muv.pdf -- median +/- 16-84th percentile b_eff vs z, one line per Muv
"""

import os
import json
import argparse
import numpy as np
import hmf as hmf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from astropy.cosmology import Planck18 as cosmo
from mpi4py import MPI
from uvlf import bpass_loader, SFH_sampler, p_muv_given_mh_sfr10
from mcmc import LikelihoodAngBase

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory_of_posteriors", type=str, required=True)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--z_list", type=str, default="6,7,8,9,10")
    parser.add_argument("--muv_targets", type=str, default="-20,-19,-18")
    parser.add_argument("--exact_specs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    directory = args.directory_of_posteriors
    z_list = [float(z) for z in args.z_list.split(",")]
    muv_targets = np.array([float(m) for m in args.muv_targets.split(",")])
    muv_grid_fine = np.linspace(-26, -10, 160)

    with open(os.path.join(directory, "run_config.json")) as f:
        run_config = json.load(f)
    param_names            = run_config["params"]
    sigma_sfr_10_explicit  = run_config["sigma_sfr_10_explicit"]
    fixed_Mknee            = run_config.get("fixed_Mknee", False)
    mass_dependent_sfr10   = run_config.get("mass_dependent_sfr10", False)

    if not sigma_sfr_10_explicit:
        raise ValueError(
            "p_muv_given_mh_sfr10 needs a sigma_sfr_10_explicit=True run "
            "(it's the SFR10/burstiness forward model); this run_config has "
            f"sigma_sfr_10_explicit={sigma_sfr_10_explicit}."
        )

    posteriors_all = np.genfromtxt(os.path.join(directory, "post_equal_weights.dat"))
    posteriors = posteriors_all[args.sample_start: args.sample_start + args.n_samples]
    my_posteriors = posteriors[rank::size]

    if rank == 0:
        print(f"[effective_bias_muv] {len(posteriors)} samples total, {size} ranks, "
              f"{len(my_posteriors)} per rank (rank 0), z={z_list}, Muv={muv_targets}", flush=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir == "/groups/astro/ivannik/programs/JWST-Inference":
        bpass_read = bpass_loader(
            filename="/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00.",
        )
    else:
        bpass_read = bpass_loader()
    vect_func = np.vectorize(bpass_read.get_UV_sfr10)

    # ── per-z setup that doesn't depend on the posterior sample ──────────────
    # broad tabulation grid for the smooth SHMR/SFMS/SFR10/BPASS relations
    # (same convention as the other scripts), the ACF halo model's own
    # (Mh, dn/dlog10Mh, halo_bias) grid, and an SFH sampler.
    hmf_locs  = {z: hmf.MassFunction(z=z, Mmin=5, Mmax=19, dlog10m=0.05, hmf_model="Tinker08") for z in z_list}
    SFR_samps = {z: SFH_sampler(z=z) for z in z_list}
    ang_bases = {z: LikelihoodAngBase(params=param_names, z=z, exact_specs=args.exact_specs,
                                       fixed_Mknee=fixed_Mknee) for z in z_list}

    mh_eval_h    = {}  # log10(Mh/Msun), ACF halo model's own grid, physical units
    dndlog10m_ac = {}  # dn/dlog10Mh on that grid
    halo_bias_ac = {}
    for z in z_list:
        ag = ang_bases[z].angular_gal
        mh_eval_h[z]    = np.log10(np.asarray(ag.m) / cosmo.h)
        dndlog10m_ac[z] = np.asarray(ag.dndlog10m)
        halo_bias_ac[z] = np.asarray(ag.halo_bias)

    n_local = len(my_posteriors)
    n_z = len(z_list)
    n_muv = len(muv_targets)
    local_b_eff = np.full((n_local, n_z, n_muv), np.nan)

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
            sigma_sfr10          = dic.get("sigma_sfr_10", dic.get("sigma_SFR_10", 0.2)),
            mass_dependent_sfr10 = mass_dependent_sfr10,
        )

        for i_z, z in enumerate(z_list):
            masses_hmf = np.log10(hmf_locs[z].m / cosmo.h)
            pdf = p_muv_given_mh_sfr10(
                muv_grid_fine, masses_hmf, mh_eval_h[z],
                z=z, vect_func=vect_func, SFH_samp=SFR_samps[z], seed=0,
                **kwargs,
            )  # (Nmh_eval, Nmuv_fine)
            cdf = cumulative_trapezoid(pdf, muv_grid_fine, axis=1, initial=0.0)

            for i_muv, muv_cut in enumerate(muv_targets):
                idx = np.clip(np.searchsorted(muv_grid_fine, muv_cut), 0, len(muv_grid_fine) - 1)
                Nc = cdf[:, idx]  # (Nmh_eval,) -- P(Muv < muv_cut | Mh, z)
                weight = dndlog10m_ac[z] * Nc
                local_b_eff[i_local, i_z, i_muv] = (
                    np.sum(weight * halo_bias_ac[z]) / np.sum(weight)
                )

        if rank == 0 and (i_local + 1) % 10 == 0:
            print(f"  rank 0: {i_local + 1}/{n_local} done", flush=True)

    gathered = comm.gather(local_b_eff, root=0)

    if rank == 0:
        b_eff = np.concatenate(gathered, axis=0)  # (Nsample, Nz, Nmuv)

        out_path = os.path.join(directory, "effective_bias_muv.npz")
        np.savez(
            out_path, b_eff=b_eff, z_list=np.array(z_list), muv_targets=muv_targets,
        )
        print(f"Saved: {out_path}", flush=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        colors = plt.cm.viridis(np.linspace(0, 0.85, n_muv))
        for i_muv, muv in enumerate(muv_targets):
            d = b_eff[:, :, i_muv]  # (Nsample, Nz)
            med = np.nanmedian(d, axis=0)
            lo  = np.nanpercentile(d, 16, axis=0)
            hi  = np.nanpercentile(d, 84, axis=0)
            ax.plot(z_list, med, color=colors[i_muv], lw=2.5, label=rf"$M_{{\rm UV}}<{muv:g}$")
            ax.fill_between(z_list, lo, hi, color=colors[i_muv], alpha=0.25, lw=0)

        ax.set_xlabel(r"$z$", fontsize=16)
        ax.set_ylabel(r"$b_{\rm eff}$", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=12, frameon=False)
        plt.tight_layout()

        plot_path = os.path.join(directory, "effective_bias_muv.pdf")
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Saved: {plot_path}", flush=True)
