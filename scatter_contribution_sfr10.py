"""
Fractional contribution of each scatter source to sigma_UV at z=10, Muv=-20.

Sources decomposed:
    sigma_SFR10  — stochasticity of SFR over 10 Myr
    sigma_SFMS   — scatter on the star-forming main sequence
    sigma_SHMR   — scatter on the stellar-to-halo-mass relation

Usage (cluster):
    mpirun -n <N> python scatter_contribution_sfr10.py --directory_of_posteriors <dir>

Saves:
    <dir>/scatter_contribution_sfr10.npz   — sigmas + fracs arrays
    <dir>/fractional_sigma_sfr10.pdf       — CDF plot
"""

import os
import json
import argparse
import numpy as np
import hmf as hmf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.cosmology import Planck18 as cosmo
from mpi4py import MPI
from uvlf import bpass_loader, SFH_sampler, p_muv_given_mh_sfr10

# ── constants ──────────────────────────────────────────────────────────────────
TARGET_Z   = 10
TARGET_MUV = -20.0
NEAR_ZERO  = 0.01

SCENARIOS = [
    ("full",     r"full",                      "k",       {}),
    ("no_sfr10", r"$\sigma_{\rm SFR_{10}}$",  "#66c2a5", {"sigma_sfr10": NEAR_ZERO}),
    ("no_SFMS",  r"$\sigma_{\rm SFMS}$",       "#fc8d62", {"sigma_SFMS_norm": NEAR_ZERO,
                                                            "a_sig_SFR": 0.0}),
    ("no_SHMR",  r"$\sigma_{\rm SHMR}$",       "#8da0cb", {"sigma_SHMR": NEAR_ZERO}),
]
SCENARIO_NAMES = [s[0] for s in SCENARIOS]

muv_bins = np.linspace(-26, -10, 160)
mh_eval  = np.arange(8.5, 12.0 + 1e-9, 0.1)


def _gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_sigma(p_row, muv_grid):
    if not np.any(p_row > 0):
        return np.nan, np.nan
    amp0 = p_row.max()
    mu0  = muv_grid[np.argmax(p_row)]
    try:
        popt, _ = curve_fit(
            _gauss, muv_grid, p_row,
            p0=[amp0, mu0, 2.0],
            bounds=([0.0, muv_grid.min(), 1e-3],
                    [np.inf, muv_grid.max(), 50.0]),
            maxfev=5000,
        )
        return popt[1], popt[2]
    except RuntimeError:
        return mu0, np.nan


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory_of_posteriors", type=str, required=True)
    parser.add_argument("--sample_start", type=int, default=2000,
                        help="First posterior sample index to use")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of posterior samples to use")
    args = parser.parse_args()
    directory = args.directory_of_posteriors

    with open(os.path.join(directory, "run_config.json")) as f:
        run_config = json.load(f)
    param_names = run_config["params"]

    posteriors_all = np.genfromtxt(os.path.join(directory, "post_equal_weights.dat"))
    posteriors = posteriors_all[args.sample_start : args.sample_start + args.n_samples]
    my_posteriors = posteriors[rank::size]

    if rank == 0:
        print(f"[scatter_sfr10] {len(posteriors)} samples total, "
              f"{size} ranks, {len(my_posteriors)} per rank (rank 0)", flush=True)

    # ── per-rank setup ─────────────────────────────────────────────────────────
    hmf_loc    = hmf.MassFunction(z=TARGET_Z, Mmin=5, Mmax=19, dlog10m=0.05,
                                  hmf_model="Tinker08")
    masses_hmf = np.log10(hmf_loc.m / cosmo.h)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir == "/groups/astro/ivannik/programs/JWST-Inference":
        bpass_read = bpass_loader(
            filename="/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00.",
        )
    else:
        bpass_read = bpass_loader()

    vect_func = np.vectorize(bpass_read.get_UV_sfr10)
    SFR_samp  = SFH_sampler(z=TARGET_Z)

    # ── main loop (over this rank's samples) ──────────────────────────────────
    n_local = len(my_posteriors)
    local_sigmas = {name: np.full(n_local, np.nan) for name in SCENARIO_NAMES}

    for i_local, post_sample in enumerate(my_posteriors):
        dic = dict(zip(param_names, post_sample))
        base_kwargs = dict(
            f_star_norm     = 10 ** dic["fstar_norm"],
            sigma_SHMR      = dic["sigma_SHMR"],
            t_star          = dic["t_star"],
            alpha_star      = dic["alpha_star_low"],
            sigma_SFMS_norm = dic["sigma_SFMS_norm"],
            a_sig_SFR       = dic["a_sig_SFR"],
            M_knee          = 10 ** dic["M_knee"],
            sigma_sfr10     = dic.get("sigma_sfr_10", dic.get("sigma_SFR_10", 0.2)),
            z               = TARGET_Z,
            vect_func       = vect_func,
            SFH_samp        = SFR_samp,
        )

        for name, _label, _color, overrides in SCENARIOS:
            p = p_muv_given_mh_sfr10(muv_bins, masses_hmf, mh_eval,
                                     **{**base_kwargs, **overrides})
            mean_muv  = np.empty(len(mh_eval))
            sigma_muv = np.empty(len(mh_eval))
            for i_mh in range(len(mh_eval)):
                mean_muv[i_mh], sigma_muv[i_mh] = _fit_sigma(p[i_mh], muv_bins)

            valid = np.isfinite(mean_muv)
            if np.any(valid):
                i_best = np.argmin(np.abs(mean_muv[valid] - TARGET_MUV))
                local_sigmas[name][i_local] = sigma_muv[valid][i_best]

        if rank == 0 and (i_local + 1) % 10 == 0:
            print(f"  rank 0: {i_local + 1}/{n_local} done", flush=True)

    # ── gather across ranks ────────────────────────────────────────────────────
    gathered = {}
    for name in SCENARIO_NAMES:
        chunks = comm.gather(local_sigmas[name], root=0)
        if rank == 0:
            gathered[name] = np.concatenate(chunks)

    if rank == 0:
        sigmas = gathered
        fracs  = {
            name: 1.0 - sigmas[name] / sigmas["full"]
            for name in SCENARIO_NAMES if name != "full"
        }

        # save raw arrays
        out_path = os.path.join(directory, "scatter_contribution_sfr10.npz")
        np.savez(
            out_path,
            **{f"sigma_{name}": sigmas[name] for name in SCENARIO_NAMES},
            **{f"frac_{name}":  fracs[name]  for name in fracs},
            target_z=TARGET_Z,
            target_muv=TARGET_MUV,
        )
        print(f"Saved: {out_path}", flush=True)

        # plot
        fig, ax = plt.subplots(figsize=(6, 5))
        bins = np.linspace(0.0, 1.0, 100)
        for name, label, color, _ in SCENARIOS[1:]:
            ax.hist(fracs[name], bins=bins, density=True, cumulative=True,
                    histtype="step", color=color, lw=3, label=label)

        ax.set_xlim(0, 1)
        ax.set_xlabel(r"Fraction of $\sigma_{\rm UV}$", fontsize=18)
        ax.set_ylabel(r"CDF", fontsize=18)
        ax.tick_params(labelsize=18)
        ax.legend(fontsize=14, frameon=False)
        ax.text(0.36, 0.34, "Fractional contribution of:", fontsize=14)
        ax.text(0.08, 0.90, rf"z={TARGET_Z}, M$_{{\rm UV}}$ = {TARGET_MUV}", fontsize=14)
        plt.tight_layout()

        plot_path = os.path.join(directory, "fractional_sigma_sfr10.pdf")
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Saved: {plot_path}", flush=True)
