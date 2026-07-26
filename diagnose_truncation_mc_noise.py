"""
Diagnostic: is the "kinky" look of individual truncated-SHMR UVLF samples a
Monte Carlo sample-starvation artifact, rather than a real feature?

setup_sample_probabilities_fast (uvlf.py) always draws mstar_samples with
seed=0 and Nmstar=10_000 -- the same fixed draw is reused for every posterior
sample and every redshift in both mcmc.py and uvlf_a_post.py.
_gauss_mstar_mh_truncated then masks out any of those fixed points that fall
above the baryon-fraction ceiling b(Mh) = log10(Ob0/Om0) + Mh. Where the
ceiling sits close to the median SHMR -- i.e. the low-mass halos needed to
produce bright galaxies at high z -- a large fraction of the fixed 10,000
points get masked per row, shrinking the *effective* sample size right where
high-z/bright-end predictions come from. Because the draw is frozen (seed=0
always), this isn't noise that averages out across a run -- it's the same
distortion baked into every likelihood evaluation.

This script recomputes UV_calc_numba for one fiducial parameter set at a
handful of redshifts under four configurations:
    baseline (trunc) : seed=0, Nmstar=10_000   (what mcmc.py / uvlf_a_post.py use)
    reseed (trunc)   : seed=1, Nmstar=10_000   (same N, different fixed draw)
    hires (trunc)    : seed=0, Nmstar=100_000  (same draw family, 10x denser)
    no_truncation    : seed=0, Nmstar=10_000, truncate_shmr=False (pre-fix model,
                       included as a reference for how far the truncated
                       curves -- and their MC noise -- sit from the old one)

If "reseed (trunc)" and "hires (trunc)" disagree visibly with "baseline
(trunc)" (curves shift / kinks move or shrink), that confirms the effect is
MC sample-starvation noise from the truncation mask, and the fix is to raise
Nmstar in UV_calc_numba / UV_calc_numba_sfr10 (uvlf.py), not to touch the
seed. Comparing all three against "no_truncation" also shows how much of the
baseline (trunc) vs. no_truncation gap is a genuine physical effect of
enforcing the baryon-fraction ceiling, versus just MC noise.

No posterior file or cluster access needed -- runs on one fiducial parameter
set. Requires the JWST_inference conda env (numba, hmf, astropy).

Usage:
    python diagnose_truncation_mc_noise.py [--sigma_SHMR 0.3] \\
        [--seeds 0,1] [--nmstar_hires 100000] [--truncate_shmr / --no-truncate_shmr]

Saves: truncation_mc_noise_diagnostic.pdf (next to this script)
"""

import os
import argparse
import numpy as np
import hmf as hmf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from uvlf import (
    bpass_loader, SFH_sampler, ms_mh_flattening, SFMS, metalicity_from_FMR,
    DeltaZ_z, Muv_Luv, sigma_SFR_variable, uvlf_fast_einsum,
)


def uvlf_with_custom_N(
    Muv, masses_hmf, dndm, *, f_star_norm, alpha_star, sigma_SHMR,
    sigma_SFMS_norm, t_star, a_sig_SFR, z, vect_func, SFH_samp, M_knee,
    sigma_kuv, Nsfr, Nmstar, Nmh, seed, truncate_shmr,
):
    """Same body as uvlf.UV_calc_numba, but with Nsfr/Nmstar/Nmh exposed
    (UV_calc_numba hardcodes them to 10_000 internally)."""
    msss = ms_mh_flattening(10 ** masses_hmf, cosmo, alpha_star_low=alpha_star,
                            fstar_norm=f_star_norm, M_knee=M_knee)
    sfrs = SFMS(msss, SFR_norm=t_star, z=z)
    Zs = metalicity_from_FMR(msss, sfrs)
    Zs += DeltaZ_z(z)
    F_UV = vect_func(Zs, msss, sfrs, z=z, SFH_samp=SFH_samp, sigma_uv=sigma_kuv)
    muvs = Muv_Luv(F_UV * 3.846e33)
    sigma_kuv_var = sigma_kuv * np.ones(np.shape(msss))
    sigma_SFMS_var = sigma_SFR_variable(msss, norm=sigma_SFMS_norm, a_sig_SFR=a_sig_SFR)
    return uvlf_fast_einsum(
        Muv, sigma_kuv_var, muvs, np.log10(sfrs), np.log10(msss),
        sigma_SFMS_var, masses_hmf, sigma_SHMR, dndm,
        Nsfr=Nsfr, Nmstar=Nmstar, Nmh=Nmh, seed=seed, truncate_shmr=truncate_shmr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_SHMR", type=float, default=0.3)
    parser.add_argument("--fstar_norm", type=float, default=1.0)
    parser.add_argument("--alpha_star_low", type=float, default=0.5)
    parser.add_argument("--M_knee", type=float, default=2.6e11)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--nmstar_hires", type=int, default=100_000)
    parser.add_argument("--truncate_shmr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bpass_filename", type=str, default=None,
                        help="Override path to BPASS spectra files, e.g. "
                             "/path/to/spectra-bin-imf135_300.a+00. "
                             "(same prefix convention as bpass_loader)")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    z_s    = [6.0, 10.0, 14.0]
    muvs_o = np.linspace(-24, -16, 40)

    hmf_locs = [
        hmf.MassFunction(z=z, Mmin=5, Mmax=19, dlog10m=0.05, hmf_model="Tinker08")
        for z in z_s
    ]
    SFR_samps = [SFH_sampler(z=z) for z in z_s]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.bpass_filename:
        bpass_read = bpass_loader(filename=args.bpass_filename)
    elif script_dir == "/groups/astro/ivannik/programs/JWST-Inference":
        bpass_read = bpass_loader(
            filename="/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00.",
        )
    else:
        bpass_read = bpass_loader()
    vect_func = np.vectorize(bpass_read.get_UV)

    fiducial = dict(
        f_star_norm=args.fstar_norm, alpha_star=args.alpha_star_low,
        sigma_SHMR=args.sigma_SHMR, sigma_SFMS_norm=0.0, t_star=0.5,
        a_sig_SFR=-0.11654893, M_knee=args.M_knee, sigma_kuv=0.2,
    )

    # (label, seed, Nmstar, truncate_shmr) -- "no_truncation" is the
    # pre-truncation-fix behaviour, included as a reference to see how far
    # the truncated curves (and their MC noise) sit from the old model.
    configs = [("baseline (trunc)", 0, 10_000, args.truncate_shmr)]
    configs += [(f"reseed(seed={s}) (trunc)", s, 10_000, args.truncate_shmr) for s in seeds if s != 0]
    configs += [("hires (trunc)", 0, args.nmstar_hires, args.truncate_shmr)]
    configs += [("no_truncation (seed=0)", 0, 10_000, False)]

    results = {label: [] for label, _, _, _ in configs}
    for i_z, z in enumerate(z_s):
        masses_hmf = np.log10(hmf_locs[i_z].m / cosmo.h)
        dndm = hmf_locs[i_z].dndlog10m * cosmo.h ** 3 * np.exp(-5e8 / (hmf_locs[i_z].m / cosmo.h))
        print(f"  z={z}: masses_hmf range [{masses_hmf.min():.2f}, {masses_hmf.max():.2f}], "
              f"dndm nan={np.isnan(dndm).sum()} finite_nonzero={(np.isfinite(dndm) & (dndm > 0)).sum()}/{dndm.size}",
              flush=True)
        for label, seed, nmstar, truncate_shmr in configs:
            phi = uvlf_with_custom_N(
                muvs_o, masses_hmf, dndm, z=z, vect_func=vect_func,
                SFH_samp=SFR_samps[i_z], Nsfr=10_000, Nmstar=nmstar, Nmh=10_000,
                seed=seed, truncate_shmr=truncate_shmr, **fiducial,
            )
            results[label].append(phi)
            n_nan = np.isnan(phi).sum()
            finite = phi[np.isfinite(phi)]
            rng_str = f"[{finite.min():.3e}, {finite.max():.3e}]" if finite.size else "n/a"
            print(f"  z={z}, {label}: nan={n_nan}/{phi.size}, finite_range={rng_str}", flush=True)

    fig, axes = plt.subplots(1, len(z_s), figsize=(4.5 * len(z_s), 4), sharey=True)
    colors = {c[0]: col for c, col in zip(configs, plt.cm.viridis(np.linspace(0, 0.85, len(configs) - 1)))}
    colors["no_truncation (seed=0)"] = "0.3"
    for i_z, z in enumerate(z_s):
        ax = axes[i_z]
        for label, _, _, _ in configs:
            phi = results[label][i_z]
            style = "--" if label == "no_truncation (seed=0)" else "-"
            ax.plot(muvs_o, np.log10(np.clip(phi, 1e-30, None)),
                    label=label, color=colors[label], lw=1.5, linestyle=style)
        ax.set_title(f"z={z}", fontsize=12)
        ax.set_xlabel(r"$M_{\rm UV}$", fontsize=11)
        if i_z == 0:
            ax.set_ylabel(r"$\log_{10}\phi$", fontsize=11)
            ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    out_path = os.path.join(script_dir, "truncation_mc_noise_diagnostic.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}", flush=True)
