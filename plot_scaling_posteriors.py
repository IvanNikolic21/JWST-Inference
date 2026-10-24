"""Figs. shmr_post and sfms_post (SHMR_posterior_with_cuts-1.pdf,
SFMS_posterior_with_cuts_conditional.pdf): posterior predictive SHMR and SFMS
of the joint run, against simulated galaxies.

Left panels: 68% / 95% intervals of log M* at fixed M_h (log SFR_100 at fixed
M* and z = 10), including parameter uncertainty and intrinsic scatter. They are
percentiles of the exact posterior mixture of the model distributions:
  SHMR  N(log M*_mean(M_h), sigma_SHMR) truncated (renormalised) at f_b M_h,
        mean from ms_mh_flattening, as in the likelihood (truncate_shmr=True)
  SFMS  N(log SFMS(M*, t*, z), sigma_SFR_variable(M*)).
Right panels: the same mixtures at a few fixed masses, peak-normalised and
smoothed by 0.05 dex (samples with very small scatter otherwise give spikes).

Simulation points come from an npz written on the cluster with --build_sims
(loaders and paths of the prior config). Keys <SIM>_mh / _ms / _sfr / _z.

  python plot_scaling_posteriors.py --build_sims prior_configs/fiducial_public.json --sims_out scaling_sims.npz
  python plot_scaling_posteriors.py <main_joint>/ --sims scaling_sims.npz [--out_dir .]
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtr

from uvlf import ms_mh_flattening, SFMS, sigma_SFR_variable

M_KNEE_FIXED = 2e12
FB = cosmo.Ob0 / cosmo.Om0
Z_SFMS = 10.0
C_68, C_95 = "#e41a1c", "#f4a6a6"      # UVLF + ACF colour of the other figures
# simulation, snapshot redshift, colour (FIRE has no z = 10 snapshot: z = 9 is the closest)
SHMR_SIMS = [("FIRE", 9.0, "plum"), ("FirstLight", 10.0, "lightskyblue")]
SFMS_SIMS = [("FLARES", 10.0, "seagreen"), ("ASTRID", 10.0, "royalblue")]
SIM_LABEL = {"FIRE": "FIRE", "FirstLight": "First Light", "FLARES": "FLARES", "ASTRID": "ASTRID"}
SHMR_COND = [9.5, 10.0, 10.5, 11.0]
SFMS_COND = [7.5, 8.0, 8.5, 9.0]


def build_sims(config, out):
    from build_priors import load_config
    from prior_pipeline.simulations import load_simulation
    cfg = load_config(config)
    save = {}
    for name, z, _ in SHMR_SIMS + SFMS_SIMS:
        sim = load_simulation(name, cfg["simulations"][name])
        snap = min(sim.snapshots, key=lambda s: abs(s.z - z))
        save[f"{name}_z"] = snap.z
        save[f"{name}_ms"], save[f"{name}_sfr"] = snap.ms, snap.sfr
        if snap.mh is not None:
            save[f"{name}_mh"] = snap.mh
        print(f"{name}: z = {snap.z}, {len(snap.ms)} galaxies")
    np.savez(out, **save)
    print("saved", out)


def mixture(x, y, mu, sig, upper=None):
    """CDF and pdf on y (Ny,) of the posterior mixture at each x: mu, sig and
    upper are (Nsamp, Nx); returns (Nx, Ny) arrays."""
    cdf, pdf = np.empty((len(x), len(y))), np.empty((len(x), len(y)))
    for i in range(len(x)):
        t = (y[None, :] - mu[:, i, None]) / sig[:, i, None]
        c = ndtr(t)
        p = np.exp(-0.5 * t ** 2) / (np.sqrt(2 * np.pi) * sig[:, i, None])
        if upper is not None:
            norm = ndtr((upper[:, i] - mu[:, i]) / sig[:, i])[:, None]
            above = y[None, :] > upper[:, i, None]
            c = np.where(above, 1.0, c / norm)
            p = np.where(above, 0.0, p / norm)
        cdf[i], pdf[i] = c.mean(0), p.mean(0)
    return cdf, pdf


def percentiles(y, cdf, qs=(0.025, 0.16, 0.84, 0.975)):
    return {q: np.array([np.interp(q, c, y) for c in cdf]) for q in qs}


def panel(fig, x, y, cdf, pdf, cond, sims, data, xlabel, ylabel, plabel, xlim, ylim, cmap, text=None):
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.08)
    ax, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    q = percentiles(y, cdf)
    ax.fill_between(x, q[0.025], q[0.975], color=C_95, alpha=0.6, lw=0)
    ax.fill_between(x, q[0.16], q[0.84], color=C_68, alpha=0.5, lw=0)
    for k in q:
        ax.plot(x, q[k], color="k", lw=1.5)
    for name, _, color in sims:
        key = "mh" if sims is SHMR_SIMS else "ms"
        if f"{name}_ms" not in data:
            print(f"  {name} not in the simulation file, skipped")
            continue
        xs = np.log10(data[f"{name}_{key}"])
        ys = np.log10(data[f"{name}_ms" if key == "mh" else f"{name}_sfr"])
        ok = np.isfinite(xs) & np.isfinite(ys)
        ax.scatter(xs[ok], ys[ok], color=color, alpha=0.4, s=18, edgecolor="none",
                   label=rf"{SIM_LABEL[name]}, $z={float(data[f'{name}_z']):.0f}$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=10, loc="lower right")
    if text:
        ax.text(0.05, 0.92, text, transform=ax.transAxes, fontsize=15, va="top")

    colors = plt.get_cmap(cmap)(np.linspace(0.3, 0.9, len(cond)))
    for xc, col in zip(cond, colors):
        p = gaussian_filter1d(pdf[np.argmin(np.abs(x - xc))], 0.05 / (y[1] - y[0]))
        p = p / p.max()
        axc.plot(p, y, color=col, lw=2)
        axc.text(1.04, y[np.argmax(p)], f"{xc:g}", color=col, fontsize=10, va="center")
    axc.set_ylim(ylim)
    axc.set_xlim(0, 1.3)
    axc.set_xticks([0, 1])
    axc.set_yticklabels([])
    axc.tick_params(labelsize=14)
    axc.grid(alpha=0.3)
    axc.set_xlabel(plabel, fontsize=15)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--sims", default=None, help="simulation points npz from --build_sims")
    ap.add_argument("--build_sims", default=None, help="prior config; write --sims_out and exit")
    ap.add_argument("--sims_out", default="scaling_sims.npz")
    ap.add_argument("--n_samples", type=int, default=1000, help="posterior samples used for the mixtures")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if args.build_sims:
        build_sims(args.build_sims, args.sims_out)
        return

    with open(os.path.join(args.run_dir, "run_config.json")) as f:
        names = json.load(f)["params"]
    post = np.genfromtxt(os.path.join(args.run_dir, "post_equal_weights.dat"))[:, :len(names)]
    rng = np.random.default_rng(0)
    post = post[rng.choice(len(post), min(args.n_samples, len(post)), replace=False)]
    par = {n: post[:, i] for i, n in enumerate(names)}
    data = dict(np.load(args.sims)) if args.sims else {}
    out_dir = args.out_dir or args.run_dir

    # SHMR
    lmh = np.linspace(9.0, 12.0, 151)
    lms_y = np.arange(3.0, 12.5, 0.01)
    mu = np.array([np.log10(ms_mh_flattening(10 ** lmh, cosmo, fstar_norm=10 ** f, alpha_star_low=a,
                                             M_knee=M_KNEE_FIXED))
                   for f, a in zip(par["fstar_norm"], par["alpha_star_low"])])
    sig = np.repeat(par["sigma_SHMR"][:, None], len(lmh), axis=1)
    cdf, pdf = mixture(lmh, lms_y, mu, sig, upper=np.log10(FB) + lmh[None, :] + 0 * mu)
    fig = plt.figure(figsize=(8, 4))
    panel(fig, lmh, lms_y, cdf, pdf, SHMR_COND, SHMR_SIMS, data,
          r"$\log M_{\rm h}\ [{\rm M_\odot}]$", r"$\log M_\ast\ [{\rm M_\odot}]$",
          r"$p(\log M_\ast\mid\log M_{\rm h})$", (9, 12), (5, 11), "Reds")
    out = os.path.join(out_dir, "SHMR_posterior_with_cuts-1.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)

    # SFMS at z = 10
    lms = np.linspace(6.0, 10.0, 161)
    lsfr_y = np.arange(-5.0, 4.0, 0.01)
    mu = np.array([np.log10(SFMS(10 ** lms, SFR_norm=t, z=Z_SFMS)) for t in par["t_star"]])
    sig = np.array([sigma_SFR_variable(10 ** lms, norm=s, a_sig_SFR=a)
                    for s, a in zip(par["sigma_SFMS_norm"], par["a_sig_SFR"])])
    cdf, pdf = mixture(lms, lsfr_y, mu, sig)
    fig = plt.figure(figsize=(8, 4))
    panel(fig, lms, lsfr_y, cdf, pdf, SFMS_COND, SFMS_SIMS, data,
          r"$\log M_\ast\ [{\rm M_\odot}]$", r"$\log {\rm SFR}_{100}\ [{\rm M_\odot\,yr^{-1}}]$",
          r"$p(\log {\rm SFR}_{100}\mid\log M_\ast)$", (6, 10), (-3, 2), "magma", text=rf"$z={Z_SFMS:.0f}$")
    out = os.path.join(out_dir, "SFMS_posterior_with_cuts_conditional.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
