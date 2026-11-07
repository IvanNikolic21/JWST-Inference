"""Figs. shmr_post and sfms_post (SHMR_posterior_with_cuts-1.pdf,
SFMS_posterior_with_cuts_conditional.pdf): posterior predictive SHMR and SFMS
of the joint run, against simulated galaxies. With --prior, the same figures
for the prior (Figs. SHMR_prior and sfms_prior: SHMR_prior.pdf, SFMS_prior.pdf,
grey bands), drawn through mcmc.py's prior transform: each parameter a normal
with the mean and effective variance of the prior file, truncated to its
sampling limits (the covariance is diagonal, so the parameters are
independent). The uniforms are a scrambled Sobol sequence (use a power of 2
for --n_samples), which keeps the right panels smooth: the prior reaches
sigma_SHMR ~ 0, and plain random draws leave visible noise there.

Left panels: 68% / 95% intervals of log M* at fixed M_h (log SFR_100 at fixed
M* and z = 10), including parameter uncertainty and intrinsic scatter. They are
percentiles of the exact posterior mixture of the model distributions:
  SHMR  N(log M*_mean(M_h), sigma_SHMR) truncated (renormalised) at f_b M_h,
        mean from ms_mh_flattening, as in the likelihood (truncate_shmr=True)
  SFMS  N(log SFMS(M*, t*, z), sigma_SFR_variable(M*)).
Right panels: the same mixtures at a few fixed masses, peak-normalised and
smoothed by 0.05 dex (samples with very small scatter otherwise give spikes).
The densities are derivatives of the mixture CDF, so a sample with tiny scatter
adds its unit mass, not a spike. ms_mh_flattening caps the mean at f_b M_h,
which 9-21% of prior draws reach at M_h = 10^9.5-10^11 (none of the posterior):
truncated there, they pile up just below the cap in the prior's right panel.

Simulation points come from an npz written on the cluster with --build_sims
(loaders and paths of the prior config). Keys <SIM>_mh / _ms / _sfr / _z.

  python plot_scaling_posteriors.py --build_sims prior_configs/fiducial_public.json --sims_out scaling_sims.npz
  python plot_scaling_posteriors.py <main_joint>/ --sims scaling_sims.npz [--out_dir .]
  python plot_scaling_posteriors.py --prior <priors_generated>/prior_best.json --sims scaling_sims.npz --n_samples 16384 --out_dir .
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from scipy.ndimage import gaussian_filter1d
from scipy.special import log_ndtr, ndtr
from scipy.stats import qmc, truncnorm

from uvlf import ms_mh_flattening, SFMS, sigma_SFR_variable

M_KNEE_FIXED = 2e12
FB = cosmo.Ob0 / cosmo.Om0
Z_SFMS = 10.0
C_68, C_95 = "#e41a1c", "#f4a6a6"      # UVLF + ACF colour of the other figures
C_PRIOR = ("0.5", "0.83")               # prior: grey, as in the other figures
# simulation, snapshot redshift, colour, alpha -- as in the prior figures (Figs. SHMR_prior, sfms_prior).
# FIRE has no z = 10 snapshot in fire_catalog.json: z = 9 is the closest.
SHMR_SIMS = [("FIRE", 9.0, "#800080", 0.15), ("SERRA", 12.0, "#ff8c00", 0.4)]
SFMS_SIMS = [("FLARES", 10.0, "#2e8b57", 0.2), ("ASTRID", 10.0, "#1e90ff", 0.5)]
SIM_LABEL = {"FIRE": "FIRE", "SERRA": "SERRA", "FirstLight": "First Light", "FLARES": "FLARES", "ASTRID": "ASTRID"}
CATALOG_FILES = {"FIRE": "fire_catalog.json", "SERRA": "serra_catalog.json"}   # in prior_pipeline/data
SHMR_COND = [9.5, 10.0, 10.5, 11.0]
SFMS_COND = [7.5, 8.0, 8.5, 9.0]
SHMR_COND_COLORS = ["#3b528b", "#23888e", "#3bbb75", "#bddf26"]     # viridis, as in the prior figure
SFMS_COND_COLORS = ["#51127c", "#a8327d", "#f4675c", "#fecf92"]     # magma, as in the prior figure
FS_LABEL, FS_TICK = 20, 18


def build_sims(config, out):
    from build_priors import load_config
    from prior_pipeline.simulations import load_simulation
    cfg = load_config(config)
    save = {}
    for name, z, *_ in SHMR_SIMS + SFMS_SIMS:
        spec = dict(cfg["simulations"][name])
        if "snapshots" in spec and isinstance(spec["snapshots"], list):     # ASTRID: read only the snapshot needed
            spec["snapshots"] = [min(spec["snapshots"], key=lambda s: abs(s["z"] - z))]
        sim = load_simulation(name, spec)
        snap = min(sim.snapshots, key=lambda s: abs(s.z - z))
        save[f"{name}_z"] = snap.z
        save[f"{name}_ms"], save[f"{name}_sfr"] = snap.ms, snap.sfr
        if snap.mh is not None:
            save[f"{name}_mh"] = snap.mh
        print(f"{name}: z = {snap.z}, {len(snap.ms)} galaxies")
    np.savez(out, **save)
    print("saved", out)


def prior_samples(path, n, seed=0):
    """n draws from a build_priors.py prior file, as mcmc.py's prior() samples it."""
    with open(path) as f:
        spec = json.load(f)
    cov = np.array(spec["cov_effective"])
    assert np.allclose(cov, np.diag(np.diag(cov))), "non-diagonal prior: use mcmc.py's sequential transform"
    u = qmc.Sobol(d=len(spec["params"]), scramble=True, seed=seed).random(n)
    return {name: truncnorm.ppf(u[:, j], (lo - m) / np.sqrt(v), (hi - m) / np.sqrt(v), loc=m, scale=np.sqrt(v))
            for j, (name, m, v, (lo, hi)) in enumerate(zip(spec["params"], spec["mean"], np.diag(cov),
                                                            spec["sampling_limits"]))}


def mixture(x, y, mu, sig, upper=None):
    """CDF and pdf on y (Ny,) of the mixture of the samples at each x: mu, sig
    and upper are (Nsamp, Nx); returns (Nx, Ny) arrays. The truncated CDF uses
    log_ndtr, which stays finite when the mean lies far above the cap."""
    cdf = np.empty((len(x), len(y)))
    for i in range(len(x)):
        t = (y[None, :] - mu[:, i, None]) / sig[:, i, None]
        if upper is None:
            c = ndtr(t)
        else:
            lnorm = log_ndtr((upper[:, i] - mu[:, i]) / sig[:, i])[:, None]
            c = np.where(y[None, :] > upper[:, i, None], 1.0, np.exp(np.minimum(log_ndtr(t) - lnorm, 0.0)))
        cdf[i] = c.mean(0)
    return cdf, np.gradient(cdf, y, axis=1)


def percentiles(y, cdf, qs=(0.025, 0.16, 0.84, 0.975)):
    return {q: np.array([np.interp(q, c, y) for c in cdf]) for q in qs}


def sim_points(name, z, data):
    """Snapshot of a simulation from the cluster npz, or from the catalogue file in the repo."""
    if f"{name}_ms" in data:
        return {k: data[f"{name}_{k}"] for k in ("z", "ms", "sfr", "mh") if f"{name}_{k}" in data}
    if name in CATALOG_FILES:
        from prior_pipeline.simulations import load_catalog_file
        snap = min(load_catalog_file(name, CATALOG_FILES[name]).snapshots, key=lambda s: abs(s.z - z))
        return dict(z=snap.z, ms=snap.ms, sfr=snap.sfr, mh=snap.mh)
    return None


def band_label(ax, x, yline, text, xpos):
    """Label a band edge in place, rotated along the line."""
    i = np.argmin(np.abs(x - xpos))
    p0, p1 = ax.transData.transform([(x[i - 5], yline[i - 5]), (x[i + 5], yline[i + 5])])
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(x[i], yline[i], text, rotation=ang, ha="center", va="center", fontsize=14,
            bbox=dict(facecolor="white", edgecolor="none", pad=1), zorder=6)


def panel(fig, x, y, cdf, pdf, cond, cond_colors, sims, data, xlabel, ylabel, plabel, xlim, ylim, text=None,
          cap=None, colors=(C_68, C_95)):
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.08)
    ax, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    q = percentiles(y, cdf)
    ax.fill_between(x, q[0.025], q[0.975], color=colors[1], alpha=0.6, lw=0)
    ax.fill_between(x, q[0.16], q[0.84], color=colors[0], alpha=0.5, lw=0)
    for k in q:     # an edge within 0.1 dex of the cap would double the cap line (prior)
        ax.plot(x, q[k] if cap is None else np.where(cap - q[k] > 0.1, q[k], np.nan), color="k", lw=2)
    if cap is not None:
        ax.plot(x, cap, color="k", lw=2)
    for name, z, color, alpha in sims:
        d = sim_points(name, z, data)
        if d is None:
            print(f"  {name} not available, skipped")
            continue
        key = "mh" if sims is SHMR_SIMS else "ms"
        with np.errstate(divide="ignore"):
            xs = np.log10(d[key])
            ys = np.log10(d["ms"] if key == "mh" else d["sfr"])
        ok = np.isfinite(xs) & np.isfinite(ys)
        ax.scatter(xs[ok], ys[ok], color=color, alpha=alpha, s=45, edgecolor="none",
                   label=rf"{SIM_LABEL[name]}, $z={float(d['z']):.0f}$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=13, loc="lower right")
    if text:
        ax.text(0.07, 0.92, text, transform=ax.transAxes, fontsize=FS_LABEL, va="top")
    xl = xlim[0] + 0.72 * (xlim[1] - xlim[0])
    band_label(ax, x, q[0.16], "68%", xl)
    band_label(ax, x, q[0.025], "95%", xl)
    if cap is not None:
        band_label(ax, x, cap, r"$M_\ast=(\Omega_{\rm b}/\Omega_{\rm m})M_{\rm h}$", xlim[0] + 0.45 * (xlim[1] - xlim[0]))

    for xc, col in zip(cond, cond_colors):
        p = gaussian_filter1d(pdf[np.argmin(np.abs(x - xc))], 0.05 / (y[1] - y[0]))
        p = p / p.max()
        axc.plot(p, y, color=col, lw=3)
        axc.text(1.04, y[np.argmax(p)], f"{xc:g}", color=col, fontsize=14, va="center", weight="bold")
    axc.set_ylim(ylim)
    axc.set_xlim(0, 1.35)
    axc.set_xticks([0, 1])
    axc.set_yticklabels([])
    axc.tick_params(labelsize=FS_TICK)
    axc.grid(alpha=0.3)
    axc.set_xlabel(plabel, fontsize=FS_LABEL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--sims", default=None, help="simulation points npz from --build_sims")
    ap.add_argument("--build_sims", default=None, help="prior config; write --sims_out and exit")
    ap.add_argument("--sims_out", default="scaling_sims.npz")
    ap.add_argument("--n_samples", type=int, default=1000, help="posterior (or prior) samples used for the mixtures")
    ap.add_argument("--prior", default=None, help="prior file (build_priors.py): plot the prior instead of run_dir")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if args.build_sims:
        build_sims(args.build_sims, args.sims_out)
        return

    if args.prior:
        par = prior_samples(args.prior, args.n_samples)
        colors, names_out = C_PRIOR, ("SHMR_prior.pdf", "SFMS_prior.pdf")
    else:
        with open(os.path.join(args.run_dir, "run_config.json")) as f:
            names = json.load(f)["params"]
        post = np.genfromtxt(os.path.join(args.run_dir, "post_equal_weights.dat"))[:, :len(names)]
        rng = np.random.default_rng(0)
        post = post[rng.choice(len(post), min(args.n_samples, len(post)), replace=False)]
        par = {n: post[:, i] for i, n in enumerate(names)}
        colors, names_out = (C_68, C_95), ("SHMR_posterior_with_cuts-1.pdf", "SFMS_posterior_with_cuts_conditional.pdf")
    data = dict(np.load(args.sims)) if args.sims else {}
    out_dir = args.out_dir or args.run_dir or "."

    # SHMR
    lmh = np.linspace(9.0, 12.0, 151)
    lms_y = np.arange(-2.0, 13.0, 0.01)          # wide enough for the prior tails
    mu = np.array([np.log10(ms_mh_flattening(10 ** lmh, cosmo, fstar_norm=10 ** f, alpha_star_low=a,
                                             M_knee=M_KNEE_FIXED))
                   for f, a in zip(par["fstar_norm"], par["alpha_star_low"])])
    sig = np.repeat(par["sigma_SHMR"][:, None], len(lmh), axis=1)
    cdf, pdf = mixture(lmh, lms_y, mu, sig, upper=np.log10(FB) + lmh[None, :] + 0 * mu)
    fig = plt.figure(figsize=(10, 4.8))
    panel(fig, lmh, lms_y, cdf, pdf, SHMR_COND, SHMR_COND_COLORS, SHMR_SIMS, data,
          r"$\log M_{\rm h}\ [{\rm M_\odot}]$", r"$\log M_\ast\ [{\rm M_\odot}]$",
          r"$p(\log M_\ast\mid\log M_{\rm h})$", (9, 12), (4, 12), cap=np.log10(FB) + lmh, colors=colors)
    out = os.path.join(out_dir, names_out[0])
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)

    # SFMS at z = 10
    lms = np.linspace(6.0, 10.0, 161)
    lsfr_y = np.arange(-8.0, 8.0, 0.01)
    mu = np.array([np.log10(SFMS(10 ** lms, SFR_norm=t, z=Z_SFMS)) for t in par["t_star"]])
    sig = np.array([sigma_SFR_variable(10 ** lms, norm=s, a_sig_SFR=a)
                    for s, a in zip(par["sigma_SFMS_norm"], par["a_sig_SFR"])])
    cdf, pdf = mixture(lms, lsfr_y, mu, sig)
    fig = plt.figure(figsize=(10, 4.8))
    panel(fig, lms, lsfr_y, cdf, pdf, SFMS_COND, SFMS_COND_COLORS, SFMS_SIMS, data,
          r"$\log M_\ast\ [{\rm M_\odot}]$", r"$\log {\rm SFR}_{100}\ [{\rm M_\odot\,yr^{-1}}]$",
          r"$p(\log {\rm SFR}_{100}\mid\log M_\ast)$", (6, 10), (-3, 2), text=rf"$z={Z_SFMS:.0f}$",
          colors=colors)
    out = os.path.join(out_dir, names_out[1])
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
