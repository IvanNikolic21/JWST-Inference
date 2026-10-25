"""Stellar-mass-limit test of the ACF likelihood: v4 runs with the Paquereau+25
clustering samples at log M_*,lim = 8.7, 9.0 (fiducial) and 9.3 (z~7 sample
only; z~9 stays at 9.0), each with the Willott+24 faint-end UVLF (UVLF + ACF)
and without it (ACF only).

Prints, for the text:
  * UVLF + ACF: median shift w.r.t. main_joint in main_joint sd, width ratios;
  * ACF pull: (joint - UVLF-only median) / UVLF-only sd, per M_lim;
  * ACF only: sd_post / sd_prior and shift from the prior median in prior sd;
  * chi^2_min = -2 max lnL of the ACF-only fits and its p-value.
Plots 1D marginals: top row UVLF + ACF (UVLF only dashed), bottom row ACF only
(prior grey).

  python compare_mlim.py <best_prior_runs_v4>/ --prior_file <prior_best.json> [--out mlim_test.pdf]
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import chi2, gaussian_kde

PARAMS = ["fstar_norm", "sigma_SHMR", "t_star", "alpha_star_low", "sigma_SFMS_norm", "a_sig_SFR", "sigma_sfr_10"]
LABELS = {"fstar_norm": r"$\log f_{\ast,0}$", "sigma_SHMR": r"$\sigma_{\rm SHMR}$", "t_star": r"$t_\ast$",
          "alpha_star_low": r"$\alpha_\ast$", "sigma_SFMS_norm": r"$\sigma_{\rm SFMS,0}$",
          "a_sig_SFR": r"$a_{\sigma,\rm SFMS}$", "sigma_sfr_10": r"$\sigma_{\rm SFR,10}$"}
PLOT = [("fstar_norm", (-5.5, 0.5)), ("sigma_SHMR", (0, 1.5)), ("t_star", (0, 0.8)), ("sigma_SFMS_norm", (0, 0.9))]

# log M_*,lim, joint run, ACF-only run, colour, line width
MLIM = [("8.7", "mlim87_joint", "acfonly_m87", "#fdae61", 2.0),
        ("9.0", "main_joint", "acfonly_m9", "#d7191c", 3.0),
        ("9.3", "mlim93_joint", "acfonly_m93", "#2c7bb6", 2.0)]
N_ACF = 10                                          # fitted ACF points (theta > 0.005 deg), all three samples


def load(runs_dir, run):
    with open(os.path.join(runs_dir, run, "run_config.json")) as f:
        names = json.load(f)["params"]
    post = np.loadtxt(os.path.join(runs_dir, run, "post_equal_weights.dat"))
    return post[:, [names.index(p) for p in PARAMS]], post[:, -1].max()


def prior_samples(prior_file, n=400_000, seed=1):
    """Gaussian prior truncated to the sampling limits (as in the sampler)."""
    with open(prior_file) as f:
        pr = json.load(f)
    lim = np.array(pr["sampling_limits"])
    s = np.random.default_rng(seed).multivariate_normal(pr["mean"], pr["cov_effective"], n)
    s = s[np.all((s >= lim[:, 0]) & (s <= lim[:, 1]), axis=1)]
    idx = [pr["params"].index(p) for p in PARAMS]
    return s[:, idx], lim[idx]


def kde(x, grid, lo, hi):
    """Gaussian KDE reflected at the sampling limits, so densities do not leak past them."""
    k = gaussian_kde(x)
    return k(grid) + k(2 * lo - grid) + k(2 * hi - grid)


def row(name, d):
    return f"  {name:14s} " + " ".join(f"{v:+6.2f}" for v in d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir")
    ap.add_argument("--prior_file", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs_dir, "mlim_test.pdf")

    runs = {r: load(args.runs_dir, r) for _, j, a, _, _ in MLIM for r in (j, a)}
    uvlf, _ = load(args.runs_dir, "main_uvlf")
    prior, lim = prior_samples(args.prior_file)
    ref = runs["main_joint"][0]
    med = lambda x: np.median(x, axis=0)

    print("params:        " + " ".join(f"{p[:6]:>6s}" for p in PARAMS))
    print("UVLF + ACF, shift from main_joint / sd(main_joint):")
    for m, j, _, _, _ in MLIM:
        print(row(f"M_lim {m}", (med(runs[j][0]) - med(ref)) / ref.std(0)))
    print("UVLF + ACF, sd / sd(main_joint):")
    for m, j, _, _, _ in MLIM:
        print(row(f"M_lim {m}", runs[j][0].std(0) / ref.std(0)))
    print("ACF pull, (UVLF + ACF - UVLF only) / sd(UVLF only):")
    for m, j, _, _, _ in MLIM:
        print(row(f"M_lim {m}", (med(runs[j][0]) - med(uvlf)) / uvlf.std(0)))
    print("ACF only, sd / sd(prior):")
    for m, _, a, _, _ in MLIM:
        print(row(f"M_lim {m}", runs[a][0].std(0) / prior.std(0)))
    print("ACF only, shift from prior median / sd(prior):")
    for m, _, a, _, _ in MLIM:
        print(row(f"M_lim {m}", (med(runs[a][0]) - med(prior)) / prior.std(0)))
    print(f"ACF only, chi2_min = -2 max lnL ({N_ACF} points):")
    for m, _, a, _, _ in MLIM:
        c = -2 * runs[a][1]
        print(f"  M_lim {m}: chi2_min {c:.1f}, p(dof={N_ACF - 2}) = {chi2.sf(c, N_ACF - 2):.3f}")

    fig, axes = plt.subplots(2, len(PLOT), figsize=(14, 6.5), sharex="col")
    for col, (p, xr) in enumerate(PLOT):
        i = PARAMS.index(p)
        grid = np.linspace(*xr, 400)
        top, bot = axes[0, col], axes[1, col]
        top.plot(grid, kde(uvlf[:, i], grid, *lim[i]), color="grey", ls="--", lw=2, label="UVLF only")
        bot.fill_between(grid, kde(prior[::20, i], grid, *lim[i]), color="lightgrey", label="prior")
        for m, j, a, c, lw in MLIM:
            lab = rf"$\log M_{{\ast,\rm lim}}={m}$" + (r" ($z\sim7$)" if m == "9.3" else "")
            top.plot(grid, kde(runs[j][0][:, i], grid, *lim[i]), color=c, lw=lw, label=lab)
            bot.plot(grid, kde(runs[a][0][:, i], grid, *lim[i]), color=c, lw=lw, label=lab)
        for ax in (top, bot):
            ax.set_xlim(xr)
            ax.set_ylim(0, None)
            ax.set_yticks([])
            ax.tick_params(labelsize=14)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))   # panels touch
        bot.set_xlabel(LABELS[p], fontsize=18)
    axes[0, 0].set_ylabel("UVLF + ACF", fontsize=16)
    axes[1, 0].set_ylabel("ACF only", fontsize=16)
    axes[0, -1].legend(fontsize=13, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    h, l = axes[1, -1].get_legend_handles_labels()
    axes[1, -1].legend(h[:1], l[:1], fontsize=13, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
