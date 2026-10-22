"""Fig. alternatives (marginals_alternatives_all.pdf): posterior of the added
parameter of each model extension (Sec. extensions of gal_inf.tex), with its
simulation prior (grey), the fiducial value (black line) and
ln B_12 = ln Z_extension - ln Z_fiducial (main_joint).

The extension parameter is column 7 of post_equal_weights.dat (after the 7
fiducial parameters, before sigma_sfr_10); its prior mean and effective sd are
read from the run's prior file (run_config.json -> prior_file, looked up by
name in --prior_dir). Prints posterior median, 68% interval, sd_post/sd_prior
and ln B_12 with its nested-sampling error, for the text.

  python plot_marginals_alternatives.py <best_prior_runs_v4>/ --prior_dir <priors_generated>/ \\
      [--out marginals_alternatives_all.pdf]
"""
import argparse
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# run, label, x label, fiducial value, colour, x range (edge ticks are pruned, panels touch)
EXTENSIONS = [("ext_fstar_z", "redshift-dependent\nSHMR normalization", r"$\alpha_{f_\ast}$", 0.0, "#66c2a5", (-2.0, 1.5)),
              ("ext_alpha_star_z", "redshift-dependent\nSHMR slope", r"$\alpha_{\ast,\rm ind}$", 0.0, "#fc8d62", (-1.0, 1.5)),
              ("ext_slope", "SFMS slope", r"$\alpha_{\rm SFR}$", 1.0, "#8da0cb", (0.5, 1.6)),
              ("ext_sigma_z", "redshift-dependent\nSHMR scatter", r"$\alpha_{\sigma,\rm SHMR,z}$", 0.0, "#e78ac3", (-1.0, 2.0)),
              ("ext_sigma_mh", "mass-dependent SHMR scatter", r"$a_{\sigma,\rm SHMR}$", 0.0, "#a6d854", (-0.2, 0.1))]
PANELS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
I_EXT = 7


def ln_evidence(run_dir):
    """(ln Z, error) from MultiNest stats.dat."""
    with open(os.path.join(run_dir, "stats.dat")) as f:
        line = f.readline()
    z, err = re.findall(r"[-+]?\d*\.\d+E[-+]\d+", line)
    return float(z), float(err)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir")
    ap.add_argument("--prior_dir", required=True)
    ap.add_argument("--fiducial", default="main_joint")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs_dir, "marginals_alternatives_all.pdf")
    lnz_fid, err_fid = ln_evidence(os.path.join(args.runs_dir, args.fiducial))

    fig = plt.figure(figsize=(14, 7.4))
    gs = GridSpec(2, 3, figure=fig, wspace=0, hspace=0.35)
    for (row, col), (run, label, xlabel, fid, color, xr) in zip(PANELS, EXTENSIONS):
        ax = fig.add_subplot(gs[row, col])
        run_dir = os.path.join(args.runs_dir, run)
        with open(os.path.join(run_dir, "run_config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(args.prior_dir, os.path.basename(cfg["prior_file"]))) as f:
            prior = json.load(f)
        name = cfg["params"][I_EXT]
        assert prior["params"] == cfg["params"], (prior["params"], cfg["params"])
        mu = prior["mean"][I_EXT]
        sd = np.sqrt(np.array(prior["cov_effective"])[I_EXT, I_EXT])
        lim = prior["sampling_limits"][I_EXT]

        x = np.loadtxt(os.path.join(run_dir, "post_equal_weights.dat"))[:, I_EXT]
        lnz, err = ln_evidence(run_dir)
        lnb, lnb_err = lnz - lnz_fid, np.hypot(err, err_fid)
        lo, med, hi = np.percentile(x, [16, 50, 84])
        print(f"{run:17s} {name:19s} median {med:+.3f} [{lo:+.3f}, {hi:+.3f}]  prior {mu:+.3f} +- {sd:.3f}  "
              f"sd_post/sd_prior {x.std() / sd:.2f}  ln B12 {lnb:+.2f} +- {lnb_err:.2f}")

        n, _, _ = ax.hist(x, bins=np.linspace(*xr, 16), density=True, color=color)
        xx = np.linspace(max(xr[0], lim[0]), min(xr[1], lim[1]), 300)       # prior is zero outside its limits
        ax.plot(xx, np.exp(-0.5 * ((xx - mu) / sd) ** 2) / (np.sqrt(2 * np.pi) * sd), color="grey", lw=2)
        ax.axvline(fid, color="k", lw=3, alpha=0.8)
        # ln B in the headroom, on the side away from the fiducial line
        left = (fid - xr[0]) / (xr[1] - xr[0]) > 0.5
        ax.text(0.04 if left else 0.96, 0.95, rf"$\ln B_{{12}} = {lnb:+.2f}$", transform=ax.transAxes,
                ha="left" if left else "right", va="top", fontsize=16)
        ax.set_xlim(xr)
        ax.set_ylim(0, 1.3 * max(n.max(), 1 / (np.sqrt(2 * np.pi) * sd)))
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10], prune="both"))
        ax.set_xlabel(xlabel, fontsize=18)
        ax.tick_params(labelsize=15)

    handles = [Patch(color=c, label=l) for _, l, _, _, c, _ in EXTENSIONS]
    handles += [Line2D([], [], color="grey", lw=2, label="prior"),
                Line2D([], [], color="k", lw=3, alpha=0.8, label="fiducial model")]
    lax = fig.add_subplot(gs[1, 2])
    lax.axis("off")
    lax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.02, 0.5), frameon=False, fontsize=13)
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
