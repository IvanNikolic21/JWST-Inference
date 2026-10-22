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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# run, label, x label, fiducial value, colour
EXTENSIONS = [("ext_fstar_z", "redshift-dependent SHMR normalization", r"$\alpha_{f_\ast}$", 0.0, "#66c2a5"),
              ("ext_alpha_star_z", "redshift-dependent SHMR slope", r"$\alpha_{\ast,\rm ind}$", 0.0, "#fc8d62"),
              ("ext_slope", "SFMS slope", r"$\alpha_{\rm SFR}$", 1.0, "#8da0cb"),
              ("ext_sigma_z", "redshift-dependent SHMR scatter", r"$\alpha_{\sigma,\rm SHMR,z}$", 0.0, "#e78ac3"),
              ("ext_sigma_mh", "mass-dependent SHMR scatter", r"$a_{\sigma,\rm SHMR}$", 0.0, "#a6d854")]
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

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for ax, (run, label, xlabel, fid, color) in zip(axes, EXTENSIONS):
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

        xr = (max(lim[0], mu - 3 * sd), min(lim[1], mu + 3 * sd))
        n, _, _ = ax.hist(x, bins=np.linspace(*xr, 25), density=True, color=color)
        xx = np.linspace(*xr, 300)
        ax.plot(xx, np.exp(-0.5 * ((xx - mu) / sd) ** 2) / (np.sqrt(2 * np.pi) * sd), color="grey", lw=2)
        top = 1.3 * max(n.max(), 1 / (np.sqrt(2 * np.pi) * sd))      # headroom for the ln B label
        ax.axvline(fid, color="k", lw=3, alpha=0.8)
        ax.text(0.03, 0.95, rf"$\ln B_{{12}}={lnb:+.2f}$", transform=ax.transAxes, ha="left", va="top", fontsize=16)
        ax.set_xlim(xr)
        ax.set_ylim(0, top)
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=18)
        ax.tick_params(labelsize=15)

    axes[-1].axis("off")
    handles = [Patch(color=c, label=l) for _, l, _, _, c in EXTENSIONS]
    handles += [Line2D([], [], color="grey", lw=2, label="prior"),
                Line2D([], [], color="k", lw=3, alpha=0.8, label="fiducial model")]
    axes[-1].legend(handles=handles, loc="center", frameon=False, fontsize=15)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
