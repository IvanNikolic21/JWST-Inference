"""Whisker plot of prior sensitivity: per parameter, each run's prior (pale bar,
68%) and posterior (dark bar 68%, thin line 95%, dot median), one row per run,
labelled with the evidence relative to the reference run. A second row of
panels shows derived quantities from scatter_contribution_sfr10*.npz.

  python plot_prior_whisker.py --runs-dir <dir> --priors-dir <priors_generated> \
      --out prior_whisker.pdf
"""
import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm

# (run folder, prior json tag, label), top to bottom; the first is the reference
RUNS = [
    ("main_joint", "best", "Fiducial"),
    ("tpool_joint", "best_tstar_pooled", r"Pooled $t_\ast$ width"),
    ("restrict_joint", "restrictive", "Restrictive"),
    ("flat_joint", "flat", "Flat"),
]
PARAMS = [
    ("fstar_norm", r"$\log_{10} f_{\ast,0}$"),
    ("sigma_SHMR", r"$\sigma_{\rm SHMR}$"),
    ("alpha_star_low", r"$\alpha_\ast$"),
    ("t_star", r"$t_\ast$"),
    ("sigma_SFMS_norm", r"$\sigma_{\rm SFMS,0}$"),
    ("a_sig_SFR", r"$a_{\sigma,\rm SFMS}$"),
]
DERIVED = [
    ("sigma_full", r"$\sigma_{\rm UV}$ [mag]", 1.0),
    ("frac_no_SFMS", "SFMS share [%]", 100.0),
    ("frac_no_SHMR", "SHMR share [%]", 100.0),
]
POST_COLOR, PRIOR_COLOR = "#1f4e79", "#c9d6e3"


def log_evidence(run_dir):
    with open(os.path.join(run_dir, "stats.dat")) as f:
        m = re.search(r"Global Log-Evidence\s*:\s*([-\d.E+]+)", f.read())
    return float(m.group(1))


def prior_interval(spec, p):
    j = spec["params"].index(p)
    m, sd = spec["mean"][j], spec["effective_sd"][j]
    lo, hi = spec["sampling_limits"][j]
    t = truncnorm((lo - m) / sd, (hi - m) / sd, loc=m, scale=sd)
    return t.ppf([0.16, 0.84])


def whisker(ax, y, samples, prior=None):
    if prior is not None:
        ax.add_patch(plt.Rectangle((prior[0], y - 0.32), prior[1] - prior[0], 0.64,
                                   color=PRIOR_COLOR, lw=0, zorder=1))
    l95, l68, med, h68, h95 = np.percentile(samples, [2.5, 16, 50, 84, 97.5])
    ax.plot([l95, h95], [y, y], color=POST_COLOR, lw=1.0, zorder=2)
    ax.plot([l68, h68], [y, y], color=POST_COLOR, lw=4.0, solid_capstyle="butt", zorder=3)
    ax.plot(med, y, "o", ms=5, color="white", mec=POST_COLOR, mew=1.5, zorder=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--priors-dir", required=True)
    ap.add_argument("--out", default="prior_whisker.pdf")
    args = ap.parse_args()

    runs = []
    for folder, tag, label in RUNS:
        d = os.path.join(args.runs_dir, folder)
        with open(os.path.join(d, "run_config.json")) as f:
            params = json.load(f)["params"]
        with open(os.path.join(args.priors_dir, f"prior_{tag}.json")) as f:
            prior = json.load(f)
        post = np.loadtxt(os.path.join(d, "post_equal_weights.dat"))[:, :len(params)]
        npz = os.path.join(d, "scatter_contribution_sfr10.npz")
        runs.append(dict(label=label, params=params, post=post, prior=prior,
                         lnz=log_evidence(d),
                         scatter=np.load(npz) if os.path.exists(npz) else None))
    lnz0 = runs[0]["lnz"]
    ys = np.arange(len(runs))[::-1]
    ylabels = [r["label"] + ("" if i == 0 else f"\n$\\Delta\\ln Z={r['lnz'] - lnz0:+.1f}$")
               for i, r in enumerate(runs)]

    fig, axes = plt.subplots(2, len(PARAMS), figsize=(2.1 * len(PARAMS), 4.8),
                             gridspec_kw=dict(height_ratios=[2, 1.1], hspace=0.75))
    for k, (p, lab) in enumerate(PARAMS):
        ax = axes[0, k]
        span = []
        for y, r in zip(ys, runs):
            x = r["post"][:, r["params"].index(p)]
            pr = prior_interval(r["prior"], p)
            whisker(ax, y, x, pr)
            if r is not runs[-1]:
                span += [*pr, *np.percentile(x, [2.5, 97.5])]
        lo, hi = min(span), max(span)
        ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        ax.set_title(lab, fontsize=12)
    drv = [r for r in runs if r["scatter"] is not None]
    ys_d = np.arange(len(drv))[::-1]
    for k, (key, lab, scale) in enumerate(DERIVED):
        ax = axes[1, k]
        for y, r in zip(ys_d, drv):
            v = r["scatter"][key]
            whisker(ax, y, scale * v[np.isfinite(v)])
        ax.set_title(lab, fontsize=11)
        ax.set_ylim(-0.6, len(drv) - 0.4)
        ax.set_yticks(ys_d)
        ax.set_yticklabels([r["label"] for r in drv] if k == 0 else [], fontsize=9)
    axes[1, 0].annotate(r"$\sigma_{\rm UV}$ decomposition at $z=10$, $M_{\rm UV}=-20$",
                        xy=(0, 1.32), xycoords="axes fraction", fontsize=10, color="0.3")
    for k in range(len(DERIVED), len(PARAMS)):
        axes[1, k].axis("off")

    for ax in axes[0]:
        ax.set_ylim(-0.6, len(runs) - 0.4)
        ax.set_yticks(ys)
        ax.set_yticklabels([])
    axes[0, 0].set_yticklabels(ylabels, fontsize=9)
    for ax in axes.ravel():
        if ax.axison:
            ax.tick_params(labelsize=8)
            ax.grid(axis="x", color="0.9", lw=0.6)
            ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=PRIOR_COLOR, label="Prior (68%)"),
               plt.Line2D([], [], color=POST_COLOR, lw=4, label="Posterior 68%"),
               plt.Line2D([], [], color=POST_COLOR, lw=1, label="Posterior 95%")]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.95, 0.06),
               fontsize=10, frameon=False)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
