"""Fig. prior_corner (Fig. 3): simulation fits, combined prior and MAP, in the style of
the original figure (corner package, unfilled contours, step histograms).

Each simulation is binned on its own range. On the axis range the bins are far wider
than the fits (0.25 dex in f*, against sd ~ 0.03), which draws them much too wide and
puts the MAP inside First Light. --axis-bins restores that binning for comparison.

  python plot_prior_corner.py --config prior_configs/fiducial_best.json \
      --reference priors_generated/means_best.txt priors_generated/cov_matr_best.txt \
      --out prior_with_MAP.pdf

Needs prior_fits/ and priors_generated/ in the working directory.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from build_priors import gather_posteriors, load_config
from plot_priors import get_model, truncated_samples

# one colour per simulation across the paper (Figs. 1, 2, 9, 11)
COLORS = {"FirstLight": "#8b4513", "ASTRID": "#1e90ff", "SERRA": "#ff8c00",
          "FIRE": "#800080", "FLARES": "#2e8b57"}
NAMES = {"FirstLight": "First Light"}
PARAMS = ["fstar_norm", "sigma_SHMR", "t_star", "alpha_star_low", "sigma_SFMS_norm", "a_sig_SFR"]
LABELS = [r"$\log f_{\ast,0}$", r"$\sigma_{\rm SHMR}$", r"$t_\ast$", r"$\alpha_\ast$",
          r"$\sigma_{{\rm SFMS},0}$", r"$a_{\sigma,{\rm SFMS}}$"]
AXES = [(-5, 0), (0.05, 0.8), (0.02, 1), (0.1, 1), (0.02, 1), (-0.18, 0.12)]
MAP = [-2.632, 0.277, 0.200, 0.510, 0.349, -0.064]      # v4 main_joint
LEVELS = (0.68, 0.95)
LW = 2.0


def own_range(x):
    lo, hi = np.percentile(x, [0.05, 99.95])
    pad = 0.4 * (hi - lo)
    return [lo - pad, hi + pad]


def contours(ax, x, y, xr, yr, color, bins):
    corner.hist2d(x, y, ax=ax, range=[xr, yr], bins=bins, levels=LEVELS, color=color, smooth=1,
                  plot_datapoints=False, plot_density=False, no_fill_contours=True, fill_contours=False,
                  contour_kwargs=dict(colors=[color], linewidths=LW))


def histogram(ax, x, xr, color, bins):
    h, e = np.histogram(x, bins=bins, range=xr)
    ax.stairs(h / h.max(), e, color=color, lw=LW)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--reference", nargs=2, required=True, metavar=("MEANS", "COV"))
    ap.add_argument("--out", default="prior_with_MAP.pdf")
    ap.add_argument("--axis-bins", action="store_true", help="bin the fits on the axis range, as originally")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = get_model(cfg["model"])
    names = list(model.param_names)
    posteriors = gather_posteriors(cfg, model, list(cfg["simulations"]), False, ())

    k_all = len(names)
    divisor = cfg["combine"].get("mcmc_cov_divisor", 5.0)
    mean = np.loadtxt(args.reference[0])[:k_all]
    sd = np.sqrt(np.diag(np.loadtxt(args.reference[1]))[:k_all] / divisor)
    limits = [cfg["sampling_limits"][p] for p in names]
    prior = truncated_samples(mean, sd, limits, 400000, np.random.default_rng(0))
    prior = prior[:, [names.index(p) for p in PARAMS]]

    k = len(PARAMS)
    inside = np.all([(prior[:, i] > AXES[i][0]) & (prior[:, i] < AXES[i][1]) for i in range(k)], axis=0)
    fig = corner.corner(prior[inside][:2000], bins=20, range=AXES, labels=LABELS, plot_datapoints=False,
                        plot_density=False, plot_contours=False, label_kwargs=dict(fontsize=24),
                        hist_kwargs=dict(color="none"), max_n_ticks=5)
    axes = np.array(fig.axes).reshape(k, k)
    colors = COLORS

    for s, (n, x) in posteriors.items():
        idx = {i: n.index(p) for i, p in enumerate(PARAMS) if p in n}
        for i, a in idx.items():
            xr = list(AXES[i]) if args.axis_bins else own_range(x[:, a])
            histogram(axes[i, i], x[:, a], xr, colors[s], 20 if args.axis_bins else 30)
            for j, b in idx.items():
                if j < i:
                    yr, xr2 = (list(AXES[i]), list(AXES[j])) if args.axis_bins else (own_range(x[:, a]), own_range(x[:, b]))
                    contours(axes[i, j], x[:, b], x[:, a], xr2, yr, colors[s], 20 if args.axis_bins else 30)

    for i in range(k):
        full = own_range(prior[:, i])
        histogram(axes[i, i], prior[:, i], full, "k", 40)
        axes[i, i].axvline(MAP[i], color="red", lw=LW)
        axes[i, i].set_xlim(AXES[i])
        axes[i, i].set_ylim(0, 1.1)
        for j in range(i):
            contours(axes[i, j], prior[:, j], prior[:, i], own_range(prior[:, j]), full, "k", 40)
            axes[i, j].plot(MAP[j], MAP[i], "x", color="red", ms=11, mew=2, zorder=10)
            axes[i, j].set_xlim(AXES[j])
            axes[i, j].set_ylim(AXES[i])
    for i in range(k):
        for j in range(i + 1):
            ax = axes[i, j]
            ax.xaxis.set_major_locator(MaxNLocator(5, prune="lower"))
            ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.tick_params(labelsize=16, direction="in", top=True, right=True,
                           labelbottom=i == k - 1, labelleft=j == 0 and i > 0)
            ax.tick_params(axis="x", labelrotation=45)
            ax.tick_params(axis="y", labelrotation=45)
            if i == j:
                ax.set_yticks([])

    handles = [Line2D([], [], color=colors[s], lw=3, label=NAMES.get(s, s)) for s in posteriors]
    handles += [Line2D([], [], color="k", lw=3, label="Combined"),
                Line2D([], [], color="red", lw=3, label="MAP from inference\nusing UVLF and ACF")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=26)
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
