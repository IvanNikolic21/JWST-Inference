"""Corner plot comparing posteriors of the same data under different priors.

  python plot_prior_sensitivity.py --runs-dir <dir with run subfolders> \
      --runs main_joint restrict_joint flat_joint tpool_joint \
      --labels Fiducial Restrictive Flat "Pooled $t_\\ast$" --out prior_sensitivity.pdf

Each run folder holds post_equal_weights.dat and run_config.json (params order).
Parameters that are fixed (zero spread) in the reference run are left out.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_priors import LABELS, contour_levels, draw_1d
from scipy.ndimage import gaussian_filter

LABELS = {**LABELS, "sigma_sfr_10": r"$\sigma_{\rm SFR,10}$"}
COLORS = ["#d62728", "0.35", "#1f78b4", "#33a02c", "#ff7f00"]
STYLES = ["-", "--", "-", "-.", ":"]


def load_run(runs_dir, name):
    with open(os.path.join(runs_dir, name, "run_config.json")) as f:
        params = json.load(f)["params"]
    post = np.loadtxt(os.path.join(runs_dir, name, "post_equal_weights.dat"))
    return params, post[:, :len(params)]


def draw_2d(ax, x, y, xr, yr, color, ls, bins=35, smooth=1.2):
    h, xe, ye = np.histogram2d(x, y, bins=bins, range=[xr, yr])
    h = gaussian_filter(h, smooth)
    if h.sum() == 0:
        return
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    ax.contour(xc, yc, h.T, levels=contour_levels(h, (0.68, 0.95)),
               colors=[color], linestyles=ls, linewidths=1.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="first run is the reference")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out", default="prior_sensitivity.pdf")
    args = ap.parse_args()
    labels = args.labels or args.runs

    runs = {r: load_run(args.runs_dir, r) for r in args.runs}
    ref_params, ref = runs[args.runs[0]]
    names = [p for j, p in enumerate(ref_params) if ref[:, j].std() > 1e-3]
    cols = {r: [params.index(p) for p in names] for r, (params, _) in runs.items()}

    ranges = []
    for p in names:
        vals = [np.percentile(x[:, cols[r][names.index(p)]], [1, 99])
                for r, (_, x) in runs.items()]
        lo, hi = np.min(vals), np.max(vals)
        ranges.append((lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)))

    k = len(names)
    fig, axes = plt.subplots(k, k, figsize=(2.0 * k, 2.0 * k))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            for n, (r, (_, x)) in enumerate(runs.items()):
                c, ls = COLORS[n % len(COLORS)], STYLES[n % len(STYLES)]
                if i == j:
                    draw_1d(ax, x[:, cols[r][i]], ranges[i], c, ls=ls)
                else:
                    draw_2d(ax, x[:, cols[r][j]], x[:, cols[r][i]],
                            ranges[j], ranges[i], c, ls)
            ax.set_xlim(ranges[j])
            if i == j:
                ax.set_yticks([])
                ax.set_ylim(0, 1.1)
            else:
                ax.set_ylim(ranges[i])
            if i < k - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(LABELS.get(names[j], names[j]), fontsize=12)
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS.get(names[i], names[i]), fontsize=12)
            elif i != j:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)

    handles = [Line2D([], [], color=COLORS[n % len(COLORS)],
                      ls=STYLES[n % len(STYLES)], lw=1.6, label=l)
               for n, l in enumerate(labels)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.97),
               fontsize=13, frameon=False)
    fig.subplots_adjust(hspace=0.06, wspace=0.06)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
