"""Corner plot: simulation posteriors, the pure-mixing prior and the prior in use.

  pure mixing  pooled mean and variance of the simulation posteriors
               (diagonal; no variance overrides, no variance_scale)
  in use       a prior file pair as mcmc.py reads it (cov / mcmc_cov_divisor)

Both priors are truncated to the config's sampling_limits, as in mcmc.py.

  python plot_priors.py --config prior_configs/fiducial_public.json --use-legacy \
      --reference means_uv.txt cov_matr_uv.txt --out prior_mixing_vs_used.pdf
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.stats import truncnorm

from build_priors import gather_posteriors, load_config
from prior_pipeline.combine import combine
from prior_pipeline.models import get_model

LABELS = {
    "fstar_norm": r"$\log_{10} f_{\ast,0}$",
    "sigma_SHMR": r"$\sigma_{\rm SHMR}$",
    "t_star": r"$t_\ast$",
    "alpha_star_low": r"$\alpha_\ast$",
    "sigma_SFMS_norm": r"$\sigma_{\rm SFMS,0}$",
    "a_sig_SFR": r"$a_{\sigma,\rm SFMS}$",
    "M_knee": r"$\log_{10} M_{\rm knee}$",
}
SIM_COLORS = {"FirstLight": "#a6cee3", "ASTRID": "#1f78b4", "SERRA": "#b2df8a",
              "FIRE": "#33a02c", "FLARES": "#fb9a99"}
MIXING_COLOR = "0.35"
USED_COLOR = "#d62728"
LEVELS = (0.68, 0.95)


def truncated_samples(mean, sd, limits, n, rng):
    lo, hi = np.array(limits).T
    a, b = (lo - mean) / sd, (hi - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=(n, len(mean)),
                         random_state=rng)


def contour_levels(h, levels):
    """Density thresholds enclosing the given probability masses."""
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return sorted(flat[np.searchsorted(cum, lev)] for lev in levels)


def draw_2d(ax, x, y, xr, yr, color, filled=False, ls="-", bins=40, smooth=1.2):
    h, xe, ye = np.histogram2d(x, y, bins=bins, range=[xr, yr])
    h = gaussian_filter(h, smooth)
    if h.sum() == 0:
        return
    lv = contour_levels(h, LEVELS)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    if filled:
        ax.contourf(xc, yc, h.T, levels=lv + [h.max()], colors=[color],
                    alpha=0.35)
    ax.contour(xc, yc, h.T, levels=lv, colors=[color], linestyles=ls,
               linewidths=1.2)


def draw_1d(ax, x, xr, color, ls="-", bins=60, smooth=1.0, fill=False):
    h, e = np.histogram(x, bins=bins, range=xr, density=True)
    h = gaussian_filter(h, smooth)
    c = 0.5 * (e[1:] + e[:-1])
    ax.plot(c, h / h.max(), color=color, ls=ls, lw=1.4)
    if fill:
        ax.fill_between(c, h / h.max(), color=color, alpha=0.25, lw=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--use-legacy", action="store_true")
    ap.add_argument("--legacy-sims", nargs="+", default=[])
    ap.add_argument("--sims", nargs="+")
    ap.add_argument("--reference", nargs=2, required=True, metavar=("MEANS", "COV"),
                    help="prior in use, as read by mcmc.py")
    ap.add_argument("--out", default="prior_mixing_vs_used.pdf")
    ap.add_argument("--n-draws", type=int, default=400000)
    ap.add_argument("--skip", nargs="+", default=[], metavar="PARAM",
                    help="parameters to leave out of the plot (e.g. a fixed M_knee)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = get_model(cfg["model"])
    names = model.param_names
    sims = args.sims or list(cfg["simulations"])
    posteriors = gather_posteriors(cfg, model, sims, args.use_legacy,
                                   args.legacy_sims)

    mixing_cfg = {**cfg["combine"], "variance_overrides": None,
                  "variance_scale": 1.0, "extra_params": None,
                  "fixed_params": None}
    _, mix_mean, mix_cov, _ = combine(names, posteriors, mixing_cfg)
    mix_sd = np.sqrt(np.diag(mix_cov))

    k = len(names)
    divisor = cfg["combine"].get("mcmc_cov_divisor", 5.0)
    used_mean = np.loadtxt(args.reference[0])[:k]
    used_sd = np.sqrt(np.diag(np.loadtxt(args.reference[1]))[:k] / divisor)

    limits = [cfg["sampling_limits"][p] for p in names]
    rng = np.random.default_rng(0)
    mix = truncated_samples(mix_mean, mix_sd, limits, args.n_draws, rng)
    used = truncated_samples(used_mean, used_sd, limits, args.n_draws, rng)

    print(f"{'param':16s} {'pure mixing':>18s} {'in use':>18s} {'width ratio':>12s}")
    for j, p in enumerate(names):
        print(f"{p:16s} {mix_mean[j]:8.3f} ± {mix_sd[j]:<7.3f} "
              f"{used_mean[j]:8.3f} ± {used_sd[j]:<7.3f} {used_sd[j] / mix_sd[j]:10.2f}")

    keep = [j for j, p in enumerate(names) if p not in args.skip]
    names = [names[j] for j in keep]
    limits = [limits[j] for j in keep]
    mix, used = mix[:, keep], used[:, keep]
    k = len(names)

    # Axis ranges: cover the simulations and the prior in use (to ~99%),
    # clipped to the sampling limits.
    ranges = []
    for j in range(k):
        vals = [np.percentile(used[:, j], [0.5, 99.5])]
        for n, x in posteriors.values():
            if names[j] in n:
                vals.append(np.percentile(x[:, n.index(names[j])], [0.5, 99.5]))
        lo, hi = np.min(vals), np.max(vals)
        pad = 0.05 * (hi - lo)
        ranges.append((max(lo - pad, limits[j][0]), min(hi + pad, limits[j][1])))

    fig, axes = plt.subplots(k, k, figsize=(2.0 * k, 2.0 * k))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                for s, (n, x) in posteriors.items():
                    if names[i] in n:
                        draw_1d(ax, x[:, n.index(names[i])], ranges[i],
                                SIM_COLORS.get(s, "0.6"), fill=True)
                draw_1d(ax, mix[:, i], ranges[i], MIXING_COLOR, ls="--")
                draw_1d(ax, used[:, i], ranges[i], USED_COLOR)
                ax.set_yticks([])
                ax.set_ylim(0, 1.1)
            else:
                for s, (n, x) in posteriors.items():
                    if names[i] in n and names[j] in n:
                        draw_2d(ax, x[:, n.index(names[j])], x[:, n.index(names[i])],
                                ranges[j], ranges[i], SIM_COLORS.get(s, "0.6"),
                                filled=True)
                draw_2d(ax, mix[:, j], mix[:, i], ranges[j], ranges[i],
                        MIXING_COLOR, ls="--")
                draw_2d(ax, used[:, j], used[:, i], ranges[j], ranges[i],
                        USED_COLOR)
                ax.set_ylim(ranges[i])
            ax.set_xlim(ranges[j])
            if i < k - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(LABELS.get(names[j], names[j]), fontsize=12)
            if j > 0 or i == 0:
                if i != j:
                    ax.set_yticklabels([])
            else:
                ax.set_ylabel(LABELS.get(names[i], names[i]), fontsize=12)
            ax.tick_params(labelsize=8)

    handles = [Line2D([], [], color=SIM_COLORS.get(s, "0.6"), lw=6, alpha=0.6,
                      label=s) for s in posteriors]
    handles += [Line2D([], [], color=MIXING_COLOR, ls="--", lw=1.4,
                       label="Pure mixing (pooled simulations)"),
                Line2D([], [], color=USED_COLOR, lw=1.4, label="Prior in use")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.97),
               fontsize=13, frameon=False)
    fig.subplots_adjust(hspace=0.06, wspace=0.06)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
