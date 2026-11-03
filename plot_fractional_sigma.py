"""Fig. frac_sigma (fractional_variance_sfr10.pdf): CDF over the joint posterior
of the share of the variance of M_UV, 1 - (sigma_UV,reduced / sigma_UV,full)^2,
due to each source of scatter at z = 10, for M_UV = -20 (solid) and -18 (dashed;
SFR10 omitted). The shares add up to ~1 (with sigma_BPASS). --sigma plots the
earlier sigma fractions 1 - sigma_UV,reduced / sigma_UV,full instead
(fractional_sigma_sfr10.pdf).

Reads the two outputs of scatter_contribution_sfr10.py in <main_joint>/
(scatter_contribution_sfr10.npz, scatter_contribution_sfr10_muv-18.npz) and
prints the numbers quoted in the text: medians, 68% intervals, the share of
samples with SFR10 < 10%, P(SFMS > SHMR), and the variance fractions
1 - (1 - f)^2, which add up to ~1.

  python plot_fractional_sigma.py <main_joint>/ [--guides] [--sigma] [--out <pdf>]

--guides adds an 'increasing contribution' arrow.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

SOURCES = [("sfr10", r"$\sigma_{\rm SFR,10}$", "#66c2a5"),
           ("SFMS", r"$\sigma_{\rm SFMS}$", "#fc8d62"),
           ("SHMR", r"$\sigma_{\rm SHMR}$", "#8da0cb")]
FILES = {-20: "scatter_contribution_sfr10.npz", -18: "scatter_contribution_sfr10_muv-18.npz"}


def cdf(ax, x, **kw):
    x = np.sort(x[np.isfinite(x)])
    ax.step(np.concatenate([[0], x, [1]]), np.concatenate([[0], np.arange(1, len(x) + 1) / len(x), [1]]),
            where="post", **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--guides", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sigma", action="store_true", help="plot sigma fractions instead of variance shares")
    args = ap.parse_args()
    name = "fractional_sigma_sfr10.pdf" if args.sigma else "fractional_variance_sfr10.pdf"
    out = args.out or os.path.join(args.run_dir, name)
    data = {m: np.load(os.path.join(args.run_dir, f)) for m, f in FILES.items()}

    for m, d in data.items():
        f = {k: d[f"frac_no_{k}"] for k, _, _ in SOURCES}
        var = {k: 1 - (1 - v) ** 2 for k, v in f.items()}
        tot = sum(var.values())
        print(f"M_UV = {m}: {np.sum(np.isfinite(f['SFMS']))} samples")
        for k, v in f.items():
            lo, med, hi = np.nanpercentile(v, [16, 50, 84])
            print(f"  {k:6s} median {med:.1%}  68% [{lo:.1%}, {hi:.1%}]  variance share {np.nanmedian(var[k]):.1%}")
        print(f"  variance shares: " + ", ".join(
            f"{k} {np.nanmedian(var[k]):.1%} [{np.nanpercentile(var[k], 16):.1%}, {np.nanpercentile(var[k], 84):.1%}]"
            for k in var) + f"; SFR10 variance share < 10% in {np.nanmean(var['sfr10'] < 0.1):.1%}"
            f", < 15% in {np.nanmean(var['sfr10'] < 0.15):.1%} of samples")
        print(f"  SFR10 < 10% in {np.nanmean(f['sfr10'] < 0.1):.1%} of samples; "
              f"SFMS > SHMR in {np.nanmean(f['SFMS'] > f['SHMR']):.1%}; "
              f"sum of variance shares {np.nanmedian(tot):.3f} (16-84% {np.nanpercentile(tot, 16):.3f}-{np.nanpercentile(tot, 84):.3f})")

    def share(d, k):
        f = d[f"frac_no_{k}"]
        return f if args.sigma else 1 - (1 - f) ** 2

    fig, ax = plt.subplots(figsize=(6, 5))
    for k, label, color in SOURCES:
        cdf(ax, share(data[-20], k), color=color, lw=3, label=label)
        if k != "sfr10":
            cdf(ax, share(data[-18], k), color=color, lw=3, ls="--")
    if args.guides:
        ax.annotate("", xy=(0.97, 0.47), xytext=(0.5, 0.47),
                    arrowprops=dict(arrowstyle="-|>", color="grey", lw=2, mutation_scale=18))
        ax.text(0.735, 0.49, "increasing contribution", ha="center", va="bottom", fontsize=13, color="grey")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel(r"Fraction of $\sigma_{\rm UV}$" if args.sigma else r"Fraction of $\sigma_{\rm UV}^2$", fontsize=18)
    ax.set_ylabel("CDF", fontsize=18)
    ax.tick_params(labelsize=18)
    src = ax.legend(fontsize=14, frameon=False, loc="lower right", title="Fractional contribution of:",
                    title_fontsize=13)
    ax.add_artist(src)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color="k", lw=2), Line2D([], [], color="k", lw=2, ls="--")],
              [r"$M_{\rm UV}=-20$", r"$M_{\rm UV}=-18$"], fontsize=14, frameon=False,
              loc="center right", bbox_to_anchor=(1.0, 0.7))
    ax.text(0.1 if args.sigma else 0.27, 0.92 if args.sigma else 0.85, r"$z=10$", fontsize=15)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
