"""Fig. frac_sigma (fractional_sigma_sfr10.pdf): CDF over the joint posterior of
the fractional contribution 1 - sigma_UV,reduced / sigma_UV,full of each source
of scatter at z = 10, for M_UV = -20 (solid) and -18 (dashed; SFR10 omitted).

Reads the two outputs of scatter_contribution_sfr10.py in <main_joint>/
(scatter_contribution_sfr10.npz, scatter_contribution_sfr10_muv-18.npz) and
prints the numbers quoted in the text: medians, 68% intervals, the share of
samples with SFR10 < 10%, P(SFMS > SHMR), and the variance fractions
1 - (1 - f)^2, which add up to ~1.

  python plot_fractional_sigma.py <main_joint>/ [--guides] [--out fractional_sigma_sfr10.pdf]

--guides marks each median (CDF = 0.5) at M_UV = -20.
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
    args = ap.parse_args()
    out = args.out or os.path.join(args.run_dir, "fractional_sigma_sfr10.pdf")
    data = {m: np.load(os.path.join(args.run_dir, f)) for m, f in FILES.items()}

    for m, d in data.items():
        f = {k: d[f"frac_no_{k}"] for k, _, _ in SOURCES}
        var = {k: 1 - (1 - v) ** 2 for k, v in f.items()}
        tot = sum(var.values())
        print(f"M_UV = {m}: {np.sum(np.isfinite(f['SFMS']))} samples")
        for k, v in f.items():
            lo, med, hi = np.nanpercentile(v, [16, 50, 84])
            print(f"  {k:6s} median {med:.1%}  68% [{lo:.1%}, {hi:.1%}]  variance share {np.nanmedian(var[k]):.1%}")
        print(f"  SFR10 < 10% in {np.nanmean(f['sfr10'] < 0.1):.1%} of samples; "
              f"SFMS > SHMR in {np.nanmean(f['SFMS'] > f['SHMR']):.1%}; "
              f"sum of variance shares {np.nanmedian(tot):.3f} (16-84% {np.nanpercentile(tot, 16):.3f}-{np.nanpercentile(tot, 84):.3f})")

    fig, ax = plt.subplots(figsize=(6, 5))
    for k, label, color in SOURCES:
        cdf(ax, data[-20][f"frac_no_{k}"], color=color, lw=3, label=label)
        if k != "sfr10":
            cdf(ax, data[-18][f"frac_no_{k}"], color=color, lw=3, ls="--")
        if args.guides:
            med = np.nanmedian(data[-20][f"frac_no_{k}"])
            ax.plot(med, 0.5, "o", color=color, ms=9, mec="k", zorder=5)
            ax.annotate(f"{med:.0%}", (med, 0.5), xytext=(6, -16), textcoords="offset points",
                        fontsize=13, color=color, fontweight="bold")
    if args.guides:
        ax.axhline(0.5, color="grey", ls=":", lw=1.5, zorder=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel(r"Fraction of $\sigma_{\rm UV}$", fontsize=18)
    ax.set_ylabel("CDF", fontsize=18)
    ax.tick_params(labelsize=18)
    src = ax.legend(fontsize=14, frameon=False, loc="lower right", title="Fractional contribution of:",
                    title_fontsize=13)
    ax.add_artist(src)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color="k", lw=2), Line2D([], [], color="k", lw=2, ls="--")],
              [r"$M_{\rm UV}=-20$", r"$M_{\rm UV}=-18$"], fontsize=14, frameon=False,
              loc="center right", bbox_to_anchor=(1.0, 0.7))
    ax.text(0.1, 0.92, r"$z=10$", fontsize=15)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
