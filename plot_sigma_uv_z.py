"""Fig. sigma_z (sigma_UV_z_trunc.pdf): total UV-magnitude scatter sigma_UV at
fixed halo mass M_h = 10^10.8 Msun.

Left: posterior 16-84% band of sigma_UV(z) (Gaussian-fit width of p(M_UV|M_h)
per posterior sample, SigmaMuvMh_<f*>.txt from pmuv_given_mh.py at z = 6, 8,
10, 12; equal-weight duplicates counted by multiplicity), against literature
values (observations.Sigm_UV).
Right: p(M_UV|M_h) of the posterior median (PMuvMh_full_median.npz) at the
four redshifts, smoothed, with its Gaussian fit dashed.

  python plot_sigma_uv_z.py <main_joint>/ [--log_mh 10.8] [--out sigma_UV_z_trunc.pdf]
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from observations import Sigm_UV

_trapz = getattr(np, "trapezoid", None) or np.trapz
Z_COLORS = ["#b10026", "#fc4e2a", "#feb24c", "#fed976"]


def gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def sigma_band(run_dir, i_mh):
    """sigma_UV (N_post, Nz) at mh_grid[i_mh], one row per posterior sample."""
    post = np.genfromtxt(os.path.join(run_dir, "post_equal_weights.dat"))
    cache, rows, missing = {}, [], 0
    for f in post[:, 0]:
        path = os.path.join(run_dir, f"SigmaMuvMh_{f:.8f}.txt")
        if path not in cache:
            cache[path] = np.loadtxt(path)[:, i_mh] if os.path.exists(path) else None
        if cache[path] is None:
            missing += 1
            continue
        rows.append(cache[path])
    print(f"{len(rows)} of {len(post)} posterior samples ({len(cache)} unique), {missing} missing")
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--log_mh", type=float, default=10.8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run_dir, "sigma_UV_z_trunc.pdf")

    npz = np.load(os.path.join(args.run_dir, "PMuvMh_full_median.npz"))
    z_s, mh_grid, muv_grid, pdf = npz["z_s"], npz["mh_grid"], npz["muv_grid"], npz["pdf"]
    i_mh = np.argmin(np.abs(mh_grid - args.log_mh))
    print(f"M_h = 10^{mh_grid[i_mh]:.2f}, z = {z_s}")

    sig = sigma_band(args.run_dir, i_mh)
    lo, med, hi = np.percentile(sig, [16, 50, 84], axis=0)
    for z, l, m, h in zip(z_s, lo, med, hi):
        print(f"  z = {z:4.1f}: sigma_UV median {m:.2f}, 68% [{l:.2f}, {h:.2f}]")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    lit = Sigm_UV()
    ax.fill_between(z_s, lo, hi, alpha=0.3, color="#e41a1c",
                    label=r"${\rm \bf This \ Work}$" + rf"  at $M_{{\rm h}} = 10^{{{args.log_mh:g}}}\,M_\odot$")
    zmin, zmax, s, s_lo, s_hi = lit.get_Carvajal_Bohorquez_25()
    zc = 0.5 * (zmin + zmax)
    ax.errorbar(zc, s, xerr=(zc - zmin, zmax - zc), yerr=(s_lo, s_hi), elinewidth=5, lw=0, fmt=".",
                color="darkviolet", ecolor="violet", ms=20, label="Carvajal-Bohorquez+25")
    z, s = lit.get_Pallottini_23()
    ax.scatter(z, s, color="darkred", marker="D", s=100, label="Pallottini & Ferrara 23")
    z, s = lit.get_Shen_23()
    ax.scatter(z - 0.05, s - 0.025, color="orangered", marker="*", s=200, label="Shen+23")
    (z0, z1), (s0, s1) = lit.get_Ciesla24()
    ax.add_patch(plt.Rectangle((z0, s0), z1 - z0, s1 - s0, color="peru", alpha=0.6, label="Ciesla+24"))
    ax.errorbar(0.5 * (z0 + z1), 0.5 * (s0 + s1), xerr=0.5 * (z1 - z0), yerr=0.5 * (s1 - s0),
                color="peru", elinewidth=5, lw=0, marker=".", ms=20)
    (z0, z1), s = lit.get_Mason_23()
    ax.add_artist(Line2D((z0 - 0.3, z1), (s, s), color="olive", lw=5, label="Mason+23"))
    ax.legend(fontsize=11, loc="upper left")
    ax.set_xlabel("redshift", fontsize=18)
    ax.set_ylabel(r"$\sigma_{\rm UV}$", fontsize=18)
    ax.set_xlim(6, 12)
    ax.tick_params(labelsize=18)

    for iz, z in enumerate(z_s):
        p = pdf[iz, i_mh]
        p = gaussian_filter1d(p / _trapz(p, muv_grid), sigma=3)
        ax2.plot(muv_grid, p, color=Z_COLORS[iz], label=f"z = {z:.0f}")
        popt, _ = curve_fit(gauss, muv_grid, p, p0=[p.max(), muv_grid[np.argmax(p)], 2.0],
                            bounds=([0, muv_grid.min(), 0.1], [np.inf, muv_grid.max(), 20.0]))
        ax2.plot(muv_grid, gauss(muv_grid, *popt), color=Z_COLORS[iz], ls="--", alpha=0.7)
    ax2.set_xlabel(r"$M_{\rm UV}$", fontsize=18)
    ax2.set_ylabel(rf"$p(M_{{\rm UV}}\mid M_{{\rm h}}=10^{{{args.log_mh:g}}}\,M_\odot)$", fontsize=18)
    ax2.set_xlim(-24, -16)
    ax2.tick_params(labelsize=18)
    leg1 = ax2.legend(fontsize=14, loc="upper right")
    ax2.legend(handles=[Line2D([0], [0], color="k", lw=2, label="PDF"),
                        Line2D([0], [0], color="k", lw=2, ls="--", label="Gaussian fit\nto the PDF")],
               fontsize=10, loc="upper left", frameon=False)
    ax2.add_artist(leg1)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
