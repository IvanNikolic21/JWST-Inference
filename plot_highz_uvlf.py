"""Fig. diff_num_PG25_trunc: posterior-predictive UV LF at z > 15 against the
photometric estimates of Castellano+25 (A&A 704, A158; Tables 2-3, Case 1:
all remaining candidates at high z) and Perez-Gonzalez+25 (ApJ, arXiv:2503.15594v4,
Table 3).

Model: uvlf_a_post_highz.py output on a redshift grid (--tag _grid, see
run_highz_v4.sh), averaged over each redshift range with comoving-volume
weights dV/dz. Left panel z = 15-20, right panel z = 20-28 (Castellano+25
ranges). Falls back to the old two-redshift files (z = 17, 23) if no grid
run exists.

Single-galaxy densities 1/(V_eff dM) for every Perez-Gonzalez+25 bin
(dM = 0.75 mag, V_eff from their Table 3) are drawn as bars. Their ratio to
the model averaged over the Perez-Gonzalez+25 redshift range (16-20, 24-26)
gives the expected number of galaxies in that survey, N = sum_bins phi/phi_1gal;
printed (median, 95% upper) with P(N >= 1) = 1 - exp(-N), for the caption.

  python plot_highz_uvlf.py <main_joint>/ [--out diff_num_PG25_trunc.pdf]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo

_trapz = getattr(np, "trapezoid", None) or np.trapz

YLABEL = r"${\rm d}n/{\rm d}M_{\rm UV}\,[{\rm mag^{-1}\,Mpc^{-3}}]$"
PANELS = [(15.0, 20.0), (20.0, 28.0)]          # model averaging range per panel

# Castellano+25, units 1e-5 Mpc^-3 mag^-1; negative upper error: upper limit
CAS_MUV = np.array([-22.5, -21.5, -20.5, -19.5, -18.5])
CAS = [(np.array([0.06, 0.07, 0.25, 0.6, 5.4]) * 1e-5,
        np.array([0.14, 0.16, 0.3, 1.4, -1]) * 1e-5,
        np.array([0.05, 0.05, 0.16, 0.5, -1]) * 1e-5),
       (np.array([0.14, 0.19, 0.55, 2.8, 9.5]) * 1e-5,
        np.full(5, -1) * 1e-5, np.full(5, -1) * 1e-5)]

# Perez-Gonzalez+25 Table 3: M_UV, log phi (+, -), V_eff [1e4 Mpc^3]; first bin an upper limit
PG_DM = 0.75
PG = [dict(zrange=(16.0, 20.0), muv=np.array([-19.79, -19.04, -18.29, -17.54]),
           logphi=np.array([-4.8, -4.93, -4.26, -3.96]), ep=np.array([np.nan, 0.34, 0.28, 0.28]),
           em=np.array([np.nan, 1.38, 0.63, 0.58]), veff=np.array([7.8, 7.8, 3.1, 1.3]) * 1e4),
      dict(zrange=(24.0, 26.0), muv=np.array([-18.99, -18.24, -17.49]),
           logphi=np.array([-4.6, -4.82, -4.66]), ep=np.array([np.nan, 0.31, 0.33]),
           em=np.array([np.nan, 0.75, 1.11]), veff=np.array([5.5, 4.7, 1.8]) * 1e4)]


def load(run_dir):
    """(z grid, M_UV grid, UV LFs (N, Nz, Nmuv))."""
    meta = os.path.join(run_dir, "highz_grid_meta.npz")
    if os.path.exists(meta):
        m = np.load(meta)
        files = sorted(glob.glob(os.path.join(run_dir, "UVLFs_highz_grid_*.txt")))
        return m["z_s"], m["muv"], np.array([np.atleast_2d(np.loadtxt(f)) for f in files])
    print("no grid run found: using the old z = 17, 23 files (no redshift averaging)")
    files = sorted(glob.glob(os.path.join(run_dir, "UVLFs_highz_-*.txt")))
    return np.array([17.0, 23.0]), np.linspace(-25, -16, 20), np.array([np.loadtxt(f) for f in files])


def z_average(z_s, uv, zrange):
    """Comoving-volume-weighted mean over zrange of uv (..., Nz, Nmuv)."""
    sel = (z_s >= zrange[0] - 1e-6) & (z_s <= zrange[1] + 1e-6)
    if sel.sum() < 2:                       # old two-redshift files
        return uv[..., np.argmin(abs(z_s - np.mean(zrange))), :]
    zz = z_s[sel]
    w = cosmo.differential_comoving_volume(zz).value
    return _trapz(uv[..., sel, :] * w[:, None], zz, axis=-2) / _trapz(w, zz)


def errorbars(ax, muv, phi, ep, em, xerr, color, marker, label):
    for m, y, p, n in zip(muv, phi, ep, em):
        if p > 0:
            ax.errorbar(m, y, xerr=xerr, yerr=[[n], [p]], color=color, capsize=5, marker=marker, label=label)
        else:
            ax.errorbar(m, y, xerr=xerr, yerr=[[y - y / 5], [0]], color=color, capsize=5, marker=marker,
                        uplims=True, label=label)
        label = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run_dir, "diff_num_PG25_trunc.pdf")

    z_s, muv, uv = load(args.run_dir)
    print(f"{len(uv)} posterior samples, z = {z_s}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
    for iz, ax in enumerate(axes):
        zr = PANELS[iz]
        lo, hi = np.percentile(z_average(z_s, uv, zr), [2.5, 97.5], axis=0)
        ax.fill_between(muv, lo, hi, color="#e41a1c", alpha=0.3, label="posterior, 95% C.I.")
        ax.plot(muv, lo, color="#e41a1c")
        ax.plot(muv, hi, color="#e41a1c")
        ax.text(-22.8, 1e-5, rf"$z={zr[0]:.0f}$–${zr[1]:.0f}$", ha="left", va="bottom", fontsize=18)

        c = CAS[iz]
        errorbars(ax, CAS_MUV, c[0], c[1], c[2], 0.5, "#377eb8", "<", "Castellano+25")
        pg = PG[iz]
        phi = 10 ** pg["logphi"]
        ep = np.where(np.isnan(pg["ep"]), -1, 10 ** (pg["logphi"] + pg["ep"]) - phi)
        em = np.where(np.isnan(pg["em"]), -1, phi - 10 ** (pg["logphi"] - pg["em"]))
        errorbars(ax, pg["muv"], phi, ep, em, PG_DM / 2, "brown", ">", "Pérez-González+25")
        one_gal = 1 / (pg["veff"] * PG_DM)
        ax.hlines(one_gal, pg["muv"] - PG_DM / 2, pg["muv"] + PG_DM / 2, color="darkslategray", lw=3,
                  label="one galaxy per bin (Pérez-González+25)")

        model_pg = z_average(z_s, uv, pg["zrange"])                       # (N, Nmuv)
        n_exp = np.sum(np.array([np.interp(pg["muv"], muv, u) for u in model_pg]) / one_gal, axis=1)
        n50, n95 = np.percentile(n_exp, [50, 95])
        print(f"Perez-Gonzalez+25 z = {pg['zrange'][0]:.0f}-{pg['zrange'][1]:.0f}: expected galaxies in their "
              f"survey: median {n50:.2g}, 95% upper {n95:.2g}; P(N>=1) < {1 - np.exp(-n95):.2%}")
        ax.set_xlabel(r"$M_{\rm UV}$ [mag]", fontsize=18)
        ax.tick_params(labelsize=18)

    axes[0].set_ylabel(YLABEL, fontsize=18)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-10, 5e-4)
    axes[0].set_xlim(-23, -17)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(l) for l in ("posterior, 95% C.I.", "one galaxy per bin (Pérez-González+25)",
                                       "Castellano+25", "Pérez-González+25")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=13, frameon=False)
    fig.subplots_adjust(wspace=0)
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
