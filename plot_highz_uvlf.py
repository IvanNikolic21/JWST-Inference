"""Fig. diff_num_PG25_trunc: posterior-predictive UV LF at z = 17 and 23
(uvlf_a_post_highz.py output, UVLFs_highz_<f*>.txt, rows z = 17, 23 on
M_UV = linspace(-25, -16, 20)) against the z > 14 photometric estimates of
Castellano+25 and Perez-Gonzalez+25.

Same figure as the paper version, with the labels following the paper's
notation (dn/dM_UV, Perez-Gonzalez accents). The "1 galaxy" points are the
number density of a single galaxy in each Perez-Gonzalez+25 bin, 1/(V_eff dM),
drawn as bars over the bin. Their ratio to the model gives the expected number
of z > 16 galaxies in that survey, N = sum_bins phi_model / phi_1gal; printed
(median, 95% upper) with P(N >= 1) = 1 - exp(-N), for the caption.

Only needs numpy and matplotlib, so it can run on the cluster:

  python plot_highz_uvlf.py <main_joint>/ [--out diff_num_PG25_trunc.pdf]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

MUV = np.linspace(-25, -16, 20)          # as in uvlf_a_post_highz.py
YLABEL = r"${\rm d}n/{\rm d}M_{\rm UV}\,[{\rm mag^{-1}\,Mpc^{-3}}]$"

# Castellano+25: z = 15-20 and z = 20-28 (negative errors: upper limits)
CAS_MUV = np.array([-22.5, -21.5, -20.5, -19.5, -18.5])
CAS_15_20 = (np.array([0.06, 0.14, 0.50, 1.2, 5.4]) * 1e-5,
             np.array([0.14, 0.18, 0.4, 1.6, -0.1]) * 1e-5,
             np.array([0.05, 0.09, 0.25, 0.8, -0.1]) * 1e-5)
CAS_20_28 = (np.array([0.14, 0.19, 0.55, 2.8, 9.5]) * 1e-5,
             np.full(5, -0.1) * 1e-5, np.full(5, -0.1) * 1e-5)

# Perez-Gonzalez+25: z = 16-20 and z = 20-24 (first point of each: upper limit)
PG_16_20_MUV = np.array([-19.79, -19.04, -18.29, -17.54])
_pg1 = 10 ** np.array([-4.8, -4.93, -4.26, -3.96])
PG_16_20 = (_pg1,
            np.r_[-0.1, (10 ** np.array([-4.93 + 0.34, -4.26 + 0.28, -3.96 + 0.28]) - _pg1[1:])],
            np.r_[-0.1, (_pg1[1:] - 10 ** np.array([-4.93 - 1.38, -4.26 - 0.63, -3.96 - 0.58]))])
PG_20_24_MUV = np.array([-18.99, -18.24, -17.49])
_pg2 = 10 ** np.array([-4.6, -4.82, -4.66])
PG_20_24 = (_pg2,
            np.r_[-1, (10 ** np.array([-4.82 + 0.31, -4.66 + 0.33]) - _pg2[1:])],
            np.r_[-1, (_pg2[1:] - 10 ** np.array([-4.82 - 0.75, -4.66 - 1.11]))])

# one galaxy per effective volume (Perez-Gonzalez+25), bins of 0.75 mag
ONE_GAL = {0: (np.array([-19.04, -18.29, -17.54]), np.array([1 / 7.8e4, 1 / 3.1e4, 1 / 1.3e4])),
           1: (np.array([-18.24, -17.49]), np.array([_pg2[1], _pg2[2] / 2]))}


def points(ax, muv, data, xerr, color, marker, label):
    phi, sp, sm = data
    for m, y, p, n in zip(muv, phi, sp, sm):
        if p > 0:
            ax.errorbar(m, y, xerr=[[xerr], [xerr]], yerr=[[n], [p]], color=color, capsize=5,
                        label=label, marker=marker)
        else:
            ax.errorbar(m, y, xerr=[[xerr], [xerr]], yerr=[[y - y / 5], [-p]], color=color, capsize=5,
                        label=label, marker=marker, uplims=True)
        label = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run_dir, "diff_num_PG25_trunc.pdf")

    files = sorted(glob.glob(os.path.join(args.run_dir, "UVLFs_highz_*.txt")))
    uv = np.array([np.loadtxt(f) for f in files])          # (N, 2, 20)
    lo, hi = np.percentile(uv, [2.5, 97.5], axis=0)
    print(f"{len(files)} posterior samples")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
    for iz, (ax, z, ztxt) in enumerate(zip(axes, (17, 23), (r"$z=17$", r"$z=23$"))):
        ax.fill_between(MUV, lo[iz], hi[iz], color="#e41a1c", alpha=0.3, label="posterior, 95% C.I.")
        ax.plot(MUV, lo[iz], color="#e41a1c")
        ax.plot(MUV, hi[iz], color="#e41a1c")
        m1, y1 = ONE_GAL[iz]
        ax.hlines(y1, m1 - 0.375, m1 + 0.375, color="darkslategray", lw=3,
                  label="one galaxy per bin (Pérez-González+25)")
        n_exp = np.sum(np.array([np.interp(m1, MUV, u[iz]) for u in uv]) / y1, axis=1)
        n50, n95 = np.percentile(n_exp, [50, 95])
        ax.text(-22.8, 1e-5, ztxt, ha="left", va="bottom", fontsize=18)
        print(f"z = {z}: expected galaxies in the PG25 bins: median {n50:.2g}, 95% upper {n95:.2g}; "
              f"P(N>=1) < {1 - np.exp(-n95):.2%}")
        ax.set_xlabel(r"$M_{\rm UV}$ [mag]", fontsize=18)
        ax.tick_params(labelsize=18)
        print(f"z = {z}: 95% upper edge / (1 galaxy per effective volume) at M_UV = "
              + ", ".join(f"{m}: {np.interp(m, MUV, hi[iz]) / y:.2g}" for m, y in zip(m1, y1)))

    points(axes[0], CAS_MUV, CAS_15_20, 0.5, "red", "<", "Castellano+25")
    points(axes[0], PG_16_20_MUV, PG_16_20, 0.3, "brown", ">", "Pérez-González+25")
    points(axes[1], CAS_MUV, CAS_20_28, 0.5, "red", "<", None)
    points(axes[1], PG_20_24_MUV, PG_20_24, 0.3, "brown", ">", None)

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
