"""App. B figure: posterior predictive UV LFs of the runs fitted to other UV LF data.

One row per run (Donnan+24, Whitler+25, Harikane+24, each with the ACFs), columns
z = 6 (not fitted), z ~ 10, z ~ 12 (fitted) and z ~ 14. Red band: 95% C.I. of the run;
black lines: the same for the fiducial main_joint. Red points: the data of that run
used in the likelihood (M_UV > -20); gray: all other estimates at that redshift.

Rows of UVLFs_<f*>.txt follow the --z_list of run_post_appB_v4.sh, repeated in ZS.

  python plot_appB_uvlfs.py RUNS_DIR --out UVLFs_appB.pdf
"""
import argparse
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from observations import Observations

Z_MAIN = [6.0, 7.99, 10.23, 11.0, 11.73, 12.5, 14.0]
ZS = {"uvlf_Donnan": [6.0, 8.99, 9.99, 10.99, 12.47, 14.0],
      "uvlf_Whitler": [6.0, 9.8, 12.8, 14.0, 14.3],
      "uvlf_Harikane": [6.0, 8.99, 10.23, 12.21, 13.5, 14.24]}
LABEL = {"uvlf_Donnan": "Donnan+24", "uvlf_Whitler": "Whitler+25", "uvlf_Harikane": "Harikane+24"}
# run -> (panel z of the run, panel z of main_joint) for each column
COLS = {"uvlf_Donnan": [(6.0, 6.0), (9.99, 10.23), (12.47, 12.5), (14.0, 14.0)],
        "uvlf_Whitler": [(6.0, 6.0), (9.8, 10.23), (12.8, 12.5), (14.0, 14.0)],
        "uvlf_Harikane": [(6.0, 6.0), (10.23, 10.23), (12.21, 12.5), (14.24, 14.0)]}
COL_TEXT = ["$z=6$", r"$z\sim10$", r"$z\sim12$", r"$z\sim14$"]
BETA = ["Hubble", None, -2.5, -2.5]        # dust convention of Fig. uvlf_post, per column
muv_ar = np.linspace(-25, -16, 20)


def gimme_dust(Muv, beta=None):
    if beta is None:
        beta = -0.17 * Muv - 5.40
    elif str(beta) == "Hubble":
        beta = -0.2 * (Muv + 19.5) - 2
    return np.clip(4.43 + 1.99 * beta, 0, 5)


def load_run(runs, name, nz):
    files = sorted(f for f in glob.glob(os.path.join(runs, name, "UVLFs_*.txt"))
                   if "highz" not in os.path.basename(f))
    if not files:
        raise FileNotFoundError(f"no UVLFs_*.txt in {runs}/{name}")
    arr = np.array([np.loadtxt(f) for f in files])
    if arr.shape[1] != nz:
        raise ValueError(f"{name}: {arr.shape[1]} rows, expected {nz}")
    print(f"{name}: {arr.shape[0]} samples")
    return arr


def band(arr, zlist, z):
    return np.percentile(arr[:, zlist.index(z), :], (2.5, 97.5), axis=0)


def draw(ax, obs, color="gray", marker="o", ms=5, cut=False):
    """Points with (upper, lower) errors; a negative error marks a limit, as in observations.py."""
    m, phi = np.asarray(obs[0]), np.asarray(obs[1])
    err = obs[2]
    p, n = (np.asarray(err[0]), np.asarray(err[1])) if isinstance(err, tuple) else (np.asarray(err),) * 2
    for mi, yi, pi, ni in zip(m, phi, p, n):
        c = "gray" if cut and mi < -20 else color
        if pi > 0:
            ax.errorbar(mi, yi, np.array([[ni], [pi]]), color=c, marker=marker, ms=ms, capsize=3, ls="none")
        else:
            ax.errorbar(mi, yi, yerr=0.5 * yi, lolims=True, color=c, marker=marker, ms=ms, capsize=3, ls="none")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs")
    ap.add_argument("--out", default="UVLFs_appB.pdf")
    args = ap.parse_args()

    o = Observations(True, True)
    fitted = {"uvlf_Donnan": [None, o.get_obs_uvlf_z10_Donnan24(), o.get_obs_uvlf_z12_5_Donnan24(), None],
              "uvlf_Whitler": [None, o.get_obs_uvlf_z9_8_Whitler25(), o.get_obs_uvlf_z12_8_Whitler25(),
                               o.get_obs_uvlf_z14_3_Whitler25()],
              "uvlf_Harikane": [None, o.get_obs_uvlf_z10_Harikane24(), o.get_obs_uvlf_z12_Harikane24(),
                                o.get_obs_uvlf_z14_Harikane24()]}
    others = [[(o.get_obs_uvlf_z6_Bouwens21(), "o")],
              [(o.get_obs_uvlf_z10_Willot23(), "^"), (o.get_obs_uvlf_z10_Donnan24(), "<"),
               (o.get_obs_uvlf_z9_8_Whitler25(), ">"), (o.get_obs_uvlf_z10_Harikane24(), "o")],
              [(o.get_obs_uvlf_z12_Willot23(), "^"), (o.get_obs_uvlf_z12_5_Donnan24(), "<"),
               (o.get_obs_uvlf_z12_8_Whitler25(), ">"), (o.get_obs_uvlf_z12_Harikane24(), "o")],
              [(o.get_obs_uvlf_z14_Finkelstein24(), "^"), (o.get_obs_uvlf_z14_3_Whitler25(), ">"),
               (o.get_obs_uvlf_z14_Harikane24(), "o"), (o.get_obs_uvlf_z13_5_McLeod26(), "v")]]

    main_run = load_run(args.runs, "main_joint", len(Z_MAIN))
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0, wspace=0))
    for r, name in enumerate(ZS):
        arr = load_run(args.runs, name, len(ZS[name]))
        for c, (z_run, z_main) in enumerate(COLS[name]):
            ax = axes[r, c]
            x = muv_ar + gimme_dust(muv_ar, beta=BETA[c])
            lo, hi = band(arr, ZS[name], z_run)
            ax.fill_between(x, lo, hi, color="#e41a1c", alpha=0.4, lw=0)
            ax.plot(x, lo, color="#e41a1c", lw=1.5)
            ax.plot(x, hi, color="#e41a1c", lw=1.5)
            for edge in band(main_run, Z_MAIN, z_main):
                ax.plot(x, edge, color="k", lw=1.2)
            own = fitted[name][c]
            for obs, mk in others[c]:
                if own is None or obs[0] is not own[0]:
                    draw(ax, obs, marker=mk)
            if own is not None:
                draw(ax, own, color="#e41a1c", marker="s", ms=6, cut=True)
            ax.set_yscale("log")
            ax.set_ylim(10**-6.8, 1e-2)
            ax.set_xlim(-22.5, -16.4)
            ax.set_yticks([10.0**e for e in range(-6, -2)])
            ax.tick_params(labelsize=14)
            if r == 0:
                ax.set_title(COL_TEXT[c], fontsize=16)
            if c == 3:
                ax.text(0.95, 0.06, LABEL[name], transform=ax.transAxes, ha="right", fontsize=14)
            if r == 2:
                ax.set_xlabel(r"$M_{\rm UV}$ [mag]", fontsize=16)
    fig.supylabel(r"$\mathrm{d}n/\mathrm{d}M_{\rm UV}$ [mag$^{-1}$ Mpc$^{-3}$]", fontsize=18, x=0.05)
    fig.savefig(args.out, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
