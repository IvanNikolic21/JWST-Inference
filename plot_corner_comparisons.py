"""Appendix corner plots comparing v4 runs:

  hmf_models.pdf       (Fig. change_hmf): main_joint (Tinker+08) and the hmf_* runs
  corner_diff_obs.pdf  (Fig. uvlf_diff_inf): main_joint (Willott+24) and the
                       uvlf_Donnan / uvlf_Whitler / uvlf_Harikane runs (faint end + ACF)
  corner_shmr.pdf      (Fig. corner_SHMR): main_uvlf and main_joint, SHMR parameters
  corner_sfms.pdf      (Fig. sfms_corner): main_joint, SFMS parameters
  corner_full.pdf      (Fig. full_corner): main_uvlf and main_joint, all free parameters

Prints the largest median shift of each run relative to main_joint, in units of
the main_joint posterior sd, for the text.

  python plot_corner_comparisons.py <best_prior_runs_v4>/ [--out_dir .] [--only corner_shmr.pdf]
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import corner

LABELS = {"fstar_norm": r"$\log f_{\ast,0}$", "sigma_SHMR": r"$\sigma_{\rm SHMR}$", "t_star": r"$t_\ast$",
          "alpha_star_low": r"$\alpha_\ast$", "sigma_SFMS_norm": r"$\sigma_{\rm SFMS,0}$",
          "a_sig_SFR": r"$a_{\sigma,\rm SFMS}$", "sigma_sfr_10": r"$\sigma_{\rm SFR,10}$"}

FIGURES = {
    "hmf_models.pdf": dict(
        params=["fstar_norm", "sigma_SHMR", "t_star", "alpha_star_low", "sigma_SFMS_norm"],
        ranges=[(-4.5, 0), (0, 0.8), (0, 1), (0, 1), (0, 1)],
        levels=(0.68, 0.95), fill=False,
        runs=[("main_joint", "Tinker et al. 2008", "#e41a1c"),
              ("hmf_SMT", "Sheth, Mo & Tormen 2001", "#1f78b4"),
              ("hmf_Behroozi", "Behroozi et al. 2013", "#ff7f00"),
              ("hmf_Warren", "Warren et al. 2006", "#6a3d9a"),
              ("hmf_PS", "Press & Schechter 1974", "#a6761d"),
              ("hmf_Jenkins", "Jenkins et al. 2001", "#17becf")]),
    "corner_diff_obs.pdf": dict(
        params=["fstar_norm", "alpha_star_low", "sigma_SHMR", "t_star"],
        ranges=[(-4.6, -0.6), (0, 1), (0, 1.15), (0, 1)],
        levels=(0.95,), fill=False,
        runs=[("main_joint", "Willott+24", "#4daf4a"),
              ("uvlf_Donnan", "Donnan+24", "#e6a817"),
              ("uvlf_Whitler", "Whitler+25", "#2b6fdc"),
              ("uvlf_Harikane", "Harikane+24", "#e7298a")]),
    # Fig. corner_SHMR: axes cover >99% of both posteriors (sigma_SHMR to 1.6)
    "corner_shmr.pdf": dict(
        params=["fstar_norm", "alpha_star_low", "sigma_SHMR"],
        ranges=[(-4.6, -1.6), (0, 1), (0, 1.6)],
        levels=(0.68, 0.95), fill=True, ref="main_joint",
        runs=[("main_uvlf", "UVLF only", "#4daf4a"),
              ("main_joint", "UVLF + ACF", "#e41a1c")]),
    # Fig. full_corner (App. D): same axis ranges as corner_shmr / corner_sfms
    "corner_full.pdf": dict(
        params=["fstar_norm", "sigma_SHMR", "t_star", "alpha_star_low", "sigma_SFMS_norm", "a_sig_SFR", "sigma_sfr_10"],
        ranges=[(-4.6, -1.6), (0, 1.6), (0, 0.8), (0, 1), (0, 0.9), (-0.18, 0.1), (0, 0.4)],
        levels=(0.68, 0.95), fill=True, ref="main_joint",
        runs=[("main_uvlf", "UVLF only", "#4daf4a"),
              ("main_joint", "UVLF + ACF", "#e41a1c")]),
    # Fig. sfms_corner: a_sigma axis covers >99.5% of the posterior
    "corner_sfms.pdf": dict(
        params=["t_star", "sigma_SFMS_norm", "a_sig_SFR"],
        ranges=[(0, 0.8), (0, 0.9), (-0.18, 0.1)],
        levels=(0.68, 0.95), fill=True,
        runs=[("main_joint", "UVLF + ACF", "#e41a1c")]),
}


def load(runs_dir, run, params):
    with open(os.path.join(runs_dir, run, "run_config.json")) as f:
        names = json.load(f)["params"]
    post = np.loadtxt(os.path.join(runs_dir, run, "post_equal_weights.dat"))
    return post[:, [names.index(p) for p in params]]


def make(runs_dir, name, spec, out_dir):
    params = spec["params"]
    samples = {run: load(runs_dir, run, params) for run, _, _ in spec["runs"]}
    ref_run = spec.get("ref", spec["runs"][0][0])
    ref = samples[ref_run]
    print(name, f"(shifts relative to {ref_run})")
    for run, label, _ in (r for r in spec["runs"] if r[0] != ref_run):
        shift = (np.median(samples[run], 0) - np.median(ref, 0)) / ref.std(0)
        width = samples[run].std(0) / ref.std(0)
        k = np.argmax(np.abs(shift))
        print(f"  {label:26s} max |shift| {abs(shift[k]):.2f} sd ({params[k]}); "
              + ", ".join(f"{p} {s:+.2f}" for p, s in zip(params, shift))
              + f"; width ratio {width.min():.2f}-{width.max():.2f}")

    fig = None
    for run, label, color in spec["runs"]:
        fig = corner.corner(
            samples[run], fig=fig, range=spec["ranges"], bins=20, color=color,
            levels=spec["levels"], smooth=1.0, smooth1d=None, plot_datapoints=False,
            plot_density=False, fill_contours=spec["fill"],
            contour_kwargs=dict(linewidths=2), hist_kwargs=dict(density=True, lw=2),
            labels=[LABELS[p] for p in params], label_kwargs=dict(fontsize=18),
            max_n_ticks=4)
    n = len(params)
    for i, ax in enumerate(fig.axes):
        ax.tick_params(labelsize=13)
        if i % (n + 1) == 0:                     # 1D panels: rescale to the tallest histogram
            ax.relim()
            ax.autoscale(axis="y")
            ax.set_ylim(0, None)
    fig.axes[n - 1].legend([Line2D([], [], color=c, lw=2.5) for _, _, c in spec["runs"]],
                           [l for _, l, _ in spec["runs"]], loc="upper right", fontsize=17 if n > 4 else 15,
                           frameon=True)
    out = os.path.join(out_dir, name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("  saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--only", nargs="+", default=None, help="make only these figures (file names)")
    args = ap.parse_args()
    for name, spec in FIGURES.items():
        if args.only is None or name in args.only:
            make(args.runs_dir, name, spec, args.out_dir or args.runs_dir)


if __name__ == "__main__":
    main()
