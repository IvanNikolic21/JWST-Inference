"""Importance-reweight existing joint runs for ACF points the old code dropped.

Until commit <this one>, observations.get_obs_* kept only w(theta) > 0, so the
z = 8-10.5 ACF likelihood lost two noise-dominated large-scale points. During
the joint runs, mcmc.py saved the model w(theta) - w_IC of every Ang_z9_m9
evaluation as <run_dir>/fs<f*>_sig<sigma_SHMR>_al<alpha*>.txt on halomod's
theta grid, so the missing likelihood terms can be added exactly:

    w_k  propto  exp( sum_i -0.5 [(w_mod,k(theta_i) - w_i) / sigma_i]^2 )

over the dropped points i, for every posterior sample k. This gives the
corrected posterior (weighted samples), the effective sample size, and
Z_new / Z_old = < exp(Delta lnL) >_posterior.

Only valid for runs whose clustering model depends on (f*, sigma_SHMR,
alpha*) alone, i.e. the fiducial parameter set with M_knee fixed (main_joint,
restrict_joint, flat_joint, tpool_joint, oldprior_joint): the file names do
not encode the extension parameters.

  python reweight_acf_points.py <run_dir> [<run_dir> ...]
"""
import argparse
import json
import os
import re
import sys

import numpy as np

from observations import Observations

THETA_MIN, THETA_MAX, THETA_NUM = 10 ** -6.3, 10 ** -0.8, 50   # LikelihoodAngBase
THETA_CUT = 0.005                                               # deg, as in call_likelihood
EXTENSION_PARAMS = {"alpha_fstar_z", "alpha_star_z", "alpha_sigma_shmr_z",
                    "a_sig_SHMR", "slope_SFR"}


def theta_grid_deg():
    """halomod's AngularCF theta grid (radians), in degrees."""
    theta = np.logspace(np.log10(THETA_MIN), np.log10(THETA_MAX), THETA_NUM)
    try:
        import halomod as hm
        t_hm = hm.AngularCF(theta_min=THETA_MIN, theta_max=THETA_MAX,
                            theta_num=THETA_NUM, theta_log=True).theta
        if not np.allclose(t_hm, theta):
            print("WARNING: halomod theta grid differs from logspace; using halomod's")
            theta = np.asarray(t_hm)
    except Exception as e:  # halomod not importable here: logspace is its definition
        print(f"(halomod not available, using logspace theta grid: {e})")
    return theta * 180.0 / np.pi


def model_file(run_dir, f, s, a):
    return os.path.join(run_dir, "fs" + str(np.round(f, 8)) + "_sig" + str(np.round(s, 8))
                        + "_al" + str(np.round(a, 8)) + ".txt")


def index_model_files(run_dir):
    """Fallback lookup: parse every fs*_sig*_al*.txt name into floats."""
    pat = re.compile(r"^fs(-?[\d.e+-]+)_sig(-?[\d.e+-]+)_al(-?[\d.e+-]+)\.txt$")
    keys, names = [], []
    with os.scandir(run_dir) as it:
        for e in it:
            m = pat.match(e.name)
            if m:
                keys.append([float(x) for x in m.groups()])
                names.append(e.name)
    return np.array(keys), names


def wquantile(x, w, qs):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / np.sum(w)
    return np.interp(qs, c, x[o])


def reweight(run_dir, theta_deg, dropped):
    with open(os.path.join(run_dir, "run_config.json")) as f:
        params = json.load(f)["params"]
    ext = EXTENSION_PARAMS & set(params)
    if ext:
        print(f"SKIP {run_dir}: extension parameters {sorted(ext)} are not encoded in the file names")
        return
    post = np.loadtxt(os.path.join(run_dir, "post_equal_weights.dat"))[:, :len(params)]
    jf, js, ja = (params.index(p) for p in ("fstar_norm", "sigma_SHMR", "alpha_star_low"))
    th_d, w_d, s_d = dropped

    index = None
    dlnl = np.full(len(post), np.nan)
    for k, row in enumerate(post):
        path = model_file(run_dir, row[jf], row[js], row[ja])
        if not os.path.exists(path):
            if index is None:
                print("  building file index (exact names not found for some samples)...")
                index = index_model_files(run_dir)
            keys, names = index
            d = np.max(np.abs(keys - row[[jf, js, ja]]), axis=1)
            j = int(np.argmin(d))
            if d[j] > 2e-8:
                continue
            path = os.path.join(run_dir, names[j])
        model = np.loadtxt(path)
        wm = np.interp(th_d, theta_deg, model)
        dlnl[k] = np.sum(-0.5 * ((wm - w_d) / s_d) ** 2)

    ok = np.isfinite(dlnl)
    print(f"\n=== {run_dir}\n  samples: {len(post)}, matched to a saved model: {ok.sum()}"
          f" ({100 * ok.mean():.1f}%)")
    if ok.sum() < 0.95 * len(post):
        print("  WARNING: >5% of samples unmatched; results below use matched samples only")
    x = post[ok]
    lw = dlnl[ok] - dlnl[ok].max()
    w = np.exp(lw)
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    dlnz = np.log(np.mean(np.exp(dlnl[ok] - dlnl[ok].max()))) + dlnl[ok].max()
    print(f"  ESS = {ess:.0f} ({100 * ess / ok.sum():.0f}% of samples);"
          f"  Delta lnZ (new - old) = {dlnz:+.3f}")
    print(f"  {'param':16s} {'old median [16,84]':>28s} {'reweighted median [16,84]':>30s} {'shift':>7s}")
    for j, p in enumerate(params):
        if p == "M_knee":
            continue
        q0 = np.percentile(x[:, j], [16, 50, 84])
        q1 = wquantile(x[:, j], w, [0.16, 0.5, 0.84])
        print(f"  {p:16s} {q0[1]:8.3f} [{q0[0]:7.3f},{q0[2]:7.3f}]   {q1[1]:8.3f} [{q1[0]:7.3f},{q1[2]:7.3f}]"
              f" {(q1[1] - q0[1]) / ((q0[2] - q0[0]) / 2):+6.2f}sd")
    np.save(os.path.join(run_dir, "reweight_acf_points_weights.npy"),
            np.where(ok, np.exp(np.nan_to_num(dlnl, nan=-np.inf) - np.nanmax(dlnl)), 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args()

    obs = Observations(ang=True, uvlf=False)
    th, w, s = (np.array(v) for v in obs.get_obs_z9_m90(positive_only=False))
    mask = (th > THETA_CUT) & (w <= 0)
    dropped = (th[mask], w[mask], s[mask])
    print("Points added back (z = 8-10.5, M* > 1e9):")
    for t_, w_, s_ in zip(*dropped):
        print(f"  theta = {t_:.4e} deg   w = {w_:+.3f} +- {s_:.3f}")
    if not mask.any():
        sys.exit("nothing to add")
    theta_deg = theta_grid_deg()
    for d in args.run_dirs:
        reweight(d.rstrip("/") + "/", theta_deg, dropped)


if __name__ == "__main__":
    main()
