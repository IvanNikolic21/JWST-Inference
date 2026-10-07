"""How much do the ACF results depend on the integral-constraint definition?

Importance-reweights existing joint runs, using the model w(theta) that
mcmc.py saved for each ACF evaluation (<run>/fs*_sig*_al*.txt for z = 8-10.5,
<run>/ang_z7_fs*... for z = 6-8). Each file holds w_true - c, where c is the
(constant) integral constraint used in the run. Three clustering likelihoods
are evaluated per posterior sample, on the scales the likelihood uses
(theta > 0.005 deg):

  0: as run -- only w > 0 points, field-wide (rectangle) integral constraint
  A: all points (fix #2), same integral constraint
  B: all points, integral constraint from the released random-pair counts
     of Paquereau et al. (2025), w_IC = sum_i w(theta_i) RR_i / sum_i RR_i
     over the measured bins only. With w_true = file + c, the constant c
     cancels: model_B = file - sum_i file(theta_i) RR_i / sum_i RR_i (exact).

Posteriors under A and B follow from weights exp(lnL_A - lnL_0) and
exp(lnL_B - lnL_0); B vs A isolates the integral-constraint choice.
Only for runs with the fiducial parameter set (see reweight_acf_points.py).

  python reweight_acf_ic.py <run_dir> [<run_dir> ...]
"""
import argparse
import json
import os
import re

import numpy as np

from reweight_acf_points import EXTENSION_PARAMS, THETA_CUT, theta_grid_deg, wquantile

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "Paquereau_2025_clustering", "GalClustering_COSMOS-Web_Paquereau2025",
                    "clustering_measurements")
# ACF bin: (file tag, data row for M* > 1e9, saved-model file prefix)
BINS = {
    "z8-10.5": ("zbin8.0-10.5_conservative", 2, "fs"),
    "z6-8": ("zbin6.0-8.0", 2, "ang_z7_fs"),
}


def load_bin(tag, row):
    f = lambda s: os.path.join(DATA, f"clustresults_Paquereau2025_COSMOS-Web_FullSurvey_{tag}_{s}.dat")
    rows = lambda s: [np.array(l.split(), float) for l in open(f(s)) if not l.startswith("#")]
    theta, w, sig = rows("theta")[row - 1], rows("wtheta")[row - 1], rows("wsig")[row - 1]
    hdr = open(f("ndgal")).readline()
    rr = np.array([float(x) for x in re.search(r"RR_pairs = \[([^\]]+)\]", hdr).group(1).split(",")])
    assert len(rr) == len(theta), (tag, len(rr), len(theta))
    return theta, w, sig, rr


def find_file(run_dir, prefix, f, s, a, cache):
    name = (f"{prefix}{np.round(f, 8)}_sig{np.round(s, 8)}_al{np.round(a, 8)}.txt")
    path = os.path.join(run_dir, name)
    if os.path.exists(path):
        return path
    if prefix not in cache:
        pat = re.compile("^" + re.escape(prefix) + r"(-?[\d.e+-]+)_sig(-?[\d.e+-]+)_al(-?[\d.e+-]+)\.txt$")
        keys, names = [], []
        with os.scandir(run_dir) as it:
            for e in it:
                m = pat.match(e.name)
                if m:
                    keys.append([float(x) for x in m.groups()])
                    names.append(e.name)
        cache[prefix] = (np.array(keys), names)
    keys, names = cache[prefix]
    if len(keys) == 0:
        return None
    d = np.max(np.abs(keys - np.array([f, s, a])), axis=1)
    j = int(np.argmin(d))
    return os.path.join(run_dir, names[j]) if d[j] < 2e-8 else None


def chi2(model, w, s, mask):
    return np.sum(((model[mask] - w[mask]) / s[mask]) ** 2)


def summarize(label, x, wts, params, ref=None):
    wts = wts / wts.sum()
    ess = 1.0 / np.sum(wts ** 2)
    out = {}
    print(f"  [{label}] ESS = {ess:.0f} ({100 * ess / len(wts):.0f}%)")
    for j, p in enumerate(params):
        if p == "M_knee":
            continue
        q = wquantile(x[:, j], wts, [0.16, 0.5, 0.84])
        out[p] = q
        line = f"     {p:16s} {q[1]:8.3f} [{q[0]:7.3f},{q[2]:7.3f}]"
        if ref is not None:
            r = ref[p]
            line += f"   shift vs A: {(q[1] - r[1]) / ((r[2] - r[0]) / 2):+6.2f}sd"
        print(line)
    return out


def run(run_dir, theta_deg, bins):
    with open(os.path.join(run_dir, "run_config.json")) as f:
        params = json.load(f)["params"]
    if EXTENSION_PARAMS & set(params):
        print(f"SKIP {run_dir}: extension run")
        return
    post = np.loadtxt(os.path.join(run_dir, "post_equal_weights.dat"))[:, :len(params)]
    jf, js, ja = (params.index(p) for p in ("fstar_norm", "sigma_SHMR", "alpha_star_low"))
    cache = {}
    lnl = {k: np.full(len(post), np.nan) for k in ("0", "A", "B")}
    for k, row in enumerate(post):
        tot = {"0": 0.0, "A": 0.0, "B": 0.0}
        for name, (theta, w, sig, rr, prefix) in bins.items():
            path = find_file(run_dir, prefix, row[jf], row[js], row[ja], cache)
            if path is None:
                tot = None
                break
            file = np.loadtxt(path)
            m_file = np.interp(theta, theta_deg, file)
            ic_rr = np.sum(m_file * rr) / np.sum(rr)
            used = theta > THETA_CUT
            tot["0"] += -0.5 * chi2(m_file, w, sig, used & (w > 0))
            tot["A"] += -0.5 * chi2(m_file, w, sig, used)
            tot["B"] += -0.5 * chi2(m_file - ic_rr, w, sig, used)
        if tot is not None:
            for key in tot:
                lnl[key][k] = tot[key]
    ok = np.isfinite(lnl["0"])
    print(f"\n=== {run_dir}\n  samples {len(post)}, matched in both ACF bins: {ok.sum()} ({100 * ok.mean():.1f}%)")
    x = post[ok]
    res = {}
    for key in ("A", "B"):
        d = lnl[key][ok] - lnl["0"][ok]
        dlnz = np.log(np.mean(np.exp(d - d.max()))) + d.max()
        print(f"  Delta lnZ ({key} - as run) = {dlnz:+.3f}")
        res[key] = summarize(key, x, np.exp(d - d.max()), params, ref=res.get("A"))
    np.save(os.path.join(run_dir, "reweight_acf_ic_lnl.npy"),
            np.vstack([lnl["0"], lnl["A"], lnl["B"]]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args()
    bins = {}
    for name, (tag, row, prefix) in BINS.items():
        theta, w, sig, rr = load_bin(tag, row)
        bins[name] = (theta, w, sig, rr, prefix)
        print(f"{name}: {len(theta)} bins up to {theta.max():.3f} deg; "
              f"{np.sum(theta > THETA_CUT)} used (theta > {THETA_CUT}); "
              f"{100 * rr[theta > 0.02].sum() / rr.sum():.1f}% of RR above 0.02 deg")
    theta_deg = theta_grid_deg()
    for d in args.run_dirs:
        run(d.rstrip("/") + "/", theta_deg, bins)


if __name__ == "__main__":
    main()
