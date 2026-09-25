"""Combine per-simulation posteriors into the Gaussian prior used by mcmc.py.

Steps (reproducing basic_prior_setup.ipynb):
  1. Pool the posterior samples of all simulations that constrain every
     model parameter; the prior mean and covariance are the moments of this
     pool (i.e. of the mixture of the individual posteriors).
  2. Simulations constraining only a subset (FLARES: SFMS block) also enter
     the mean of those parameters (`partial_sims_in_mean`), and optionally
     their variance (`partial_sims_in_variance`, off in the notebook).
  3. Keep only the diagonal (`diagonalize`).
  4. Replace variances listed in `variance_overrides`.
  5. Add parameters not fit to simulations (`extra_params`, e.g.
     sigma_sfr_10) and pin `fixed_params` (e.g. M_knee).
  6. Multiply everything by `variance_scale`.

All variances in the config are *before* step 6. The effective prior that
mcmc.py samples is N(mean, variance * variance_scale); the covariance file is
written multiplied by `mcmc_cov_divisor`, because mcmc.py divides what it
loads by that number (currently 5).
"""
import json
import os
import subprocess

import numpy as np


def combine(model_params, posteriors, cfg):
    """Build the prior. `posteriors` maps simulation -> (names, samples).
    Returns (param_names, mean, effective_variance, info)."""
    names = list(model_params)
    full = {s: _reorder(n, x, names) for s, (n, x) in posteriors.items()
            if set(names) <= set(n)}
    partial = {s: (n, x) for s, (n, x) in posteriors.items() if s not in full}
    if not full:
        raise ValueError("No simulation constrains all model parameters.")

    if cfg.get("equal_weight_sims", False):
        n_min = min(len(x) for x in full.values())
        rng = np.random.default_rng(cfg.get("seed", 0))
        full = {s: x[rng.choice(len(x), n_min, replace=False)]
                for s, x in full.items()}

    pool = np.vstack(list(full.values()))
    mean = pool.mean(axis=0)
    cov = np.cov(pool.T)

    for j, p in enumerate(names):
        extra = [x[:, n.index(p)] for n, x in partial.values() if p in n]
        if not extra:
            continue
        column = np.concatenate([pool[:, j]] + extra)
        if cfg.get("partial_sims_in_mean", True):
            mean[j] = column.mean()
        if cfg.get("partial_sims_in_variance", False):
            cov[j, j] = column.var(ddof=1)

    info = {"pooled_variance": dict(zip(names, np.diag(cov).tolist()))}
    if cfg.get("diagonalize", True):
        cov = np.diag(np.diag(cov))

    for p, var in cfg.get("variance_overrides", {}).items():
        j = names.index(p)
        cov[j, j] = var

    for p, spec in cfg.get("fixed_params", {}).items():
        j = names.index(p)
        mean[j] = spec["mean"]
        cov[j, :] = cov[:, j] = 0.0
        cov[j, j] = spec["variance"]

    extra = cfg.get("extra_params", {})
    if extra:
        k = len(names)
        names += list(extra)
        mean = np.concatenate([mean, [s["mean"] for s in extra.values()]])
        big = np.zeros((len(names), len(names)))
        big[:k, :k] = cov
        for i, s in enumerate(extra.values()):
            big[k + i, k + i] = s["variance"]
        cov = big

    cov = cov * cfg.get("variance_scale", 1.0)
    info["n_samples"] = {s: len(x) for s, x in full.items()}
    info["n_samples_partial"] = {s: len(x) for s, (n, x) in partial.items()}
    return names, mean, cov, info


def write_prior(outdir, tag, names, mean, cov, info, cfg):
    """Write means_<tag>.txt / cov_matr_<tag>.txt in mcmc.py's format, plus
    prior_<tag>.json describing what was done."""
    os.makedirs(outdir, exist_ok=True)
    divisor = cfg.get("mcmc_cov_divisor", 5.0)
    means_path = os.path.join(outdir, f"means_{tag}.txt")
    cov_path = os.path.join(outdir, f"cov_matr_{tag}.txt")
    np.savetxt(means_path, mean)
    np.savetxt(cov_path, cov * divisor)
    meta = {
        "params": names,
        "mean": mean.tolist(),
        "effective_sd": np.sqrt(np.diag(cov)).tolist(),
        "cov_file_multiplier": divisor,
        "git_commit": _git_commit(),
        "config": cfg,
        **info,
    }
    with open(os.path.join(outdir, f"prior_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return means_path, cov_path


def compare(names, mean, cov_file, ref_means_path, ref_cov_path):
    """Print new vs reference prior files, parameter by parameter."""
    ref_mean = np.loadtxt(ref_means_path)
    ref_cov = np.diag(np.loadtxt(ref_cov_path))
    new_cov = np.diag(cov_file)
    print(f"{'param':22s} {'mean new':>11s} {'mean ref':>11s} "
          f"{'cov new':>11s} {'cov ref':>11s}")
    for j, p in enumerate(names):
        rm = ref_mean[j] if j < len(ref_mean) else np.nan
        rc = ref_cov[j] if j < len(ref_cov) else np.nan
        flag = "" if (np.isclose(mean[j], rm, rtol=0.02, atol=1e-3)
                      and np.isclose(new_cov[j], rc, rtol=0.05)) else "  <--"
        print(f"{p:22s} {mean[j]:11.5g} {rm:11.5g} {new_cov[j]:11.5g} "
              f"{rc:11.5g}{flag}")


def _reorder(n, x, names):
    return x[:, [n.index(p) for p in names]]


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None
