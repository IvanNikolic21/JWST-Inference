"""Build the simulation-informed prior (Sec. 4) from scratch.

Replaces basic_prior_setup.ipynb. Two stages:

  fit      fit the galaxy model to each simulation separately (MultiNest by
           default) and store the posterior samples in <fits_dir>/<model>/<sim>/
  combine  pool the posteriors into the Gaussian prior and write
           means_<tag>.txt / cov_matr_<tag>.txt (mcmc.py format) and
           prior_<tag>.json (what was done) to <output_dir>

Examples:
  python build_priors.py fit --config prior_configs/fiducial.json
  python build_priors.py combine --config prior_configs/fiducial.json \
      --compare means_uv.txt cov_matr_uv.txt
  python build_priors.py combine --config prior_configs/fiducial.json --use-legacy
  python build_priors.py all --config prior_configs/sigma_shmr_z.json

A config may name a "base" config whose settings it inherits and overrides
(nested dictionaries are merged), so an extension only lists what changes.
"""
import argparse
import json
import os

import numpy as np

from prior_pipeline.combine import combine, compare, write_prior
from prior_pipeline.fitting import (fit_parameters, fit_simulation,
                                    load_legacy_posterior, load_samples)
from prior_pipeline.models import get_model
from prior_pipeline.simulations import load_simulation


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    base = cfg.pop("base", None)
    if base:
        base_cfg = load_config(os.path.join(os.path.dirname(path), base))
        cfg = _merge(base_cfg, cfg)
    return cfg


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(
            out.get(k), dict) else v
    return out


def fit_dir(cfg, model, sim_name):
    return os.path.join(cfg["fits_dir"], model.name, sim_name)


def run_fits(cfg, model, sims, sampler, force):
    for name in sims:
        spec = cfg["simulations"][name]
        outdir = fit_dir(cfg, model, name)
        if not force and os.path.exists(os.path.join(outdir, "samples.npy")):
            print(f"[{name}] fit exists in {outdir}, skipping (use --force)")
            continue
        sim = load_simulation(name, spec)
        print(f"[{name}] {sim.summary()}; fitting "
              f"{[p.name for p in fit_parameters(model, sim)]}")
        _, samples = fit_simulation(model, sim, outdir,
                                    bounds_override=spec.get("fit_bounds"),
                                    sampler=sampler, seed=cfg.get("seed", 0))
        print(f"[{name}] {len(samples)} posterior samples -> {outdir}")


def gather_posteriors(cfg, model, sims, use_legacy):
    posteriors = {}
    for name in sims:
        spec = cfg["simulations"][name]
        legacy = spec.get("legacy_posterior")
        needed = _needed_params(model, spec)
        if use_legacy and legacy and set(legacy["params"]) == set(needed):
            posteriors[name] = load_legacy_posterior(**legacy)
            source = legacy["path"]
        else:
            outdir = fit_dir(cfg, model, name)
            if not os.path.exists(os.path.join(outdir, "samples.npy")):
                raise FileNotFoundError(
                    f"No posterior for {name} under model '{model.name}': "
                    f"run `fit` first (expected {outdir}).")
            posteriors[name] = load_samples(outdir)
            source = outdir
        print(f"[{name}] {len(posteriors[name][1])} samples from {source}")
    return posteriors


def _needed_params(model, spec):
    """Parameters a simulation constrains, without loading its data."""
    blocks = ("sfms",) if spec.get("halo_masses") is False else ("shmr", "sfms")
    return [p.name for p in model.parameters_in_blocks(blocks)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fit", "combine", "all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--sims", nargs="+", help="subset of simulations")
    ap.add_argument("--sampler", default="multinest",
                    choices=["multinest", "emcee"])
    ap.add_argument("--force", action="store_true", help="redo existing fits")
    ap.add_argument("--use-legacy", action="store_true",
                    help="use the notebook's posteriors where available")
    ap.add_argument("--compare", nargs=2, metavar=("MEANS", "COV"),
                    help="reference prior files to compare against")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = get_model(cfg["model"])
    sims = args.sims or list(cfg["simulations"])

    if args.stage in ("fit", "all"):
        run_fits(cfg, model, sims, args.sampler, args.force)
    if args.stage in ("combine", "all"):
        posteriors = gather_posteriors(cfg, model, sims, args.use_legacy)
        names, mean, cov, info = combine(model.param_names, posteriors,
                                         cfg["combine"])
        means_path, cov_path = write_prior(cfg["output_dir"], cfg["tag"],
                                           names, mean, cov, info,
                                           cfg["combine"])
        print(f"\nWrote {means_path}\n      {cov_path}\n")
        for p, m, v in zip(names, mean, np.diag(cov)):
            print(f"  {p:22s} mean={m:9.4f}  effective sd={np.sqrt(v):.4f}")
        if args.compare:
            print()
            compare(names, mean, np.loadtxt(cov_path), *args.compare)


if __name__ == "__main__":
    main()
