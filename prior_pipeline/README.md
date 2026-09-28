# Simulation-informed priors (Sec. 4)

Replaces `basic_prior_setup.ipynb`. Entry point: `build_priors.py`.

```bash
# 1. fit the model to each simulation (MultiNest; --sampler emcee without it)
python build_priors.py fit --config prior_configs/fiducial.json
# 2. pool the fits into the prior and compare to the files in use
python build_priors.py combine --config prior_configs/fiducial.json \
    --compare means_uv.txt cov_matr_uv.txt
# or reuse the notebook's existing per-simulation posteriors
python build_priors.py combine --config prior_configs/fiducial.json --use-legacy
# mix: the notebook's posterior for some simulations, new fits for the rest
python build_priors.py combine --config prior_configs/fiducial_public.json \
    --legacy-sims FIRE --tag uv_fireLegacy
# an extension: fit + combine in one go
python build_priors.py all --config prior_configs/sigma_shmr_z.json
```

Outputs go to `output_dir` (default `priors_generated/`, never over the files
in use): `means_<tag>.txt`, `cov_matr_<tag>.txt` in the format `mcmc.py` reads,
and `prior_<tag>.json` with the effective widths, the pooled (pre-override)
variances, the sample counts and the config used.

## Layout

| file | contents |
|---|---|
| `relations.py` | SHMR/SFMS relations, identical to `uvlf.py` (tested) |
| `models.py` | parameters + relations per model; add Sec. 5.1 extensions here |
| `simulations.py` | loaders: First Light, ASTRID, SERRA/generic JSON, FLARES |
| `fitting.py` | likelihood and per-simulation fits |
| `combine.py` | pooling, diagonalization, overrides, output files |
| `export_fire2.py` | builds `data/fire_catalog.json` from public FIRE-2 data |
| `../prior_configs/*.json` | one config per prior; extensions inherit via `"base"` |

## Adding a model

1. In `models.py`, subclass `FiducialModel`, append a `Parameter` (name as in
   `mcmc.py`'s params list), override the relation it changes, register it
   in `MODELS`.
2. Add `prior_configs/<name>.json` with `"base": "fiducial.json"`, `"model"`
   and `"tag"`.
3. `python build_priors.py all --config prior_configs/<name>.json`.

## Conventions

Variances in the config are before `variance_scale` (1.2). The covariance
file is written multiplied by `mcmc_cov_divisor` (5), because `mcmc.py`
divides what it loads by 5; the prior sampled is therefore
N(mean, variance x 1.2), truncated to the limits in `mcmc.py`.

## Kept from the notebook, to revisit

- First Light and ASTRID are evaluated at the target z (5, 10, 15), not the
  snapshot's actual z (ASTRID PIG_035 is z=5.5).
- ASTRID halo mass is the DM-only `MassByType[:, 1]`, and ASTRID masses are
  in 1e10 Msun/h but are multiplied by 1e10 only (0.17 dex too high).
- ASTRID and FLARES SFRs are instantaneous; the model's SFMS is SFR_100.
- Pooling weights simulations by their sample count (`equal_weight_sims`).
- FLARES enters the SFMS means but not the variances.
- FLARES `Mstar_30` is assumed to be in 1e10 Msun; the FLARES fit was not in
  the notebook.
- FIRE's data loading was not in the notebook; see below.

## FIRE-2

`data/fire_catalog.json` is built from the public FIRE-2 High Redshift suite
(22 z5* runs; z = 5, 7, 9; M_vir > 1e9 Msun; low-res fraction < 1%):

```bash
python -m prior_pipeline.export_fire2 --out prior_pipeline/data/fire_catalog.json
```

The Rockstar catalogs (~3 MB/snapshot) give M_vir and M*, but no SFR, so the
SFR over the last 100 Myr is computed from star-particle formation times.
Only those two particle datasets are streamed from the snapshots via HTTP
range requests (~25-35 MB per 2-9 GB snapshot). Particle masses are current,
not initial, masses, so SFRs are slightly low; subhalos are kept.
