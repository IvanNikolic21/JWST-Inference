"""Fit a galaxy model to one simulation.

The likelihood treats every simulated galaxy as a draw from the model's
log-normal SHMR (log10 M* | Mh) and SFMS (log10 SFR | M*), as in
basic_prior_setup.ipynb. Galaxies with M* = 0 are skipped, as are
galaxies with SFR = 0 in the SFMS term.
"""
import json
import os

import numpy as np

BAD_LOGL = -1e90  # returned for unphysical (non-positive) scatters


def fit_parameters(model, sim):
    """Parameters a simulation can constrain: all of them if it has halo
    masses, otherwise only the SFMS block."""
    blocks = ("shmr", "sfms") if sim.has_halos else ("sfms",)
    return model.parameters_in_blocks(blocks)


def log_likelihood(p, model, sim):
    """ln L of simulation `sim` given parameter dict `p`."""
    lnl = 0.0
    for snap in sim.snapshots:
        has_ms = snap.ms > 0

        sf = has_ms & (snap.sfr > 0)
        sigma = model.sigma_sfms(snap.ms[sf], snap.z, p)
        if np.any(sigma <= 0):
            return BAD_LOGL
        resid = np.log10(snap.sfr[sf]) - model.log_sfr_mean(snap.ms[sf], snap.z, p)
        lnl += np.sum(-0.5 * resid ** 2 / sigma ** 2 - np.log(sigma))

        if snap.mh is None:
            continue
        sigma = model.sigma_shmr(snap.mh[has_ms], snap.z, p)
        if np.any(sigma <= 0):
            return BAD_LOGL
        resid = np.log10(snap.ms[has_ms]) - model.log_ms_mean(snap.mh[has_ms], snap.z, p)
        lnl += np.sum(-0.5 * resid ** 2 / sigma ** 2 - np.log(sigma))
    return lnl


def fit_simulation(model, sim, outdir, bounds_override=None, sampler="multinest",
                   n_live=1000, seed=0):
    """Sample the posterior of `model` given `sim` under uniform priors and
    save it to <outdir>/samples.npy (physical units) + params.json."""
    params = fit_parameters(model, sim)
    names = [q.name for q in params]
    bounds = np.array([(bounds_override or {}).get(q.name, q.fit_bounds)
                       for q in params], dtype=np.float64)
    os.makedirs(outdir, exist_ok=True)

    def loglike_phys(x):
        return log_likelihood(dict(zip(names, x)), model, sim)

    if sampler == "multinest":
        samples = _run_multinest(loglike_phys, bounds, outdir, n_live)
    elif sampler == "emcee":
        samples = _run_emcee(loglike_phys, bounds, seed)
    else:
        raise ValueError(f"Unknown sampler '{sampler}'")

    save_samples(outdir, names, samples, meta={
        "model": model.name, "simulation": sim.name, "sampler": sampler,
        "bounds": bounds.tolist(), "snapshots": sim.summary()})
    return names, samples


def _run_multinest(loglike_phys, bounds, outdir, n_live):
    import pymultinest

    lo, width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    ndim = len(bounds)

    def prior(cube, ndim_, nparams):
        # In place, so MultiNest stores physical values in its outputs.
        for i in range(ndim_):
            cube[i] = lo[i] + cube[i] * width[i]

    def loglike(cube, ndim_, nparams, lnew):
        return loglike_phys(np.array([cube[i] for i in range(ndim_)]))

    # MultiNest truncates output paths to 100 characters (the notebook's
    # FIRE posterior ended up as "post_equal_weights.da"), so run from
    # inside outdir with a short relative basename.
    cwd = os.getcwd()
    os.chdir(outdir)
    try:
        pymultinest.run(
            LogLikelihood=loglike, Prior=prior, n_dims=ndim,
            outputfiles_basename="mn_",
            use_MPI=False, importance_nested_sampling=False,
            sampling_efficiency=0.8, evidence_tolerance=0.5, multimodal=False,
            n_iter_before_update=20, n_live_points=n_live, resume=False,
            verbose=False)
    finally:
        os.chdir(cwd)
    post = np.loadtxt(os.path.join(outdir, "mn_post_equal_weights.dat"))
    return post[:, :ndim]


def _run_emcee(loglike_phys, bounds, seed, n_walkers=32, n_steps=4000,
               n_burn=1500):
    """Lightweight alternative for machines without MultiNest."""
    import emcee

    rng = np.random.default_rng(seed)
    ndim = len(bounds)

    def logpost(x):
        if np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
            return -np.inf
        return loglike_phys(x)

    # Start from the best of a batch of prior draws to avoid BAD_LOGL regions.
    draws = rng.uniform(bounds[:, 0], bounds[:, 1], size=(20000, ndim))
    best = draws[np.argmax([logpost(x) for x in draws])]
    width = 1e-3 * (bounds[:, 1] - bounds[:, 0])
    p0 = np.clip(best + width * rng.standard_normal((n_walkers, ndim)),
                 bounds[:, 0], bounds[:, 1])
    ens = emcee.EnsembleSampler(n_walkers, ndim, logpost)
    ens.run_mcmc(p0, n_steps, progress=False)
    return ens.get_chain(discard=n_burn, thin=5, flat=True)


def save_samples(outdir, names, samples, meta=None):
    np.save(os.path.join(outdir, "samples.npy"), samples)
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump({"params": names, **(meta or {})}, f, indent=1)


def load_samples(outdir):
    with open(os.path.join(outdir, "params.json")) as f:
        names = json.load(f)["params"]
    return names, np.load(os.path.join(outdir, "samples.npy"))


def load_legacy_posterior(path, params, bounds):
    """Posterior written by basic_prior_setup.ipynb. Its prior function did
    not modify the cube in place, so post_equal_weights.dat holds unit-cube
    values that must be mapped back through the fit bounds."""
    post = np.loadtxt(path)[:, :len(params)]
    bounds = np.asarray(bounds, dtype=np.float64)
    return list(params), bounds[:, 0] + post * (bounds[:, 1] - bounds[:, 0])
