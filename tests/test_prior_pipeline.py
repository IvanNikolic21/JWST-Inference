"""Checks that prior_pipeline reproduces basic_prior_setup.ipynb and uvlf.py.

Run with:  python -m pytest tests/test_prior_pipeline.py
"""
import json
import os
import sys

import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prior_pipeline import relations as rel
from prior_pipeline.combine import combine
from prior_pipeline.fitting import log_likelihood
from prior_pipeline.models import MODELS, get_model
from prior_pipeline.simulations import load_catalog_file

P7 = ["fstar_norm", "sigma_SHMR", "t_star", "alpha_star_low",
      "sigma_SFMS_norm", "a_sig_SFR", "M_knee"]
THETA = dict(zip(P7, [-2.5, 0.3, 0.15, 0.5, 0.25, -0.06, 12.3]))


# --- notebook functions, copied verbatim (cells 4, 7, 8, 21) ---------------
def nb_ms_mh_flattening(mh, fstar_norm=1.0, alpha_star_low=0.5, M_knee=2.6e11):
    f_star_mean = fstar_norm
    f_star_mean /= (mh / M_knee) ** (-alpha_star_low) + (mh / M_knee) ** 0.61
    f_star_mean *= (1e10 / M_knee) ** (-alpha_star_low) + (1e10 / M_knee) ** 0.61
    return f_star_mean * mh


def nb_sigma_SFR_variable(Mstar, norm=0.187, a_sig_SFR=-0.117):
    Mstar = np.asarray(Mstar)
    sigma = a_sig_SFR * np.log10(Mstar / 1e10) + norm
    sigma[Mstar > 10 ** 10] = norm
    return sigma


def nb_SFMS(Mstar, SFR_norm=1., z=9.25):
    b_SFR = -np.log10(SFR_norm) + np.log10(cosmo.H(z).to(u.yr ** (-1)).value)
    return Mstar * 10 ** b_SFR


def nb_likelihood_SERRA(p, serra):
    (mh6, ms6, sfr6), (mh12, ms12, sfr12) = serra
    lnL = 0
    for ms, sfr, z in ((ms6, sfr6, 6), (ms12, sfr12, 12)):
        pred = nb_SFMS(ms, SFR_norm=p["t_star"], z=z)
        sig = nb_sigma_SFR_variable(ms, norm=p["sigma_SFMS_norm"],
                                    a_sig_SFR=p["a_sig_SFR"])
        for i, s in enumerate(sfr):
            if s == 0.0:
                continue
            lnL += (-0.5 * (np.log10(s) - np.log10(pred[i])) ** 2 / sig[i] ** 2
                    - np.log(sig[i]))
    for mh, ms in ((mh6, ms6), (mh12, ms12)):
        pred = nb_ms_mh_flattening(mh, fstar_norm=10 ** p["fstar_norm"],
                                   alpha_star_low=p["alpha_star_low"],
                                   M_knee=10 ** p["M_knee"])
        for i, m in enumerate(ms):
            lnL += (-0.5 * (np.log10(m) - np.log10(pred[i])) ** 2
                    / p["sigma_SHMR"] ** 2 - np.log(p["sigma_SHMR"]))
    return lnL


# --- tests ---------------------------------------------------------------
def test_relations_match_uvlf():
    uvlf = pytest.importorskip("uvlf")
    mh = np.logspace(8, 13, 50)
    ms = np.logspace(5, 11, 50)
    np.testing.assert_allclose(
        rel.ms_mh_flattening(mh, 10 ** -2.5, 0.5, 2e12),
        uvlf.ms_mh_flattening(mh, cosmo, 10 ** -2.5, 0.5, 2e12))
    np.testing.assert_allclose(rel.sfms(ms, 0.15, 10), uvlf.SFMS(ms, 0.15, 10))
    np.testing.assert_allclose(rel.sfms_slope(ms, 0.15, 10, 0.8),
                               uvlf.SFMS_new(ms, 0.15, 10, 0.8))
    np.testing.assert_allclose(rel.sigma_sfms(ms, 0.25, -0.06),
                               uvlf.sigma_SFR_variable(ms, 0.25, -0.06))
    np.testing.assert_allclose(rel.sigma_shmr_mass(mh, 0.3, -0.1, 2e12),
                               uvlf.sigma_SHMR_variable(mh, 0.3, -0.1, 2e12))
    assert rel.sigma_linear_z(0.3, 0.5, 8) == uvlf.sigma_linear_z(0.3, 0.5, 8)


def test_likelihood_matches_notebook_serra():
    sim = load_catalog_file("SERRA", "serra_catalog.json")
    serra = [(s.mh, s.ms, s.sfr) for s in sim.snapshots]
    model = get_model("fiducial")
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = {k: v + 0.05 * rng.standard_normal() for k, v in THETA.items()}
        p["sigma_SHMR"] = abs(p["sigma_SHMR"])
        assert np.isclose(log_likelihood(p, model, sim),
                          nb_likelihood_SERRA(p, serra), rtol=1e-10)


def test_extensions_reduce_to_fiducial():
    sim = load_catalog_file("SERRA", "serra_catalog.json")
    ref = log_likelihood(THETA, get_model("fiducial"), sim)
    null = {"sigma_shmr_z": {"alpha_sigma_shmr_z": 0.0},
            "sigma_shmr_mh": {"a_sig_SHMR": 0.0}}
    for name, extra in null.items():
        assert np.isclose(log_likelihood({**THETA, **extra}, get_model(name), sim), ref)
    assert set(MODELS) >= {"fiducial", *null, "slope_sfr"}


def test_negative_scatter_rejected():
    sim = load_catalog_file("SERRA", "serra_catalog.json")
    p = {**THETA, "a_sig_SFR": 0.5, "sigma_SFMS_norm": 0.01}
    assert log_likelihood(p, get_model("fiducial"), sim) < -1e80


def notebook_combination(full, flares):
    """Cells 36-55 of the notebook, on (n_param, n_sample) arrays."""
    cov_matr = np.cov(np.hstack(full))
    mean = np.mean(np.hstack(full), axis=1)
    tup_sfr = [2, 4, 5]
    mean[tup_sfr] = np.mean(np.hstack([f[tup_sfr] for f in full] + [flares]),
                            axis=1)
    cov = np.diag(cov_matr.diagonal())
    cov[0, 0], cov[1, 1], cov[2, 2], cov[4, 4] = 2.0, 0.3, 0.038, 0.02
    cov_uv = np.zeros((8, 8))
    cov_uv[:7, :7] = cov
    cov_uv[7, 7] = 0.1 / 60
    mean_uv = np.append(mean, 0.2)
    return mean_uv, cov_uv * 6  # as saved to means_uv.txt / cov_matr_uv.txt


def test_combine_matches_notebook():
    rng = np.random.default_rng(2)
    theta = np.array([THETA[p] for p in P7])
    centers = [theta + 0.2 * rng.standard_normal(7) for _ in range(4)]
    full = [c[:, None] + 0.05 * rng.standard_normal((7, 5000 + 300 * i))
            for i, c in enumerate(centers)]
    flares = np.array([0.3, 0.3, -0.12])[:, None] + 0.05 * rng.standard_normal((3, 4000))
    ref_mean, ref_cov_file = notebook_combination(full, flares)

    with open(os.path.join(os.path.dirname(__file__), "..", "prior_configs",
                           "fiducial.json")) as f:
        cfg = json.load(f)["combine"]
    posteriors = {f"sim{i}": (P7, x.T) for i, x in enumerate(full)}
    posteriors["FLARES"] = (["t_star", "sigma_SFMS_norm", "a_sig_SFR"], flares.T)
    names, mean, cov, _ = combine(P7, posteriors, cfg)

    assert names == P7 + ["sigma_sfr_10"]
    np.testing.assert_allclose(mean, ref_mean, rtol=1e-12)
    np.testing.assert_allclose(cov * cfg["mcmc_cov_divisor"], ref_cov_file,
                               rtol=1e-12)


def test_config_reproduces_hand_set_file_values():
    """The hand-set entries of cov_matr_uv.txt follow from the config."""
    ref = np.diag(np.loadtxt(os.path.join(os.path.dirname(__file__), "..",
                                          "cov_matr_uv.txt")))
    with open(os.path.join(os.path.dirname(__file__), "..", "prior_configs",
                           "fiducial.json")) as f:
        cfg = json.load(f)["combine"]
    k = cfg["variance_scale"] * cfg["mcmc_cov_divisor"]
    for j, p in enumerate(P7):
        if p in cfg["variance_overrides"]:
            assert np.isclose(cfg["variance_overrides"][p] * k, ref[j])
    assert np.isclose(cfg["extra_params"]["sigma_sfr_10"]["variance"] * k, ref[7])
