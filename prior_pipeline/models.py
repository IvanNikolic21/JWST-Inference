"""Galaxy models whose parameters are fit to simulations to build the prior.

A model is a list of Parameters plus four relations:

    log10 <M*>(Mh, z),  sigma_SHMR(Mh, z),  log10 <SFR>(M*, z),  sigma_SFMS(M*, z)

Every Parameter belongs to a block: 'shmr' parameters need halo masses to be
constrained, 'sfms' parameters only need stellar masses and SFRs. Simulations
without halo masses (e.g. FLARES) are therefore fit only for the 'sfms' block.

Adding an extension (Sec. 5.1 of the paper)
-------------------------------------------
Subclass FiducialModel, append the new Parameter to PARAMETERS (name must
match the name used in mcmc.py's params list, since the prior files are
ordered by it), override the relation it modifies, and register the class in
MODELS. See SigmaSHMRz below for a minimal example.
"""
from dataclasses import dataclass

import numpy as np

from . import relations as rel


@dataclass(frozen=True)
class Parameter:
    name: str
    block: str          # 'shmr' or 'sfms'
    fit_bounds: tuple   # uniform prior used when fitting a single simulation


FIDUCIAL_PARAMETERS = (
    Parameter("fstar_norm", "shmr", (-6.0, 1.0)),       # log10 f_*,0
    Parameter("sigma_SHMR", "shmr", (0.001, 2.0)),
    Parameter("t_star", "sfms", (0.001, 1.0)),
    Parameter("alpha_star_low", "shmr", (0.0, 2.0)),
    Parameter("sigma_SFMS_norm", "sfms", (0.001, 1.5)),
    Parameter("a_sig_SFR", "sfms", (-1.0, 0.5)),
    Parameter("M_knee", "shmr", (10.0, 16.0)),          # log10 M_knee
)


class FiducialModel:
    """Fiducial model of Sec. 2: knee SHMR with constant scatter, SFMS set by
    t_star with mass-dependent scatter."""

    name = "fiducial"
    PARAMETERS = FIDUCIAL_PARAMETERS

    @property
    def param_names(self):
        return [p.name for p in self.PARAMETERS]

    def parameters_in_blocks(self, blocks):
        return [p for p in self.PARAMETERS if p.block in blocks]

    # --- SHMR -------------------------------------------------------------
    def log_ms_mean(self, mh, z, p):
        return np.log10(rel.ms_mh_flattening(
            mh, fstar_norm=10 ** p["fstar_norm"],
            alpha_star_low=p["alpha_star_low"], M_knee=10 ** p["M_knee"]))

    def sigma_shmr(self, mh, z, p):
        return np.full(np.shape(mh), p["sigma_SHMR"], dtype=np.float64)

    # --- SFMS -------------------------------------------------------------
    def log_sfr_mean(self, ms, z, p):
        return np.log10(rel.sfms(ms, p["t_star"], z))

    def sigma_sfms(self, ms, z, p):
        return rel.sigma_sfms(ms, p["sigma_SFMS_norm"], p["a_sig_SFR"])


class SigmaSHMRz(FiducialModel):
    """Redshift-dependent SHMR scatter (uvlf.sigma_linear_z)."""

    name = "sigma_shmr_z"
    PARAMETERS = FIDUCIAL_PARAMETERS + (
        Parameter("alpha_sigma_shmr_z", "shmr", (-1.0, 2.0)),)

    def sigma_shmr(self, mh, z, p):
        sigma = rel.sigma_linear_z(p["sigma_SHMR"], p["alpha_sigma_shmr_z"], z)
        return np.full(np.shape(mh), sigma, dtype=np.float64)


class SigmaSHMRmass(FiducialModel):
    """Mass-dependent SHMR scatter anchored at M_knee
    (uvlf.sigma_SHMR_variable with M_char=M_knee)."""

    name = "sigma_shmr_mh"
    PARAMETERS = FIDUCIAL_PARAMETERS + (
        Parameter("a_sig_SHMR", "shmr", (-1.0, 1.0)),)

    def sigma_shmr(self, mh, z, p):
        return rel.sigma_shmr_mass(mh, p["sigma_SHMR"], p["a_sig_SHMR"],
                                   10 ** p["M_knee"])


class SlopeSFMS(FiducialModel):
    """SFMS with a free slope pivoting at 10^9 Msun (uvlf.SFMS_slope);
    slope_SFR = 1 is the fiducial model."""

    name = "slope_sfr"
    PARAMETERS = FIDUCIAL_PARAMETERS + (
        Parameter("slope_SFR", "sfms", (0.5, 1.5)),)

    def log_sfr_mean(self, ms, z, p):
        return np.log10(rel.sfms_slope(ms, p["t_star"], z, p["slope_SFR"]))


class _SHMRz(FiducialModel):
    """Base for the redshift-dependent SHMR normalization/slope
    (uvlf.shmr_params_at_z, anchored at z = 10)."""

    def log_ms_mean(self, mh, z, p):
        fstar, alpha = rel.shmr_params_at_z(
            10 ** p["fstar_norm"], p["alpha_star_low"], z,
            alpha_fstar_z=p.get("alpha_fstar_z", 0.0),
            alpha_star_z=p.get("alpha_star_z", 0.0))
        return np.log10(rel.ms_mh_flattening(mh, fstar_norm=fstar,
                                             alpha_star_low=alpha,
                                             M_knee=10 ** p["M_knee"]))


class FstarZ(_SHMRz):
    """Redshift-dependent SHMR normalization: f_*(z) = f_*,0 [(1+z)/11]^a."""

    name = "fstar_z"
    PARAMETERS = FIDUCIAL_PARAMETERS + (
        Parameter("alpha_fstar_z", "shmr", (-3.0, 3.0)),)


class AlphaStarZ(_SHMRz):
    """Redshift-dependent SHMR low-mass slope:
    alpha_*(z) = alpha_*,0 + a [(1+z)/11 - 1]."""

    name = "alpha_star_z"
    PARAMETERS = FIDUCIAL_PARAMETERS + (
        Parameter("alpha_star_z", "shmr", (-2.0, 2.0)),)


MODELS = {cls.name: cls for cls in (FiducialModel, SigmaSHMRz, SigmaSHMRmass,
                                    SlopeSFMS, FstarZ, AlphaStarZ)}


def get_model(name):
    try:
        return MODELS[name]()
    except KeyError:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
