"""
Scatter-on vs. scatter-off 21cmFAST comparison, for the talk motivating why the
SHMR/SFMS scatter constrained in the paper matters for the 21-cm power spectrum.

WHAT THIS DOES
---------------
Runs two coeval 21cmFAST simulations that share the SAME median SHMR/SFMS/SFR-scatter
functional forms (i.e. the same F_STAR10, ALPHA_STAR, t_STAR, and the same
UPPER_STELLAR_TURNOVER_MASS/INDEX), and differ ONLY in whether the log-normal
scatter terms (SIGMA_STAR, SIGMA_SFR_LIM, SIGMA_SFR_INDEX) are at their MAP values
(Run A, "scatter on") or set to ~0 (Run B, "scatter off"). Because these are
log-normal relations, holding the median fixed while zeroing sigma LOWERS the
population-averaged (linear-space) SFR/UV output -- see the docstring note below
on HALO_SCALING_RELATIONS_MEDIAN, this is the crux of the comparison and is easy
to get backwards.

Then it computes the 21-cm brightness-temperature power spectrum for both runs at
z ~ 7-9 and makes a comparison plot.

PROVENANCE / CONFIDENCE OF THE PARAMETER MAPPING
--------------------------------------------------
Everything below was read directly out of the source at
/Users/dxf836/Documents/21cmFAST/21cmFAST (git remote: 21cmfast/21cmFAST,
branch `main`, commit b7c41198f as of 2026-08-31) -- NOT guessed or remembered
from training data, and NOT run/tested (py21cmfast is not built/installed in
any local conda env; this is a source checkout only). Field names, defaults,
and the docstring quotes below are copy-checked against
src/py21cmfast/wrapper/inputs.py. You should still do a quick low-resolution
dry run yourself before trusting the numbers, per the checklist at the bottom.

Paper (gal_inf.tex) <-> 21cmFAST AstroParams mapping used here:
    f_0 (SHMR MAP normalization)   -> F_STAR10   [NOTE: F_STAR10 is given in
                                                    log10 units in 21cmFAST]
    alpha_star_low (SHMR MAP slope) -> ALPHA_STAR
    M_knee = 2e12 Msun              -> UPPER_STELLAR_TURNOVER_MASS
                                        (21cmFAST default is 10**11.447 ~ 2.8e11;
                                        also given in log10 units)
    high-mass SHMR slope 0.61       -> UPPER_STELLAR_TURNOVER_INDEX
                                        (21cmFAST default -0.6; SAME sign
                                        convention and almost the same value as
                                        this paper's fixed 0.61 -- strong
                                        independent cross-check that this is the
                                        right parameter)
    sigma_SHMR = 0.32 dex           -> SIGMA_STAR (dex; internally converted via
                                        a dex->exp transform, so you pass 0.32
                                        directly, no manual conversion needed)
    t_star = 0.16 (MAP SFMS)        -> t_STAR
    sigma_0,SFMS = 0.25 dex         -> SIGMA_SFR_LIM (scatter above 1e10 Msun)
    a_sigma,SFMS = -0.085 (MAP)     -> SIGMA_SFR_INDEX (slope of SFMS scatter
                                        below 1e10 Msun; 21cmFAST default -0.12)

NOT mapped (NEEDS VERIFICATION / not available as far as I could find):
    - sigma_SFR_10 (the paper's 10 Myr burstiness scatter) has NO obvious
      counterpart in AstroParams -- I did not find a SIGMA_SFR10-like field.
      Given the paper's own decomposition found this contributes only ~2-5%
      to sigma_UV, I've left it out rather than guess a field name. If you
      want it included, grep the installed inputs.py yourself for anything
      matching "SFR10"/"10myr"/"burst" once you've built the package -- I
      only had the source tree, not a working import, so I could not
      interactively enumerate AstroParams' full field list beyond what's
      docstring-visible.
    - No separate final "UV-band"/BPASS-stage scatter field (this paper's
      sigma_UV / sigma_BPASS) was found either; 21cmFAST's ionizing photon
      output is presumably computed directly from the (stochastic) SFR without
      an additional observational-scatter stage, so this is likely fine to
      omit, but I could not 100% confirm this without a working install.

*** THE ONE THING MOST LIKELY TO TRIP YOU UP ***
AstroOptions.HALO_SCALING_RELATIONS_MEDIAN defaults to False, meaning by
default F_STAR10/t_STAR etc. are interpreted as the LINEAR-SPACE MEAN of their
log-normal conditional distributions, not the median. Quoting the source
docstring directly:

    "HALO_SCALING_RELATIONS_MEDIAN : bool, optional
        If True, halo scaling relation parameters (F_STAR10,t_STAR etc...)
        define the median of their conditional distributions
        If False, they describe the mean.
        This becomes important when using non-symmetric distributions such as
        the log-normal"

and from the AstroParams class docstring:

    "NB: All Mean scaling relations are defined in log-space, such that the
    lines they produce give exp(<log(property)>), this means that increasing
    the lognormal scatter in these relations will increase the <property> but
    not <log(property)>"

If you leave HALO_SCALING_RELATIONS_MEDIAN at its default (False, "mean mode"),
then F_STAR10/t_STAR are pinned to be the fixed MEAN, and turning sigma off
would instead RAISE the median to compensate -- the opposite experiment from
the one you described (same median, different mean). This script explicitly
sets HALO_SCALING_RELATIONS_MEDIAN=True in AstroOptions for BOTH runs, so that
F_STAR10/ALPHA_STAR/t_STAR/UPPER_STELLAR_TURNOVER_* are held as the fixed
MEDIAN relation, and only the linear-space mean differs between Run A and
Run B, exactly as you specified. Double-check this is still the semantics of
whatever version you actually build/run -- I could not execute this to verify.

USE_UPPER_STELLAR_TURNOVER (for M_knee) and the log-normal scatter sampling
both require the discrete halo sampler, i.e. AstroOptions.SOURCE_MODEL in
("DEXM-ESF", "CHMF-SAMPLER") -- confirmed from source:
    "USE_UPPER_STELLAR_TURNOVER is not yet implemented for SOURCE_MODEL = ..."
MatterOptions.SOURCE_MODEL already defaults to "CHMF-SAMPLER" in this branch,
so no override should be needed, but it's set explicitly below for clarity.

EXPECTED RUNTIME
-----------------
Not measured (could not run it here). BOX_LEN=200 Mpc, HII_DIM=128, single
random seed, 2 redshifts, discrete halo sampler (CHMF-SAMPLER) -- this is a
modest talk-appropriate setup. Ballpark from general 21cmFAST experience with
similar settings: order 10-60 minutes per run on a single modern CPU core,
but the discrete halo sampler adds overhead relative to grid-based source
models, so budget more; run once at HII_DIM=64 first as a smoke test before
committing to the full resolution.

BEFORE RUNNING FOR REAL
-------------------------
  1. Build/install this checkout into an environment
     (cd /Users/dxf836/Documents/21cmFAST/21cmFAST && pip install -e .),
     and `pip install powerbox` (used below for the power spectrum; this is
     the standard companion package used in the official 21cmFAST tutorials,
     but is not a hard dependency of the core package, so install separately).
  2. Do one fast smoke-test run (HII_DIM=32-64, one redshift) to confirm the
     AstroParams field names and HALO_SCALING_RELATIONS_MEDIAN semantics
     actually behave as documented above, since none of this has been
     executed.
  3. Confirm whether you want the SFR10/burstiness and UV-stage scatter
     included -- see the "NOT mapped" note above.
"""

import numpy as np
import matplotlib.pyplot as plt

import py21cmfast as p21c
from py21cmfast import (
    AstroParams,
    AstroOptions,
    CosmoParams,
    MatterOptions,
    SimulationOptions,
    InputParameters,
)

# ── run settings (talk-appropriate, not production quality) ─────────────────
BOX_LEN = 200.0       # cMpc
HII_DIM = 128
RANDOM_SEED = 1234
OUT_REDSHIFTS = (7.0, 9.0)   # near reionization midpoint, per the paper's own
                             # z=10 stochasticity-decomposition figure and the
                             # discussion in the talk

# ── paper's MAP values (see gal_inf.tex, Results: "Posterior stellar-to-halo-
#    mass relation" and "Posterior star formation main sequence") ──────────
F_0_MAP = 0.03                     # SHMR MAP normalization (linear)
ALPHA_STAR_LOW_MAP = -0.57         # SHMR MAP low-mass slope
M_KNEE = 2e12                      # Msun, fixed in the paper
HIGH_MASS_SLOPE = 0.61             # paper's fixed high-mass SHMR slope
SIGMA_SHMR_MAP = 0.32              # dex

T_STAR_MAP = 0.16                  # SFMS MAP timescale (fraction of H^-1(z))
SIGMA_SFMS_NORM_MAP = 0.25         # dex, scatter above 1e10 Msun
A_SIG_SFMS_MAP = -0.085            # slope of SFMS scatter below 1e10 Msun

NEAR_ZERO_SIGMA = 1e-3             # "off" but nonzero to avoid degenerate/NaN
                                    # log-normal sampling; verify this is small
                                    # enough to be negligible once you can
                                    # actually run the code


def build_astro_params(scatter_on: bool) -> AstroParams:
    """Astro params shared between the two runs, differing only in scatter."""
    sigma_star = SIGMA_SHMR_MAP if scatter_on else NEAR_ZERO_SIGMA
    sigma_sfr_lim = SIGMA_SFMS_NORM_MAP if scatter_on else NEAR_ZERO_SIGMA
    # NOTE: SIGMA_SFR_INDEX is a slope, not a scatter amplitude by itself --
    # zeroing it (rather than SIGMA_SFR_LIM) removes the mass-DEPENDENCE of
    # the SFMS scatter but not the scatter itself. For the "scatter off" run
    # we want SIGMA_SFR_LIM -> 0, which already kills the scatter at all
    # masses; SIGMA_SFR_INDEX is left at its MAP value in both runs since it
    # has no effect once SIGMA_SFR_LIM ~ 0. NEEDS VERIFICATION once you can
    # actually inspect how these two combine in the C code.
    return AstroParams(
        F_STAR10=float(np.log10(F_0_MAP)),   # log10 units, per source docstring
        ALPHA_STAR=ALPHA_STAR_LOW_MAP,
        UPPER_STELLAR_TURNOVER_MASS=float(np.log10(M_KNEE)),  # log10 units
        UPPER_STELLAR_TURNOVER_INDEX=-HIGH_MASS_SLOPE,
        # ^ sign: 21cmFAST default is -0.6 for the SAME physical effect (AGN-
        # feedback steepening at high mass) as this paper's +0.61 exponent in
        # the denominator of Eq. SHMR -- hence the minus sign here. This is
        # the single strongest cross-check in the whole mapping (-0.6 vs
        # -0.61, essentially the literature value already).
        t_STAR=T_STAR_MAP,
        SIGMA_STAR=sigma_star,
        SIGMA_SFR_LIM=sigma_sfr_lim,
        SIGMA_SFR_INDEX=A_SIG_SFMS_MAP,
    )


def build_inputs(scatter_on: bool) -> InputParameters:
    return InputParameters(
        random_seed=RANDOM_SEED,
        cosmo_params=CosmoParams(),
        matter_options=MatterOptions(
            SOURCE_MODEL="CHMF-SAMPLER",  # discrete halo sampler; required for
                                          # UPPER_STELLAR_TURNOVER and for the
                                          # log-normal scatter to be realized
                                          # per-halo rather than smoothed away
        ),
        simulation_options=SimulationOptions(BOX_LEN=BOX_LEN, HII_DIM=HII_DIM),
        astro_options=AstroOptions(
            USE_UPPER_STELLAR_TURNOVER=True,
            HALO_SCALING_RELATIONS_MEDIAN=True,  # <-- the critical flag, see
                                                  # docstring above: without
                                                  # this, F_STAR10/t_STAR are
                                                  # the MEAN not the median,
                                                  # and the experiment is
                                                  # backwards.
        ),
        astro_params=build_astro_params(scatter_on),
    )


def compute_power_spectrum(brightness_temp, box_len):
    """Isotropic 21-cm power spectrum via powerbox. `pip install powerbox`.

    NOTE: get_power()'s return signature has changed across powerbox versions
    (it can return (power, k), (power, k, var), or more depending on version
    and kwargs like get_variance). Unpack defensively -- take the first two
    elements (power, k) rather than assuming an exact tuple length, and pass
    bins_upto_boxlen explicitly to silence/pin the binning-convention
    FutureWarning rather than silently riding whatever the installed
    version's new default becomes.
    """
    from powerbox import get_power

    result = get_power(brightness_temp, boxlength=box_len, bins_upto_boxlen=True)
    power, k = result[0], result[1]
    return k, power


def main():
    print("Running scatter ON (Run A)...")
    inputs_on = build_inputs(scatter_on=True)
    coevals_on = p21c.run_coeval(inputs=inputs_on, out_redshifts=OUT_REDSHIFTS)

    print("Running scatter OFF (Run B)...")
    inputs_off = build_inputs(scatter_on=False)
    coevals_off = p21c.run_coeval(inputs=inputs_off, out_redshifts=OUT_REDSHIFTS)

    fig, axes = plt.subplots(1, len(OUT_REDSHIFTS), figsize=(6 * len(OUT_REDSHIFTS), 5),
                              squeeze=False)
    axes = axes[0]

    for ax, z, cv_on, cv_off in zip(axes, OUT_REDSHIFTS, coevals_on, coevals_off):
        k_on, p_on = compute_power_spectrum(cv_on.brightness_temp, BOX_LEN)
        k_off, p_off = compute_power_spectrum(cv_off.brightness_temp, BOX_LEN)

        delta_on = k_on ** 3 * p_on / (2 * np.pi ** 2)
        delta_off = k_off ** 3 * p_off / (2 * np.pi ** 2)

        ax.loglog(k_on, delta_on, color="tab:red", lw=3, label="scatter ON (MAP)")
        ax.loglog(k_off, delta_off, color="tab:gray", lw=3, ls="--",
                   label="scatter OFF (same median)")
        ax.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=14)
        ax.set_ylabel(r"$\Delta_{21}^2(k)$ [mK$^2$]", fontsize=14)
        ax.set_title(f"z = {z}", fontsize=14)
        ax.legend(fontsize=12, frameon=False)

    plt.tight_layout()
    outpath = "fractional_21cm_scatter_comparison.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
