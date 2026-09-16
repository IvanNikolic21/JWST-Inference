"""
BPASS spectra vs. best-fit blackbody curves, for a few representative
GALAXIES (defined by stellar mass, SFR, metallicity) rather than raw
single-age bursts -- extension of compare_bpass_blackbody.py.

WHY THIS VERSION: the single-age-burst version answers "how blackbody-like
is a BPASS SSP at age X, metallicity Y". This version answers the more
directly relevant question for the paper: "what does the full,
SFH-integrated spectrum of an actual model galaxy at (M*, SFR, Z) look
like, and how well does a blackbody describe it" -- i.e. the same kind of
spectrum the BPASS-based UV magnitudes in gal_inf.tex (Sec. 2.5, UV_calc_BPASS
in uvlf.py) are built from, just kept at full wavelength resolution instead
of collapsed to a single 1500A luminosity.

GALAXY SELECTION: rather than picking (M*, SFR, Z) by hand, each "galaxy"
here is defined by a halo mass, and M*, SFR, Z are derived self-consistently
from the paper's own fiducial MAP relations (ms_mh_flattening + SFMS +
metalicity_from_FMR/DeltaZ_z -- same functions and same MAP values used in
compute_kappa_uv_scatter.py), at a single fiducial redshift matching the
paper's own reference figure (fig:sfms_prior is shown at z=10). Halo masses
span faint/dwarf to the SHMR knee scale, giving a genuine spread in stellar
mass, SFR, and metallicity (checked locally without BPASS data, since
ms_mh_flattening/SFMS/metalicity_from_FMR don't need it):

    Mh=3e9    -> M*=4.5e7  SFR=0.4     Msun/yr  12+log(O/H)=7.60
    Mh=3e10   -> M*=1.7e9  SFR=15      Msun/yr  12+log(O/H)=7.81
    Mh=3e11   -> M*=4.7e10 SFR=418     Msun/yr  12+log(O/H)=8.01
    Mh=2e12   -> M*=3.2e11 SFR=2790    Msun/yr  12+log(O/H)=8.12  (M_knee)

PROVENANCE / SIMPLIFICATIONS relative to the paper's actual UV_calc_BPASS
pipeline (verified against uvlf.py source, but NOT run against real BPASS
data -- see compare_bpass_blackbody.py's docstring for the same caveat):
  - SFH: built via SFH_sampler.get_SFH_const(Mstar, SFR) -- the SAME call
    get_UV() makes internally in its default (sigma_uv=True) branch, which
    is what UV_calc_BPASS actually uses.
  - Spectrum: for each age bin, weight = (SFH[age]/1e6) * (ag[age+1]-ag[age])
    -- IDENTICAL normalization to get_UV()'s internal `self.SFH /= 1e6`
    step, just kept per-wavelength (SEDS[metal_idx, age, :]) instead of
    pre-summed over the narrow 1449:1549 UV window.
  - Metallicity->BPASS bin: get_UV() does a BSpline interpolation of the
    (age-summed) UV luminosity across the 10 lowest tabulated metallicities.
    Reproducing that per-wavelength would mean re-doing the SFH-weighted sum
    once per metallicity bin then splining every wavelength -- more
    complexity than this diagnostic plot needs. Instead we just pick the
    NEAREST tabulated BPASS metallicity bin (bp.metal_avail) to the
    Strom+18-corrected FMR value.
  - Fit range: now 912-3000A (Lyman limit excluded), via
    bpass_blackbody_utils -- was 100-3000A before, which let the
    unphysical-for-a-blackbody photoionization edge bias the fit.

CANNOT BE RUN LOCALLY (bpass_loader needs the real spectra files, cluster
only). The M*/SFR/Z numbers above WERE verified locally (no BPASS data
needed for those).

Usage:
    python3 compare_bpass_blackbody_galaxies.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo

from uvlf import (
    bpass_loader, SFH_sampler, ms_mh_flattening, SFMS,
    metalicity_from_FMR, DeltaZ_z, OH_to_mass_fraction,
)
from bpass_blackbody_utils import (
    planck_lambda, fit_blackbody, LYMAN_LIMIT_A, REST_1500_A,
)

# ── fiducial MAP parameters (same as compute_kappa_uv_scatter.py) ──────────
F_STAR_NORM_MAP = 0.03
ALPHA_STAR_MAP = 0.57
M_KNEE_MAP = 2e12
T_STAR_MAP = 0.16

REDSHIFT = 10.0                              # matches fig:sfms_prior
HALO_MASSES = [3e9, 3e10, 3e11, 2e12]        # Msun; dwarf -> M_knee scale

PLOT_LAMBDA_MIN_A = 100.0
PLOT_LAMBDA_MAX_A = 1e4


def build_galaxy(Mh, z):
    """Derive (M*, SFR, 12+log(O/H)) self-consistently from the fiducial
    MAP relations, matching UV_calc_BPASS's own recipe exactly."""
    Mstar = float(ms_mh_flattening(Mh, cosmo=cosmo, fstar_norm=F_STAR_NORM_MAP,
                                    alpha_star_low=ALPHA_STAR_MAP, M_knee=M_KNEE_MAP))
    sfr = float(SFMS(Mstar, SFR_norm=T_STAR_MAP, z=z))
    oh = float(metalicity_from_FMR(Mstar, sfr) + DeltaZ_z(z))
    return Mstar, sfr, oh


def galaxy_spectrum(bp, Mstar, sfr, oh, z, sfh_sampler):
    """Full SFH-weighted L_lambda(lambda) for one galaxy, at the nearest
    tabulated BPASS metallicity bin. See module docstring for the exact
    correspondence to get_UV()'s internal recipe."""
    mass_frac = OH_to_mass_fraction(oh) / 10 ** 0.42   # matches get_UV()
    metal_idx = int(np.argmin(np.abs(bp.metal_avail - mass_frac)))

    SFH_short, _ = sfh_sampler.get_SFH_const(Mstar, sfr)
    SFH_full = np.zeros(bp.ages - 1)
    SFH_full[:len(SFH_short)] = np.array(SFH_short)
    SFH_full /= 1e6

    dt = bp.ag[1:] - bp.ag[:-1]                 # yr, shape (51,)
    weight = SFH_full * dt                       # shape (51,)
    L_lambda = np.tensordot(weight, bp.SEDS[metal_idx, :, :], axes=(0, 0))
    return L_lambda, bp.metal_avail[metal_idx]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cluster_prefix = "/groups/astro/ivannik/programs/JWST-Inference"
    if script_dir[:10] == cluster_prefix[:10]:
        bp = bpass_loader(
            filename="/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00.",
        )
    else:
        bp = bpass_loader()

    wave_A = bp.wv[: bp.wv_b]
    sfh_sampler = SFH_sampler(z=REDSHIFT)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(HALO_MASSES)))

    for Mh, color in zip(HALO_MASSES, colors):
        Mstar, sfr, oh = build_galaxy(Mh, REDSHIFT)
        flux_raw, metal_used = galaxy_spectrum(bp, Mstar, sfr, oh, REDSHIFT, sfh_sampler)

        T_fit, amp_fit, flux, _ = fit_blackbody(wave_A, flux_raw)
        if T_fit is None:
            print(f"  [Mh={Mh:.1e}] fit failed, skipping")
            continue

        label = (rf"$M_h$={Mh:.0e}, $M_\ast$={Mstar:.1e}, SFR={sfr:.1f}, "
                  rf"Z={metal_used:.1e} (T={T_fit:,.0f} K)")
        print(f"  Mh={Mh:.1e}  M*={Mstar:.3e}  SFR={sfr:.3f}  "
              f"12+log(O/H)={oh:.3f} (bin Z={metal_used:.1e})  best-fit T={T_fit:.0f} K")

        plot_mask = (wave_A >= PLOT_LAMBDA_MIN_A) & (wave_A <= PLOT_LAMBDA_MAX_A) & (flux_raw > 0)
        ax.plot(wave_A[plot_mask], flux[plot_mask], color=color, lw=2, label=label)
        ax.plot(
            wave_A[plot_mask],
            planck_lambda(wave_A[plot_mask], T_fit, amp_fit),
            color=color, lw=1.5, ls="--",
        )

    ax.axvline(LYMAN_LIMIT_A, color="gray", ls=":", lw=1)
    ax.axvline(REST_1500_A, color="gray", ls=":", lw=1)
    ax.text(LYMAN_LIMIT_A, 1.02, "912Å", color="gray", fontsize=9,
            ha="center", transform=ax.get_xaxis_transform())
    ax.text(REST_1500_A, 1.02, "1500Å", color="gray", fontsize=9,
            ha="center", transform=ax.get_xaxis_transform())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 10)  # explicit, not autoscaled; see compare_bpass_blackbody.py
    ax.set_xlabel(r"Rest-frame wavelength [$\mathrm{\AA}$]", fontsize=13)
    ax.set_ylabel(r"$L_\lambda$ (normalized to peak of fit window)", fontsize=13)
    ax.set_title(f"SFH-integrated BPASS spectra (solid) vs. best-fit blackbody "
                 f"(dashed, fit to 912-3000Å only), z={REDSHIFT:.0f}", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    plt.tight_layout()

    outpath = "bpass_vs_blackbody_galaxies.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()