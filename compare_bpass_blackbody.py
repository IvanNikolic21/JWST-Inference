"""
Sanity-check plot: BPASS spectra vs. best-fit blackbody curves, varying
stellar-population AGE at fixed metallicity and IMF.

WHY: recreating a plot the user made previously and can no longer find (it's
on another laptop). Rather than search further, this recomputes it from the
same `bpass_loader` class already used throughout the pipeline (uvlf.py),
so it's built on the exact same SED tables the paper's BPASS-based UV
magnitudes come from.

WHAT IT DOES: for a low metallicity (representative of the faint/high-z
galaxies this paper targets) and a few stellar-population ages, pulls the
raw single-burst SED L_lambda(lambda) directly out of `bpass_loader.SEDS`
(bypassing get_UV()'s SFH-weighted integration -- we want the intrinsic
spectral shape here, not an integrated UV luminosity), fits a Planck
blackbody curve (free amplitude + temperature) to each one over the
912-3000A window (see bpass_blackbody_utils.py -- the fit deliberately
excludes lambda<912A, the Lyman limit, since a blackbody cannot represent
the photoionization opacity edge there and including those points would
just bias the fitted temperature), and overlays both. This is meant as a
qualitative/diagnostic check of how blackbody-like (or not) the BPASS
continuum is at each age.

Companion scripts (share fitting code via bpass_blackbody_utils.py):
  - compare_bpass_blackbody_galaxies.py -- SFH-integrated, varies (M*, SFR, Z)
  - compare_bpass_blackbody_metallicity.py -- fixed age, varies metallicity
  - compare_bpass_blackbody_imf.py -- fixed age+metallicity, varies IMF

STYLED TO MATCH THE ORIGINAL "eff_temp.pdf" REFERENCE PLOT: linear (not
log) wavelength axis over 0-3500 A. The PLOT still shows lambda down to
~10A (so the Lyman-limit jump and absorption forest are visible), but the
FIT only uses 912-3000A -- plot range and fit range are intentionally
different windows now, see bpass_blackbody_utils.FIT_LAMBDA_MIN_A.

PROVENANCE / ASSUMPTIONS (verified against uvlf.py source, NOT run against
real data -- the actual BPASS spectra files live only on the cluster,
/home/inikolic/... or /groups/astro/ivannik/..., not on this machine):
  - `bpass_loader.SEDS` has shape (13 metallicities, 51 ages, wv_b
    wavelengths). Row i of the raw file is wavelength bin i; column j in
    [1,52) is the age bin. So SEDS[metal_idx, age_idx, :] is L_lambda(lambda)
    for one (Z, age) pair, NOT yet SFH-weighted.
  - Wavelength grid: `self.wv = np.linspace(1, 1e5+1, wv_b+1)` (Angstrom).
    Row i of the data corresponds to wavelength self.wv[i]. This is the one
    assumption I could NOT directly verify without the real data files; if
    the resulting plot looks shifted by ~1 Angstrom, check this first.
  - `self.ag[k+1]` is the age (yr) for SED column index k (0-indexed).

CANNOT BE RUN LOCALLY -- bpass_loader() needs the real spectra files. Run
this on the cluster.

Usage:
    python3 compare_bpass_blackbody.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uvlf import bpass_loader
from bpass_blackbody_utils import (
    planck_lambda, fit_blackbody, LYMAN_LIMIT_A, REST_1500_A,
)

# ── which (metallicity, ages) to show ───────────────────────────────────────
METALLICITY = 1e-5          # lowest available in bpass_loader.metal_avail;
                             # representative of the faint high-z population
AGE_INDICES = [0, 5, 10, 20]  # SED column indices -> ages ~1.4, ~4.5, ~14,
                               # ~141 Myr (see bpass_loader.ag) -- log-spaced
                               # to show a clear age -> temperature trend

# Plot range matches the recovered eff_temp.pdf reference (linear, 0-3500A).
# Note this is WIDER than the fit range (912-3000A, see bpass_blackbody_utils)
# -- we want to see the Lyman-limit jump even though the fit doesn't use it.
PLOT_LAMBDA_MIN_A = 10.0
PLOT_LAMBDA_MAX_A = 3500.0


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cluster_prefix = "/groups/astro/ivannik/programs/JWST-Inference"
    if script_dir[:10] == cluster_prefix[:10]:
        bp = bpass_loader(
            filename="/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00.",
        )
    else:
        bp = bpass_loader()

    metal_idx = int(np.argmin(np.abs(bp.metal_avail - METALLICITY)))
    wave_A = bp.wv[: bp.wv_b]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(AGE_INDICES)))

    for age_idx, color in zip(AGE_INDICES, colors):
        age_yr = bp.ag[age_idx + 1]
        flux_raw = bp.SEDS[metal_idx, age_idx, :]

        T_fit, amp_fit, flux, _ = fit_blackbody(wave_A, flux_raw)
        if T_fit is None:
            print(f"  [age idx {age_idx}, age={age_yr:.2e} yr] fit failed, skipping")
            continue

        label = f"age = {age_yr/1e6:.2f} Myr (best-fit T = {T_fit:,.0f} K)"
        print(f"  age idx {age_idx}: age={age_yr:.3e} yr, best-fit T={T_fit:.0f} K")

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

    ax.set_yscale("log")
    ax.set_ylim(1e-6, 10)  # explicit, not autoscaled (see git history for
                           # why that matters). Upper bound raised to 10,
                           # not 1-ish, because flux is now normalized to
                           # the 912-3000A FIT window's peak, not the
                           # global peak -- for a hot (~40,000 K) blackbody
                           # Wien's law puts the true peak below 912A
                           # (~725A), so the plotted sub-Lyman-limit region
                           # can legitimately exceed 1 on this scale.
    ax.set_xlim(0, PLOT_LAMBDA_MAX_A)
    ax.set_xlabel(r"Rest-frame wavelength [$\mathrm{\AA}$]", fontsize=13)
    ax.set_ylabel(r"$L_\lambda$ (normalized to peak of fit window)", fontsize=13)
    ax.set_title(f"BPASS (solid) vs. best-fit blackbody (dashed, fit to "
                 f"912-3000Å only), Z={bp.metal_avail[metal_idx]:.0e}",
                 fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    plt.tight_layout()

    outpath = "bpass_vs_blackbody.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()