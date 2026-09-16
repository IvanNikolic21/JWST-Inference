"""
Sanity-check plot: BPASS spectra vs. best-fit blackbody curves.

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
blackbody curve (free amplitude + temperature) to each one, and overlays
both. This is meant as a qualitative/diagnostic check of how blackbody-like
(or not) the BPASS continuum is at each age -- departures indicate spectral
features (line blanketing, the Balmer/Lyman jumps, nebular-free stellar
photosphere effects) a pure blackbody can't capture.

STYLED TO MATCH THE ORIGINAL "eff_temp.pdf" REFERENCE PLOT: linear (not
log) wavelength axis over 0-3500 A, single ages (not SFH-integrated -- see
compare_bpass_blackbody_galaxies.py for the SFH-integrated version). The
recovered original showed a much sharper Lyman-limit (912A) discontinuity
and stronger blackbody departure than the SFH-integrated version did; a
linear x-axis stretches the 200-1000A region where that structure lives,
and single-age bursts don't blend a young population's sharp opacity break
with older, redder stars the way SFH-integration does -- both contribute to
why the SFH-integrated plot looked "washed out" by comparison.

PROVENANCE / ASSUMPTIONS (verified against uvlf.py source, NOT run against
real data -- the actual BPASS spectra files live only on the cluster,
/home/inikolic/... or /groups/astro/ivannik/..., not on this machine):
  - `bpass_loader.SEDS` has shape (13 metallicities, 51 ages, wv_b
    wavelengths). Row i of the raw file is wavelength bin i; column j in
    [1,52) is the age bin. So SEDS[metal_idx, age_idx, :] is L_lambda(lambda)
    for one (Z, age) pair, NOT yet SFH-weighted.
  - Wavelength grid: `self.wv = np.linspace(1, 1e5+1, wv_b+1)` (Angstrom).
    Row i of the data corresponds to wavelength self.wv[i] -- i.e. the
    standard BPASS 1-Angstrom-spaced grid from 1 to 1e5 A. This is the one
    assumption I could NOT directly verify without the real data files
    (there's a plausible off-by-one between self.wv[i] and self.wv[i+1]);
    if the resulting plot looks shifted by ~1 Angstrom, that's the place to
    check first -- it won't matter for the qualitative blackbody comparison
    either way.
  - `self.ag[k+1]` is the age (yr) for SED column index k (0-indexed) --
    see bpass_loader.__init__: self.ag = [0] + [10**(6.05+0.1*i) for i in
    range(1,52)], and SEDS' 51 columns correspond to i=1..51.
  - SED units are BPASS's native L_lambda per unit starburst mass (Lsun/A
    per 1e6 Msun formed, standard BPASS convention) -- since we fit a
    free amplitude, the exact absolute units don't matter for the shape
    comparison, only the *relative* spectral shape.

CANNOT BE RUN LOCALLY -- same reason as compute_kappa_uv_scatter.py and
run_21cmfast_scatter_comparison.py: bpass_loader() needs the real spectra
files. Run this on the cluster.

Usage:
    python3 compare_bpass_blackbody.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy import constants as const
from astropy import units as u

from uvlf import bpass_loader

# ── which (metallicity, ages) to show ───────────────────────────────────────
METALLICITY = 1e-5          # lowest available in bpass_loader.metal_avail;
                             # representative of the faint high-z population
AGE_INDICES = [0, 5, 10, 20]  # SED column indices -> ages ~1.4, ~4.5, ~14,
                               # ~141 Myr (see bpass_loader.ag) -- log-spaced
                               # to show a clear age -> temperature trend

# Wavelength range used for the fit -- avoid lambda->0 (numerical blow-up)
# and the very far-IR tail (irrelevant for a hot-star blackbody check).
FIT_LAMBDA_MIN_A = 100.0
FIT_LAMBDA_MAX_A = 3000.0

# Plot range matches the recovered eff_temp.pdf reference (linear, 0-3500A).
PLOT_LAMBDA_MAX_A = 3500.0

LYMAN_LIMIT_A = 912.0
REST_1500_A = 1500.0


def planck_lambda(wave_A, T, amplitude):
    """Planck function B_lambda(T), free overall amplitude (arbitrary units
    -- BPASS's native units don't match B_lambda's cgs units, so we only
    fit the shape via T and let `amplitude` absorb the normalization)."""
    wave = (wave_A * u.AA).to(u.m).value
    h = const.h.value
    c = const.c.value
    k = const.k_B.value
    with np.errstate(over="ignore", divide="ignore"):
        x = h * c / (wave * k * T)
        bb = (2 * h * c**2 / wave**5) / (np.expm1(x))
    return amplitude * bb


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

        fit_mask = (
            (wave_A >= FIT_LAMBDA_MIN_A)
            & (wave_A <= FIT_LAMBDA_MAX_A)
            & (flux_raw > 0)
        )
        # Fit in flux-normalized units (peak of the fit window -> 1). Fitting
        # directly in BPASS's native units against Planck_lambda's SI-scale
        # output badly scales the two free parameters (amplitude ends up
        # ~1e50-ish to compensate) and the least-squares solver never
        # converges away from the initial guess -- caught by smoke-testing
        # with a mock SED before handing this off, since the real BPASS
        # files aren't available locally to test against directly.
        norm = flux_raw[fit_mask].max()
        flux = flux_raw / norm

        p0 = [4e4, flux[fit_mask].max() / planck_lambda(REST_1500_A, 4e4, 1.0)]
        try:
            popt, _ = curve_fit(
                planck_lambda, wave_A[fit_mask], flux[fit_mask],
                p0=p0, bounds=([1e3, 0], [3e5, np.inf]), maxfev=10000,
            )
            T_fit, amp_fit = popt
        except RuntimeError:
            print(f"  [age idx {age_idx}, age={age_yr:.2e} yr] fit failed, skipping")
            continue

        label = f"age = {age_yr/1e6:.2f} Myr (best-fit T = {T_fit:,.0f} K)"
        print(f"  age idx {age_idx}: age={age_yr:.3e} yr, best-fit T={T_fit:.0f} K")

        plot_mask = (wave_A >= 0) & (wave_A <= PLOT_LAMBDA_MAX_A) & (flux_raw > 0)
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
    ax.set_xlim(0, PLOT_LAMBDA_MAX_A)
    ax.set_xlabel(r"Rest-frame wavelength [$\mathrm{\AA}$]", fontsize=13)
    ax.set_ylabel(r"$L_\lambda$ (normalized to peak of fit window)", fontsize=13)
    ax.set_title(f"BPASS (solid) vs. best-fit blackbody (dashed), Z={bp.metal_avail[metal_idx]:.0e}",
                 fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    plt.tight_layout()

    outpath = "bpass_vs_blackbody.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()