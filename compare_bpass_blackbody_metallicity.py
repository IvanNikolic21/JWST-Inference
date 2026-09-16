"""
BPASS spectra vs. best-fit blackbody curves, varying METALLICITY at fixed
stellar-population age (~10 Myr) and fixed IMF.

WHY: isolates the metallicity effect on the BPASS continuum shape from the
age effect. compare_bpass_blackbody.py varies age at fixed (low)
metallicity, which mixes the two together; here age is held fixed so any
change across curves is attributable to metallicity alone. Spans the full
tabulated BPASS metallicity range (bp.metal_avail: 1e-5 to 0.04, ~2x solar),
not just the narrow FMR-predicted sub-solar range explored in
compare_bpass_blackbody_galaxies.py's four galaxies.

AGE: nearest tabulated SED column to TARGET_AGE_MYR (default 10 Myr) --
picked automatically via bpass_blackbody_utils.nearest_age_index rather
than hardcoding an index, since BPASS's log-spaced age grid (10^6.05,
10^6.15, ... dex) doesn't land exactly on 10 Myr. The script prints the
actual age used.

FIT: 912-3000A only (excludes the Lyman limit) -- see
bpass_blackbody_utils.py.

CANNOT BE RUN LOCALLY -- bpass_loader() needs the real spectra files. Run
this on the cluster.

Usage:
    python3 compare_bpass_blackbody_metallicity.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uvlf import bpass_loader
from bpass_blackbody_utils import (
    planck_lambda, fit_blackbody, nearest_age_index, LYMAN_LIMIT_A, REST_1500_A,
)

TARGET_AGE_MYR = 10.0

# All 13 tabulated BPASS metallicities except the very top one (0.04, ~2x
# solar -- included would work fine too, dropped just to avoid an overly
# crowded legend). Covers the full range from extremely metal-poor to
# solar-ish.
METAL_INDICES = list(range(0, 13, 2))  # every other bin: 1e-5 ... 0.03

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

    wave_A = bp.wv[: bp.wv_b]
    age_idx = nearest_age_index(bp, TARGET_AGE_MYR * 1e6)
    age_yr = bp.ag[age_idx + 1]
    print(f"Using age index {age_idx}: age={age_yr:.3e} yr ({age_yr/1e6:.2f} Myr), "
          f"target was {TARGET_AGE_MYR} Myr")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(METAL_INDICES)))

    for metal_idx, color in zip(METAL_INDICES, colors):
        Z = bp.metal_avail[metal_idx]
        flux_raw = bp.SEDS[metal_idx, age_idx, :]

        T_fit, amp_fit, flux, _ = fit_blackbody(wave_A, flux_raw)
        if T_fit is None:
            print(f"  [Z={Z:.1e}] fit failed, skipping")
            continue

        label = f"Z = {Z:.1e} (best-fit T = {T_fit:,.0f} K)"
        print(f"  Z={Z:.1e}  best-fit T={T_fit:.0f} K")

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
    ax.set_ylim(1e-6, 10)  # explicit, not autoscaled -- see compare_bpass_blackbody.py
    ax.set_xlim(0, PLOT_LAMBDA_MAX_A)
    ax.set_xlabel(r"Rest-frame wavelength [$\mathrm{\AA}$]", fontsize=13)
    ax.set_ylabel(r"$L_\lambda$ (normalized to peak of fit window)", fontsize=13)
    ax.set_title(f"BPASS vs. best-fit blackbody (fit to 912-3000Å only), "
                 f"age={age_yr/1e6:.1f} Myr", fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()

    outpath = "bpass_vs_blackbody_metallicity.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()