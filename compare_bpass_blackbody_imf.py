"""
BPASS spectra vs. best-fit blackbody curves, varying IMF at fixed
stellar-population age (~10 Myr) and fixed (low) metallicity.

WHY: isolates the IMF's effect on the BPASS continuum from age/metallicity.

IMF VARIANTS -- UNVERIFIED, READ THIS FIRST: the pipeline's default is
"imf135_300" (Kroupa-like high-mass slope -1.35, upper mass cutoff 300
Msun), encoded directly in the BPASS filename:
`spectra-bin-<imf_token>.a+00.<metal_name>.dat`. Standard BPASS v2.2.1
releases also ship (from memory of BPASS's documented naming convention,
NOT verified against what's actually present on this cluster):
  - imf135_100   (same slope, upper cutoff 100 Msun instead of 300 --
                   tests whether very massive 100-300 Msun stars matter)
  - imf_chab300  (Chabrier IMF, same 300 Msun upper cutoff -- tests a
                   genuinely different functional form, not just a cutoff)
  - imf100_300   (high-mass slope -1.00, flatter/more top-heavy than the
                   default -1.35, same upper cutoff)
This script tries all three as alternates to the default and SKIPS any
that fail to load (wrong filename -> FileNotFoundError caught explicitly),
printing exactly what it tried and what worked -- if none of the
alternates load, tell me the actual available imf tokens (e.g. `ls
.../BPASS/` on the cluster for the directory the default filename lives
in) and I'll fix the list.

FIT: 912-3000A only (excludes the Lyman limit) -- see
bpass_blackbody_utils.py.

CANNOT BE RUN LOCALLY -- bpass_loader() needs the real spectra files. Run
this on the cluster.

Usage:
    python3 compare_bpass_blackbody_imf.py
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
METALLICITY = 1e-5

# (label, imf token). First one MUST match the pipeline's actual default
# (see uvlf.bpass_loader.__init__'s own default filename) so it's a
# meaningful baseline, not just another guess.
IMF_VARIANTS = [
    ("default (imf135_300)", "imf135_300"),
    ("imf135_100 (M_up=100 not 300)", "imf135_100"),
    ("imf_chab300 (Chabrier)", "imf_chab300"),
    ("imf100_300 (flatter slope, -1.00)", "imf100_300"),
]

PLOT_LAMBDA_MIN_A = 10.0
PLOT_LAMBDA_MAX_A = 3500.0


def load_bpass_for_imf(imf_token):
    """Build a bpass_loader for a given IMF token, matching the same
    cluster-path-detection logic used by every other script here, but with
    the IMF token substituted into the filename."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cluster_prefix = "/groups/astro/ivannik/programs/JWST-Inference"
    if script_dir[:10] == cluster_prefix[:10]:
        base = f"/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-{imf_token}.a+00."
    else:
        base = f"/home/inikolic/projects/stochasticity/stoc_sampler/BPASS/spectra-bin-{imf_token}.a+00."
    return bpass_loader(filename=base)


def main():
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = plt.cm.cool(np.linspace(0.1, 0.85, len(IMF_VARIANTS)))

    loaded_any = False
    for (label, imf_token), color in zip(IMF_VARIANTS, colors):
        print(f"Trying IMF '{imf_token}'...")
        try:
            bp = load_bpass_for_imf(imf_token)
        except (FileNotFoundError, OSError) as e:
            print(f"  -> could not load '{imf_token}': {e}. Skipping.")
            continue
        loaded_any = True

        wave_A = bp.wv[: bp.wv_b]
        metal_idx = int(np.argmin(np.abs(bp.metal_avail - METALLICITY)))
        age_idx = nearest_age_index(bp, TARGET_AGE_MYR * 1e6)
        age_yr = bp.ag[age_idx + 1]
        flux_raw = bp.SEDS[metal_idx, age_idx, :]

        T_fit, amp_fit, flux, _ = fit_blackbody(wave_A, flux_raw)
        if T_fit is None:
            print(f"  [{label}] fit failed, skipping")
            continue

        print(f"  loaded OK: age={age_yr/1e6:.2f} Myr, Z={bp.metal_avail[metal_idx]:.1e}, "
              f"best-fit T={T_fit:.0f} K")

        plot_mask = (wave_A >= PLOT_LAMBDA_MIN_A) & (wave_A <= PLOT_LAMBDA_MAX_A) & (flux_raw > 0)
        ax.plot(wave_A[plot_mask], flux[plot_mask], color=color, lw=2,
                label=f"{label} (T = {T_fit:,.0f} K)")
        ax.plot(
            wave_A[plot_mask],
            planck_lambda(wave_A[plot_mask], T_fit, amp_fit),
            color=color, lw=1.5, ls="--",
        )

    if not loaded_any:
        print("\nNone of the guessed IMF filenames loaded. Please check "
              "`ls` on the BPASS directory on the cluster and tell me the "
              "actual available imf tokens so I can fix IMF_VARIANTS.")
        return

    ax.axvline(LYMAN_LIMIT_A, color="gray", ls=":", lw=1)
    ax.axvline(REST_1500_A, color="gray", ls=":", lw=1)
    ax.text(LYMAN_LIMIT_A, 1.02, "912Å", color="gray", fontsize=9,
            ha="center", transform=ax.get_xaxis_transform())
    ax.text(REST_1500_A, 1.02, "1500Å", color="gray", fontsize=9,
            ha="center", transform=ax.get_xaxis_transform())

    ax.set_yscale("log")
    ax.set_ylim(1e-6, 10)
    ax.set_xlim(0, PLOT_LAMBDA_MAX_A)
    ax.set_xlabel(r"Rest-frame wavelength [$\mathrm{\AA}$]", fontsize=13)
    ax.set_ylabel(r"$L_\lambda$ (normalized to peak of fit window)", fontsize=13)
    ax.set_title(f"BPASS vs. best-fit blackbody (fit to 912-3000Å only), "
                 f"age~{TARGET_AGE_MYR:.0f} Myr, Z={METALLICITY:.0e}", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)
    plt.tight_layout()

    outpath = "bpass_vs_blackbody_imf.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()