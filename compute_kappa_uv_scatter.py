"""
Median value and scatter of kappa_UV = SFR_100 / L_1500 for the fiducial
(MAP) model -- fills in the "? percent scatter" placeholder in the footnote
after the BPASS paragraph in Section 2.5 ("UV magnitudes") of gal_inf.tex.

WHY THIS EXISTS
----------------
kappa_UV is normally quoted as a single number (e.g. Madau & Dickinson 2014:
1.15e-28 Msun/yr/(erg/s/Hz)) derived from a single-age, single-metallicity
SSP/constant-SFH calibration. In our model it is not a constant: BPASS maps
each galaxy's own (SFH, metallicity, redshift) to an L_1500, and different
halo masses sit at different points on the mean SHMR+SFMS+FMR tracks, so
kappa_UV = SFR_100 / L_1500 varies across the modeled galaxy population even
at fixed z (and further with z, since the age of the universe sets how much
star-formation history BPASS has to draw on). This script computes that
population, at the paper's fiducial MAP parameters, and reports the median
and scatter.

WHAT COUNTS AS "THE POPULATION" -- flagged explicitly because it's a choice,
not a fact:
  - Evaluated across the full halo-mass function grid (same Tinker08 HMF /
    grid used throughout uvlf.py and mcmc.py), weighted by dn/dM with the
    same exp(-5e8 Msun / Mh) low-mass (atomic-cooling/feedback) suppression
    used everywhere else in the pipeline (see LikelihoodUVLFBase.dndm terms
    in mcmc.py) -- NOT by an arbitrary flat mass grid.
  - Evaluated at the six redshifts shown in Fig. uvlf_post (z = 6, 8, 10,
    11, 12.5, 14), pooled with equal weight per redshift, to reflect "the
    fiducial model" across the range actually presented in the paper (not
    just one z).
  - Reported both for the FULL mass grid and restricted to M_UV >= -20 (the
    paper's own bright-end cut, Sec. 2.5.1: "we restrict our analysis to
    M_UV >~ -20 where uncertainties due to dust attenuation remain
    subdominant") -- use whichever matches what the footnote is meant to
    describe.

CANNOT BE RUN LOCALLY: bpass_loader() needs the BPASS spectra files
(default path /home/inikolic/projects/stochasticity/stoc_sampler/BPASS/...,
or the /groups/astro/ivannik/... cluster path), which do not exist on this
machine. Run this on the cluster (same script_dir auto-detection as
mcmc.py/scatter_contribution_sfr10.py -- no path edits needed there).

Usage:
    python3 compute_kappa_uv_scatter.py
"""

import os

import numpy as np
import hmf as hmf
from astropy.cosmology import Planck18 as cosmo

from uvlf import (
    bpass_loader,
    SFH_sampler,
    ms_mh_flattening,
    SFMS,
    metalicity_from_FMR,
    DeltaZ_z,
    Muv_Luv,
)

# ── fiducial MAP parameters (gal_inf.tex, Results: SHMR MAP eq. and SFMS MAP eq.) ──
F_STAR_NORM_MAP = 0.03
ALPHA_STAR_MAP = 0.57
M_KNEE_MAP = 2e12          # Msun
T_STAR_MAP = 0.16          # fraction of H^-1(z)

LSUN_CGS = 3.846e33        # erg/s per Lsun, matches UV_calc_BPASS's conversion

REDSHIFTS = [6, 8, 10, 11, 12.5, 14]   # matches Fig. uvlf_post
MUV_BRIGHT_CUT = -20.0     # paper's own dust-uncertainty cut, Sec. 2.5.1


def weighted_percentile(values, weights, q):
    """q in [0,100]. Standard weighted-percentile via the cumulative weight."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return np.interp(q / 100.0, cw, v)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cluster_prefix = "/groups/astro/ivannik/programs/JWST-Inference"
    if script_dir[:10] == cluster_prefix[:10]:
        bpass_read = bpass_loader(
            filename="/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00.",
        )
    else:
        bpass_read = bpass_loader()
    vect_func = np.vectorize(bpass_read.get_UV)

    all_kappa_full, all_weight_full = [], []
    all_kappa_cut, all_weight_cut = [], []

    for z in REDSHIFTS:
        hmf_loc = hmf.MassFunction(z=z, Mmin=5, Mmax=19, dlog10m=0.05, hmf_model="Tinker08")
        masses_hmf = np.log10(hmf_loc.m / cosmo.h)
        dndm = hmf_loc.dndlog10m * cosmo.h ** 3 * np.exp(-5e8 / (hmf_loc.m / cosmo.h))

        msss = ms_mh_flattening(10 ** masses_hmf, cosmo=cosmo,
                                 alpha_star_low=ALPHA_STAR_MAP,
                                 fstar_norm=F_STAR_NORM_MAP, M_knee=M_KNEE_MAP)
        sfrs = SFMS(msss, SFR_norm=T_STAR_MAP, z=z)

        Zs = metalicity_from_FMR(msss, sfrs)
        Zs = Zs + DeltaZ_z(z)

        SFH_samp = SFH_sampler(z=z)
        F_UV = vect_func(Zs, msss, sfrs, z=z, SFH_samp=SFH_samp)   # Lsun/Hz-like
        L1500 = F_UV * LSUN_CGS                                    # erg/s/Hz
        muvs = Muv_Luv(L1500)

        kappa_uv = sfrs / L1500   # Msun/yr / (erg/s/Hz)

        finite = np.isfinite(kappa_uv) & (L1500 > 0)
        k = kappa_uv[finite]
        w = dndm[finite]
        muv_f = muvs[finite]

        med = weighted_percentile(k, w, 50)
        p16 = weighted_percentile(k, w, 16)
        p84 = weighted_percentile(k, w, 84)
        pct_scatter = 0.5 * (p84 - p16) / med * 100.0
        print(f"z={z:5.1f}  median kappa_UV={med:.3e}  "
              f"68% range=[{p16:.3e},{p84:.3e}]  ~{pct_scatter:.1f}% scatter "
              f"(full mass grid, {finite.sum()} pts)")

        all_kappa_full.append(k)
        all_weight_full.append(w)

        cut = finite & (muvs >= MUV_BRIGHT_CUT)
        all_kappa_cut.append(kappa_uv[cut])
        all_weight_cut.append(dndm[cut])

    print()
    for label, kap_list, w_list in [
        ("FULL mass grid, all z pooled", all_kappa_full, all_weight_full),
        (f"M_UV >= {MUV_BRIGHT_CUT} cut, all z pooled", all_kappa_cut, all_weight_cut),
    ]:
        k = np.concatenate(kap_list)
        w = np.concatenate(w_list)
        med = weighted_percentile(k, w, 50)
        p16 = weighted_percentile(k, w, 16)
        p84 = weighted_percentile(k, w, 84)
        pct_scatter = 0.5 * (p84 - p16) / med * 100.0
        # also report a std/mean version and a log-based (dex) version, for
        # cross-checking against whatever definition of "percent scatter"
        # was originally intended
        lin_std_pct = np.sqrt(np.average((k - med) ** 2, weights=w)) / med * 100.0
        log_dex = np.sqrt(np.average((np.log10(k) - np.log10(med)) ** 2, weights=w))
        print(f"[{label}]")
        print(f"  median kappa_UV        = {med:.3e} Msun/yr/(erg/s/Hz)")
        print(f"  68% range              = [{p16:.3e}, {p84:.3e}]")
        print(f"  scatter, (p84-p16)/2   = {pct_scatter:.1f}%")
        print(f"  scatter, weighted std  = {lin_std_pct:.1f}%")
        print(f"  scatter, dex (log10)   = {log_dex:.3f} dex "
              f"(~{log_dex * np.log(10) * 100:.1f}% if treated as lognormal)")
        print()


if __name__ == "__main__":
    main()