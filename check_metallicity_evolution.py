"""Does a single metallicity per galaxy bias M_UV?

The model gives all stars of a galaxy one metallicity, Z = FMR(M*, SFR100, z)
at the observed redshift. Here the same SFH (get_SFH_const, no burst) is
also evaluated with each BPASS age bin at the metallicity the FMR gives at
the time those stars formed: Z_i = FMR(M*(t_i), SFR100(t_i)) + DeltaZ(z(t_i)),
with M*(t_i) the mass already formed at the bin midpoint and SFR100(t_i) the
SFH averaged over the 100 Myr before it. Bins younger than 100 Myr keep the
present-day Z unless --evolve_young. Galaxies sit on the mean SFMS and at
+-1 sigma_SFMS around it, at the Willott effective redshifts.

Also prints dM_UV for a +-0.054 / 0.1 / 0.2 dex offset in Z (FMR scatter),
for the sigma_BPASS = 0.1 mag discussion.

CANNOT BE RUN LOCALLY -- bpass_loader() needs the real spectra files.

  python check_metallicity_evolution.py [--t_star 0.25] [--sigma_sfms 0.36] [--evolve_young]
"""
import argparse

import numpy as np
import astropy.units as u
from astropy.cosmology import z_at_value

from uvlf import (SFH_sampler, bpass_loader, metalicity_from_FMR, DeltaZ_z,
                  OH_to_mass_fraction, SFMS_slope, Muv_Luv, ang_to_hz, cosmo)

L_SUN = 3.846e33


def oh_to_bpass(oh):
    """12+log(O/H) -> BPASS mass fraction, as in bpass_loader.get_UV_sfr10."""
    return OH_to_mass_fraction(oh) / 10 ** 0.42


def bin_luminosities(bp):
    """L_1500 per Msun/yr of constant SF in each age bin: (n_metal, n_age)."""
    dt = bp.ag[1:] - bp.ag[:-1]
    return np.sum(bp.SEDS[0:10, :, 1449:1549], axis=2) / 100 * dt / 1e6 * ang_to_hz


def l_uv(lum_bins, metal_grid, sfh, metals):
    """Sum over age bins with a metallicity per bin (log-log interpolation, clamped)."""
    lz = np.log10(np.clip(metals, metal_grid[0], metal_grid[-1]))
    total = 0.0
    for i, (s, m) in enumerate(zip(sfh, lz)):
        total += s * 10 ** np.interp(m, np.log10(metal_grid), np.log10(lum_bins[:, i]))
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bpass_prefix", default=None)
    ap.add_argument("--t_star", type=float, default=0.25, help="v4 main_joint median")
    ap.add_argument("--sigma_sfms", type=float, default=0.36, help="v4 main_joint median")
    ap.add_argument("--evolve_young", action="store_true",
                    help="also give bins younger than 100 Myr their formation-time Z")
    args = ap.parse_args()

    bp = bpass_loader(filename=args.bpass_prefix) if args.bpass_prefix else bpass_loader()
    metal_grid = bp.metal_avail[:10]
    lum_bins = bin_luminosities(bp)
    edges = bp.ag
    mid = np.sqrt(edges[:-1] * np.maximum(edges[1:], 1.0))
    mid[0] = 0.5 * edges[1]

    print(f"t* = {args.t_star}, sigma_SFMS = {args.sigma_sfms}, evolve_young = {args.evolve_young}")
    print("dM_UV = M_UV(Z per age bin) - M_UV(single Z); > 0 means fainter\n")
    print("   z      log M*  dlogSFR  12+log(O/H)  M_UV(single)  dM_UV   "
          "check   frac L from >100 Myr   min Z_i/Z")
    for z in (7.99, 8.99, 10.23, 11.73):
        samp = SFH_sampler(z=z)
        t_obs = cosmo.lookback_time(z).to(u.yr).value
        for log_ms in (8.0, 9.0, 10.0):
            ms = 10 ** log_ms
            for dlog in (-args.sigma_sfms, 0.0, args.sigma_sfms):
                sfr = SFMS_slope(ms, SFR_norm=args.t_star, z=z) * 10 ** dlog
                sfh, _ = samp.get_SFH_const(ms, sfr)
                n = len(sfh)
                dt = edges[1:n + 1] - edges[:n]
                oh0 = metalicity_from_FMR(ms, sfr) + DeltaZ_z(z)

                # stellar mass already in place when the stars of bin i formed (bin midpoint)
                formed = np.cumsum(sfh * dt)                 # formed since the observation, per bin
                m_mid = ms - (formed - 0.5 * sfh * dt)
                m_mid = np.maximum(m_mid, 1e-3 * ms)
                # SFR100 at the time of bin i: SFH averaged over the following 100 Myr of age
                sfr100 = np.empty(n)
                for i in range(n):
                    a0, a1 = mid[i], mid[i] + 1e8
                    w = np.clip(np.minimum(edges[1:n + 1], a1) - np.maximum(edges[:n], a0), 0, None)
                    sfr100[i] = max(np.sum(sfh * w) / 1e8, 1e-6 * sfr)
                z_i = np.array([z if a < 1e6 else
                                z_at_value(cosmo.lookback_time, (t_obs + a) * u.yr).value
                                for a in mid[:n]])
                oh_i = metalicity_from_FMR(m_mid, sfr100) + DeltaZ_z(z_i)
                if not args.evolve_young:
                    oh_i[mid[:n] < 1e8] = oh0  # young stars: present-day Z (as the model)
                oh_i = np.minimum(oh_i, oh0)   # stars never more enriched than today

                l_single = l_uv(lum_bins, metal_grid, sfh, np.full(n, oh_to_bpass(oh0)))
                l_evol = l_uv(lum_bins, metal_grid, sfh, oh_to_bpass(oh_i))
                l_code = bp.get_UV_sfr10(oh0, ms, sfr, z, SFH_samp=samp, sfr_10=sfr)
                old = edges[:n] >= 1e8
                l_old = l_uv(lum_bins, metal_grid, np.where(old, sfh, 0.0),
                             np.full(n, oh_to_bpass(oh0)))
                m_single = Muv_Luv(l_single * L_SUN)
                dm = Muv_Luv(l_evol * L_SUN) - m_single
                print(f"{z:6.2f}  {log_ms:6.1f}  {dlog:+6.2f}   {oh0:8.2f}     "
                      f"{m_single:8.2f}    {dm:+.3f}   {l_code / l_single:.3f}       "
                      f"{l_old / l_single:.3f}            {10 ** (oh_i.min() - oh0):.2f}")
        print()

    print("Metallicity offset at fixed SFH (M* = 1e9, mean SFMS, z = 10.23):")
    z = 10.23
    samp = SFH_sampler(z=z)
    ms = 1e9
    sfr = SFMS_slope(ms, SFR_norm=args.t_star, z=z)
    oh0 = metalicity_from_FMR(ms, sfr) + DeltaZ_z(z)
    m0 = Muv_Luv(bp.get_UV_sfr10(oh0, ms, sfr, z, SFH_samp=samp, sfr_10=sfr) * L_SUN)
    for d in (0.054, 0.1, 0.2):
        mp = Muv_Luv(bp.get_UV_sfr10(oh0 + d, ms, sfr, z, SFH_samp=samp, sfr_10=sfr) * L_SUN)
        mm = Muv_Luv(bp.get_UV_sfr10(oh0 - d, ms, sfr, z, SFH_samp=samp, sfr_10=sfr) * L_SUN)
        print(f"   dlogZ = +-{d:.3f}: dM_UV = {mp - m0:+.3f} / {mm - m0:+.3f} mag")


if __name__ == "__main__":
    main()
