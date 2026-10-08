"""Sanity check of bpass_loader.get_Qion_sfr10 against published BPASS values.

CANNOT BE RUN LOCALLY -- bpass_loader() needs the real spectra files.

Prints
  1. log xi_ion = Q_ion / L_UV(1500 A) [Hz erg^-1] for constant star formation
     (no burst) at several metallicities. Expected for BPASS v2 binaries with
     ~100 Myr of constant SF: log xi_ion ~ 25.4-25.6 at Z <~ 0.002, decreasing
     towards ~25.3 near solar (Stanway et al. 2016; Wilkins et al. 2016).
  2. How Q_ion and L_UV respond to the burst term sfr_10 = SFR * 10^x when it
     is added over the last 100 Myr (as in get_UV_sfr10) or the last 10 Myr.

  python check_qion_bpass.py [--bpass_prefix /path/spectra-bin-imf135_300.a+00.]
"""
import argparse

import numpy as np

from uvlf import SFH_sampler, bpass_loader, metalicity_from_FMR, DeltaZ_z

L_SUN = 3.846e33


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bpass_prefix", default=None)
    ap.add_argument("--z", type=float, default=10.0)
    args = ap.parse_args()

    bp = bpass_loader(filename=args.bpass_prefix) if args.bpass_prefix else bpass_loader()
    samp = SFH_sampler(z=args.z)
    mstar, sfr = 1e9, 3.0

    print("1. Constant SF, no burst: log xi_ion [Hz/erg]")
    for zm in (1e-4, 1e-3, 0.002, 0.004, 0.008, 0.014):
        oh = np.log10(zm * 10 ** 0.42 / 0.02) + 8.69
        q = bp.get_Qion_sfr10(oh, mstar, sfr, args.z, SFH_samp=samp, sfr_10=0.0)
        luv = bp.get_UV_sfr10(oh, mstar, sfr, args.z, SFH_samp=samp, sfr_10=0.0) * L_SUN
        print(f"   Z = {zm:7.4f}: log Q = {np.log10(q):6.2f}  log xi_ion = {np.log10(q / luv):6.3f}")

    oh = metalicity_from_FMR(mstar, sfr) + DeltaZ_z(args.z)
    print(f"\n2. Burst response at M* = 1e9, SFR = {sfr}, 12+log(O/H) = {oh:.2f}")
    q0 = bp.get_Qion_sfr10(oh, mstar, sfr, args.z, SFH_samp=samp, sfr_10=0.0)
    l0 = bp.get_UV_sfr10(oh, mstar, sfr, args.z, SFH_samp=samp, sfr_10=0.0)
    print("     x    dlogL_UV(100Myr)  dlogQ(100Myr)  dlogQ(10Myr)")
    for x in (-1.0, -0.5, 0.0, 0.5, 1.0):
        s10 = sfr * 10 ** x
        luv = bp.get_UV_sfr10(oh, mstar, sfr, args.z, SFH_samp=samp, sfr_10=s10)
        q100 = bp.get_Qion_sfr10(oh, mstar, sfr, args.z, SFH_samp=samp, sfr_10=s10)
        q10 = bp.get_Qion_sfr10(oh, mstar, sfr, args.z, SFH_samp=samp, sfr_10=s10,
                                burst_window=1e7)
        print(f"   {x:+5.1f}   {np.log10(luv / l0):+8.3f}        {np.log10(q100 / q0):+8.3f}"
              f"       {np.log10(q10 / q0):+8.3f}")


if __name__ == "__main__":
    main()
