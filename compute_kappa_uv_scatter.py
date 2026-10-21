"""
kappa_UV = SFR_100 / L_1500 of the fiducial model, for the footnote after the
BPASS paragraph and the kappa_UV sentence in Sec. 2.5 of gal_inf.tex.

kappa_UV is not a constant in our model: BPASS maps each galaxy's SFH and
metallicity to L_1500, so it varies with M*, SFR and z. Galaxies sit on the
mean SHMR and SFMS of the posterior median (--run_dir, default v4 main_joint
medians), with no burst (SFR_10 = SFR_100), evaluated with get_UV_sfr10 as in
the inference. The population at each z is the HMF grid weighted by
dn/dlog10 M_h (with the M_turn suppression), restricted to

  observed   -22 <= M_UV <= -17 (the range of the UV LF data), and
  all        halos above M_turn = 5e8 Msun.

Printed per redshift and pooled: weighted median, 16-84% range, and the
spread across the observed M_UV range.

CANNOT BE RUN LOCALLY -- bpass_loader() needs the BPASS spectra.

  python compute_kappa_uv_scatter.py [--run_dir <main_joint>/]
"""
import argparse
import json
import os

import numpy as np
import hmf as hmf
from astropy.cosmology import Planck18 as cosmo

from cosmology_settings import HMF_COSMO_KW
from uvlf import (bpass_loader, SFH_sampler, ms_mh_flattening, SFMS, metalicity_from_FMR,
                  DeltaZ_z, Muv_Luv)

V4_MEDIAN = {"fstar_norm": -2.65, "alpha_star_low": 0.51, "t_star": 0.25, "M_knee": 12.301}
REDSHIFTS = [6.0, 7.99, 10.23, 11.73, 14.0]
MUV_RANGE = (-22.0, -17.0)
L_SUN = 3.846e33
BPASS_CLUSTER = "/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00."


def weighted_percentile(values, weights, q):
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return np.interp(np.asarray(q) / 100.0, cw / cw[-1], values[order])


def summary(label, k, w):
    p16, p50, p84 = weighted_percentile(k, w, [16, 50, 84])
    return (f"{label:22s} median {p50:.3e}  16-84% [{p16:.3e}, {p84:.3e}]  "
            f"half-width {0.5 * (p84 - p16) / p50 * 100:4.1f}%  full range [{k.min():.3e}, {k.max():.3e}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=None, help="posterior directory; default: v4 main_joint medians")
    args = ap.parse_args()

    pars = dict(V4_MEDIAN)
    if args.run_dir:
        with open(os.path.join(args.run_dir, "run_config.json")) as f:
            names = json.load(f)["params"]
        post = np.genfromtxt(os.path.join(args.run_dir, "post_equal_weights.dat"))[:, :len(names)]
        pars.update({n: v for n, v in zip(names, np.median(post, axis=0)) if n in pars})
    print("parameters:", ", ".join(f"{k}={v:.3f}" for k, v in pars.items()))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bp = bpass_loader(filename=BPASS_CLUSTER) if script_dir.startswith("/groups/astro") else bpass_loader()
    vect = np.vectorize(bp.get_UV_sfr10)

    pooled = {"observed": ([], []), "all": ([], [])}
    for z in REDSHIFTS:
        h = hmf.MassFunction(z=z, Mmin=5, Mmax=19, dlog10m=0.05, hmf_model="Tinker08", **HMF_COSMO_KW)
        mh = h.m / cosmo.h
        dn = h.dndlog10m * cosmo.h ** 3 * np.exp(-5e8 / mh)
        ms = ms_mh_flattening(mh, cosmo, fstar_norm=10 ** pars["fstar_norm"],
                              alpha_star_low=pars["alpha_star_low"], M_knee=10 ** pars["M_knee"])
        sfr = SFMS(ms, SFR_norm=pars["t_star"], z=z)
        oh = metalicity_from_FMR(ms, sfr) + DeltaZ_z(z)
        l1500 = vect(oh, ms, sfr, z, SFH_samp=SFH_sampler(z=z), sfr_10=sfr) * L_SUN
        muv = Muv_Luv(l1500)
        kappa = sfr / l1500
        ok = np.isfinite(kappa) & (l1500 > 0)
        sel = {"observed": ok & (muv >= MUV_RANGE[0]) & (muv <= MUV_RANGE[1]), "all": ok & (mh > 5e8)}
        print(f"\nz = {z:5.2f}")
        for lab, s in sel.items():
            if s.sum() < 2:
                print(f"  {lab}: fewer than two halos in range")
                continue
            print("  " + summary(lab, kappa[s], dn[s]))
            pooled[lab][0].append(kappa[s])
            pooled[lab][1].append(dn[s] / dn[s].sum())
        s = sel["observed"]
        if s.sum() >= 2:
            print(f"  kappa at M_UV = -17 / -22: {np.interp(-17, muv[s][::-1], kappa[s][::-1]):.3e} / "
                  f"{np.interp(-22, muv[s][::-1], kappa[s][::-1]):.3e}")

    print("\npooled over z (equal total weight per z, HMF-weighted within each z):")
    for lab, (k, w) in pooled.items():
        print("  " + summary(lab, np.concatenate(k), np.concatenate(w)))


if __name__ == "__main__":
    main()
