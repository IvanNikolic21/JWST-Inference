"""How much do satellites (and with them the one-halo term) contribute to the
model ACF on the scales used in the likelihood (theta > 0.005 deg)?

Evaluates the z~7 and z~9 (M* > 10^9) ACFs exactly as LikelihoodAngBase does,
once with the fiducial satellite occupation (M_0 = 10^11.65, M_1 = 10^12.3
Msun/h, alpha = 1) and once with centrals only (M_0 -> 10^18), and prints
the satellite fraction and the fractional change of w(theta) - w_IC. Also
varies M_0, M_1 by +-0.5 dex to check the claim that the satellite choice
does not matter.

CANNOT BE RUN LOCALLY -- needs halomod.

  python check_acf_satellites.py [--params v4 main_joint medians]
"""
import argparse

import numpy as np

from mcmc import LikelihoodAngBase
from effective_redshift import acf_redshift
from observations import Observations
from ulty import w_IC

_trapz = getattr(np, "trapezoid", None) or np.trapz

PARAMS = ["fstar_norm", "sigma_SHMR", "t_star", "alpha_star_low",
          "sigma_SFMS_norm", "a_sig_SFR", "M_knee", "sigma_sfr_10"]
V4_MEDIAN = [-2.65, 0.35, 0.25, 0.51, 0.36, -0.05, 12.301, 0.20]


def model_w(base, p, obs_name, data, M_0=None, M_1=None):
    """w(theta) - w_IC for parameters p, optionally overriding M_0 / M_1."""
    thet, w, sig = data
    base.call_likelihood(p, obs=obs_name, thet=thet, w=w, sig_w=sig)
    ag = base.angular_gal
    if M_0 is not None or M_1 is not None:
        hp = dict(ag.hod_params)
        if M_0 is not None:
            hp["M_0"] = M_0
        if M_1 is not None:
            hp["M1"] = M_1
        ag.hod_params = hp
    ang = np.asarray(ag.angular_corr_gal)
    m, dn = ag.m, ag.dndm
    fsat = _trapz(dn * ag.satellite_occupation, m) / _trapz(dn * ag.total_occupation, m)
    return ang - w_IC(ag.theta, ang, 41.5 / 60, 46.6 / 60), fsat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", type=float, nargs=len(PARAMS), default=V4_MEDIAN)
    args = ap.parse_args()
    p = np.array(args.params)

    obs = Observations(True, True)
    for key, z_fid, obs_name, data in [("z7", 7.0, "Ang_z7_m9", obs.get_obs_z7_m90()),
                                       ("z9", 9.25, "Ang_z9_m9", obs.get_obs_z9_m90())]:
        base = LikelihoodAngBase(PARAMS, z=acf_redshift(key, z_fid), realistic_Nz=True,
                                 fixed_Mknee=True)
        theta = np.asarray(base.angular_gal.theta) * 180 / np.pi
        w_fid, fsat = model_w(base, p, obs_name, data)
        w_cen, _ = model_w(base, p, obs_name, data, M_0=18.0)
        print(f"\n{key} (z_eff = {base.z:.2f}): satellite fraction {fsat:.4f}")
        print("   theta[deg]   w_fid       w_centrals   (w_fid - w_cen)/w_fid")
        for t in (0.001, 0.003, 0.005, 0.01, 0.03, 0.1):
            i = np.argmin(np.abs(theta - t))
            print(f"   {theta[i]:8.4f}   {w_fid[i]:10.4g}  {w_cen[i]:10.4g}   {(w_fid[i] - w_cen[i]) / w_fid[i]:+.3f}")

        thet_d = np.asarray(data[0])
        used = thet_d > 0.005
        print("   M_0, M_1 [log Msun/h] -> f_sat, max |dw/w| on the fitted data angles")
        for M_0, M_1 in [(11.15, 12.3), (12.15, 12.3), (11.65, 11.8), (11.65, 12.8)]:
            w_var, fs = model_w(base, p, obs_name, data, M_0=M_0, M_1=M_1)
            d = np.interp(thet_d[used], theta, (w_var - w_fid) / w_fid)
            print(f"   {M_0:5.2f}, {M_1:5.2f}  ->  {fs:.4f}, {np.max(np.abs(d)):.3f}")


if __name__ == "__main__":
    main()
