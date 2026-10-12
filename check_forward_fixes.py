"""How much do the 2026-10-12 forward-model fixes change the fit at fixed parameters?

CANNOT BE RUN LOCALLY -- needs halomod and the BPASS spectra.

Evaluates the main_joint likelihood terms at the best-fitting sample of an
existing run (default: the v3 main_joint), switching the two big fixes on and
off one at a time:

  fix 1 (ACF): halomod masses (Msun/h) converted to Msun inside My_HOD.
         "old" = the pre-fix HOD, which fed Msun/h straight into the SHMR.
  fix 2 (UV LF): BPASS age-bin edges [0, 10^6.05, 10^6.15, ...].
         "old" = the pre-fix edges [0, 10^6.15, 10^6.25, ...].

The smaller fixes (SFMS scatter at M*, non-compounding SFH, scatter floor,
log-log metallicity interpolation) are on in both cases.

Prints, per dataset, the model and data at each point used, ln L old vs new,
and the ACF ratio w_old / w_new.

  python check_forward_fixes.py [--run_dir <dir with post_equal_weights.dat>]
"""
import argparse
import json

import numpy as np
from astropy.cosmology import Planck18 as cosmo

import uvlf
from effective_redshift import acf_redshift, uvlf_redshift
from likelihood_terms import asymmetric_gaussian_lnl
from mcmc import LikelihoodAngBase
from observations import Observations
from ulty import My_HOD, w_IC
from cosmology_settings import HMF_COSMO_KW

OLD_EDGES = np.array([0] + [10 ** (6.05 + 0.1 * i) for i in range(1, 52)])
THETA_CUT = 0.005
FAINT = -20.0
WILLOTT = {8: "UVLF_z8_Willot23", 10: "UVLF_z10_Willot23", 12: "UVLF_z12_Willot23"}


class OldHOD(My_HOD):
    """The HOD before fix 1: masses in Msun/h used as Msun."""

    def _p_above_threshold(self, m, sigma):
        return super()._p_above_threshold(m * cosmo.h, sigma)

    @property
    def mmin(self):
        return super().mmin - np.log10(cosmo.h)


def best_sample(run_dir):
    params = json.load(open(run_dir + "run_config.json"))["params"]
    post = np.loadtxt(run_dir + "post_equal_weights.dat")
    return params, post[np.argmax(post[:, -1]), :len(params)]


def uvlf_model(z, muv, d, bp, samp):
    import hmf
    mf = hmf.MassFunction(z=z, Mmin=5, Mmax=19, dlog10m=0.05, hmf_model="Tinker08", **HMF_COSMO_KW)
    m = mf.m / cosmo.h
    phi = uvlf.UV_calc_numba_sfr10(
        muv, np.log10(m), mf.dndlog10m * cosmo.h ** 3 * np.exp(-5e8 / m),
        f_star_norm=10 ** d["fstar_norm"], alpha_star=d["alpha_star_low"],
        sigma_SHMR=d["sigma_SHMR"], sigma_SFMS_norm=d["sigma_SFMS_norm"],
        t_star=d["t_star"], a_sig_SFR=d["a_sig_SFR"], z=z,
        vect_func=np.vectorize(bp.get_UV_sfr10), SFH_samp=samp,
        M_knee=10 ** d["M_knee"], sigma_sfr10=d["sigma_sfr_10"], mass_dependent_sfr10=False,
    )
    return np.asarray(phi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir",
                    default="/lustre/astro/ivannik/UVLF_clust_runs/best_prior_v3/main_joint/")
    args = ap.parse_args()
    run_dir = args.run_dir.rstrip("/") + "/"
    params, p = best_sample(run_dir)
    d = dict(zip(params, p))
    print("Best-fitting sample of", run_dir)
    print("  " + ", ".join(f"{k}={v:.3f}" for k, v in d.items()))
    obs = Observations(True, True)

    # ---------------- fix 1: ACF ----------------
    print("\n=== Fix 1 (HOD halo-mass units), ACF")
    tot = {"old": 0.0, "new": 0.0}
    for key, name, getter in (("z9", "Ang_z9_m9", obs.get_obs_z9_m90),
                              ("z7", "Ang_z7_m9", obs.get_obs_z7_m90)):
        thet, w, sig = (np.asarray(x) for x in getter())
        ab = LikelihoodAngBase(params, realistic_Nz=True, z=acf_redshift(key, 0.0),
                               fixed_Mknee=True)
        res = {}
        for label, hod in (("new", My_HOD), ("old", OldHOD)):
            ab.angular_gal.update(hod_model=hod)
            lnl = ab.call_likelihood(p, obs=name, thet=thet, w=w, sig_w=sig)
            th = ab.angular_gal.theta * 180 / np.pi
            wm = ab.angular_gal.angular_corr_gal
            wm = wm - w_IC(ab.angular_gal.theta, wm, 41.5 / 60, 46.6 / 60)
            res[label] = (lnl, np.interp(thet, th, wm))
            tot[label] += lnl
        use = thet > THETA_CUT
        print(f"  {name}: ln L old {res['old'][0]:8.3f}  new {res['new'][0]:8.3f}")
        print("     theta[deg]   w_data +- sig     w_old     w_new   old/new")
        for t, wd, sd, wo, wn in zip(thet[use], w[use], sig[use], res["old"][1][use], res["new"][1][use]):
            print(f"     {t:9.4f}  {wd:8.4f} +- {sd:6.4f}  {wo:8.4f}  {wn:8.4f}  {wo / wn:6.3f}")
    print(f"  ACF total: ln L old {tot['old']:.3f}  new {tot['new']:.3f}  (new - old {tot['new'] - tot['old']:+.3f})")

    # ---------------- fix 2: UV LF ----------------
    print("\n=== Fix 2 (BPASS age bins), UV LF")
    bp = uvlf.bpass_loader()
    new_edges = uvlf.BPASS_AGE_EDGES.copy()
    tot = {"old": 0.0, "new": 0.0}
    for nom, name in WILLOTT.items():
        z = uvlf_redshift(name, nom)
        mo, uo, so = getattr(obs, "get_obs_uvlf_" + name[5:])()
        keep = mo >= FAINT
        res = {}
        for label, edges in (("new", new_edges), ("old", OLD_EDGES)):
            uvlf.BPASS_AGE_EDGES = edges           # read by SFH_sampler at construction
            samp = uvlf.SFH_sampler(z=z)
            bp.ag = edges.copy()
            pred = uvlf_model(z, mo[keep], d, bp, samp)
            lnl = sum(asymmetric_gaussian_lnl(pp, oo, sp, sm) for pp, oo, sp, sm
                      in zip(pred, uo[keep], so[0][keep], so[1][keep]))
            res[label] = (lnl, pred)
            tot[label] += lnl
        uvlf.BPASS_AGE_EDGES = new_edges
        bp.ag = new_edges.copy()
        print(f"  {name} (z = {z}): ln L old {res['old'][0]:8.3f}  new {res['new'][0]:8.3f}")
        print("      M_UV     phi_data   phi_old    phi_new   old/new")
        for m_, u_, po, pn in zip(mo[keep], uo[keep], res["old"][1], res["new"][1]):
            print(f"     {m_:6.2f}  {u_:9.3e}  {po:9.3e}  {pn:9.3e}  {po / pn:6.3f}")
    print(f"  UV LF total: ln L old {tot['old']:.3f}  new {tot['new']:.3f}  (new - old {tot['new'] - tot['old']:+.3f})")


if __name__ == "__main__":
    main()
