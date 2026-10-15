"""Independent check of the ACF likelihood model (likelihood validation, step 2b).

For one parameter set (MAP or median of a run), compares the w(theta) that
mcmc.LikelihoodAngBase feeds into the likelihood (halomod AngularCF + My_HOD,
projected over the Paquereau N(z), minus the integral constraint) with
independent calculations:

  A  halomod angular_corr_gal (what the likelihood uses)
  B  my Limber projection (fine z grid, xi-space) of halomod's own xi_gg(r);
     beyond halomod's rmax the 2-halo term b_eff^2 xi_mm is appended.
     A vs B tests the projection: N(z) normalisation, z grid, units, rmax.
  C  b_eff^2 w_mm(theta) from scratch: own HOD integral over the hmf mass
     function, Tinker+10 bias formula, k-space Limber with J0, linear and
     halofit P(k), all at the fixed model redshift as halomod does.
     A vs C tests the HOD -> bias -> clustering amplitude chain (h units etc.).
  D  as C (linear), but with b_eff(z) and D(z) evolving across N(z): tests the
     single-redshift (z_eff) approximation of the likelihood.

Also compares n_gal, b_eff and N_cen / N_sat with halomod's, and gives the
likelihood (chi^2) of the data for every variant, after the same
rectangle integral constraint.

Cluster only (needs halomod). Run, e.g.:
  python check_acf_limber.py --run_dir <best_prior_v4/main_joint> --which map
"""
import argparse
import json
import os

import numpy as np
import hmf
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from scipy.special import erfc, j0

from cosmology_settings import HMF_COSMO_KW
from effective_redshift import acf_redshift
from likelihood_terms import integral_constraint_rectangle
from mcmc import LikelihoodAngBase
from observations import Observations
from uvlf import ms_mh_flattening, shmr_params_at_z

trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2.0 on the cluster
H = cosmo.h
C_KMS = const.c.to("km/s").value
DELTA_C = 1.686
FIELD_DEG = (41.5 / 60, 46.6 / 60)      # rectangle used by mcmc.py for the IC
THETA_CUT_DEG = 0.005                    # mcmc.py: only theta > 0.005 deg is fitted
THETA_MAX_RAD = 0.03                     # beyond the field diagonal (0.019 rad), enough for the IC
ACF_SETS = {                             # obs key -> (z key, M* threshold, data getter)
    "Ang_z7_m9": ("z7", 9.0, "get_obs_z7_m90"),
    "Ang_z9_m9": ("z9", 9.0, "get_obs_z9_m90"),
}
# satellite HOD defaults hard-wired in LikelihoodAngBase.call_likelihood
M_0_SAT, M1_SAT, ALPHA_SAT = 11.65, 12.3, 1.0


def pick_params(run_dir, which):
    names = list(json.load(open(os.path.join(run_dir, "priors_params.json"))))
    post = np.loadtxt(os.path.join(run_dir, "post_equal_weights.dat"))
    if which == "map":
        row = post[np.argmax(post[:, -1]), :len(names)]
    else:
        row = np.median(post[:, :len(names)], axis=0)
    return names, row


def tinker10_bias(sigma, delta_halo=200.0):
    """Tinker et al. (2010) Eq. 6 with Table 2, nu = delta_c / sigma."""
    y = np.log10(delta_halo)
    A = 1.0 + 0.24 * y * np.exp(-(4.0 / y) ** 4)
    a = 0.44 * y - 0.88
    B, b = 0.183, 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4.0 / y) ** 4)
    c = 2.4
    nu = DELTA_C / sigma
    return 1.0 - A * nu ** a / (nu ** a + DELTA_C ** a) + B * nu ** b + C * nu ** c


def occupation(m_h, fstar_lin, alpha_star, M_knee, sigma_shmr, ms_min):
    """N_cen and N_sat as functions of halo mass m_h [Msun/h], written from the
    model definition: P(M* > ms_min | Mh) for a lognormal SHMR truncated at
    the baryon fraction, and the satellite term of call_likelihood."""
    m = m_h / H
    mu = np.log10(ms_mh_flattening(m, cosmo, fstar_norm=fstar_lin,
                                   alpha_star_low=alpha_star, M_knee=M_knee))
    top = np.log10(cosmo.Ob0 / cosmo.Om0 * m)
    s2 = sigma_shmr * np.sqrt(2.0)
    cdf_top = 0.5 * erfc(-(top - mu) / s2)
    cdf_min = 0.5 * erfc(-(ms_min - mu) / s2)
    n_cen = np.maximum(0.0, (cdf_top - cdf_min) / np.maximum(cdf_top, 1e-300))
    n_sat = np.zeros_like(m_h)
    sat = m_h > 10 ** M_0_SAT
    n_sat[sat] = n_cen[sat] * ((m_h[sat] - 10 ** M_0_SAT) / 10 ** M1_SAT) ** ALPHA_SAT
    return n_cen, n_sat


def mass_function(z, hmf_model):
    return hmf.MassFunction(z=z, Mmin=8, Mmax=16, dlog10m=0.01, hmf_model=hmf_model,
                            transfer_model="EH", **HMF_COSMO_KW)


def ngal_beff(mf, hod_args):
    """n_gal [h^3/Mpc^3] and b_eff of centrals+satellites, from mf (hmf)."""
    n_cen, n_sat = occupation(mf.m, *hod_args)
    n_tot = n_cen + n_sat
    bias = tinker10_bias(mf.sigma)
    lnm = np.log(mf.m)
    ngal = trapz(mf.dndlnm * n_tot, lnm)
    beff = trapz(mf.dndlnm * n_tot * bias, lnm) / ngal
    return ngal, beff


def chi_h(z):
    """Comoving distance [Mpc/h]."""
    return cosmo.comoving_distance(z).value * H


def dz_dchi(z):
    """dz/dchi [h/Mpc]."""
    return cosmo.H(z).value / C_KMS / H


def nz_grid(p1, n=1201, nout=300):
    z = np.linspace(0.0, 15.0, n)
    p = np.asarray(p1(z), dtype=float)
    keep = p > 1e-6 * p.max()
    lo, hi = np.argmax(keep), len(z) - np.argmax(keep[::-1])
    z = np.linspace(z[max(lo - 1, 0)], z[min(hi, n - 1)], nout)
    p = np.asarray(p1(z), dtype=float)
    return z, p / trapz(p, z)


def limber_k(theta, z, pz, k, pk_of_z):
    """w(theta) = int dz p(z)^2 dz/dchi int dk k/(2 pi) P(k, z) J0(k chi theta).
    pk_of_z(i) returns P on k for the i-th z (Mpc/h units)."""
    lnk = np.log(k)
    w = np.zeros_like(theta)
    weight = pz ** 2 * dz_dchi(z)
    chi = chi_h(z)
    for i in range(len(z)):
        if weight[i] == 0:
            continue
        pk = pk_of_z(i)
        integrand = (k ** 2 * pk / (2 * np.pi))[None, :] * j0(np.outer(theta * chi[i], k))
        w += weight[i] * trapz(integrand, lnk, axis=1)
    return w * (z[1] - z[0])


def limber_xi(theta, z, pz, xi_fn):
    """w(theta) = int dz p(z)^2 dz/dchi 2 int_0^inf du xi(sqrt(u^2 + (chi theta)^2))."""
    u = np.concatenate([[0.0], np.logspace(-4, np.log10(400.0), 1500)])
    chi = chi_h(z)
    weight = pz ** 2 * dz_dchi(z)
    w = np.zeros_like(theta)
    for i in range(len(z)):
        rp = theta * chi[i]
        r = np.sqrt(u[None, :] ** 2 + rp[:, None] ** 2)
        w += weight[i] * 2.0 * trapz(xi_fn(r), u, axis=1)
    return w * (z[1] - z[0])


def xi_from_pk(r, k, pk):
    """xi(r) = 1/(2 pi^2) int dk k^2 P(k) sin(kr)/(kr), Gaussian-damped at k > 10 h/Mpc."""
    lnk = np.log(k)
    damp = np.exp(-(k / 10.0) ** 2)
    out = np.empty_like(r)
    for i, ri in enumerate(r):
        out[i] = trapz(k ** 3 * pk * damp * np.sinc(k * ri / np.pi), lnk) / (2 * np.pi ** 2)
    return out


def chi2(w_model_at_data, w, sig):
    return float(np.sum(((w_model_at_data - w) / sig) ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--which", choices=["map", "median"], default="map")
    ap.add_argument("--out", default=None, help="npz output (default <run_dir>/check_acf_<which>.npz)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "run_config.json")))
    names, row = pick_params(args.run_dir, args.which)
    dic = dict(zip(names, row))
    hmf_model = cfg.get("hmf", "Tinker08")
    print(f"run {args.run_dir}  ({args.which})")
    print("  " + ", ".join(f"{n}={v:.4f}" for n, v in dic.items()))

    obs = Observations(True, True)
    k = np.logspace(-4, np.log10(30.0), 2 ** 15)      # h/Mpc, for the J0 Limber integrals
    k_xi = np.logspace(-4, 1.5, 2 ** 17)               # h/Mpc, for xi(r) at r up to 450 Mpc/h
    results = {}
    for key, (zkey, ms_min, getter) in ACF_SETS.items():
        z_eff = acf_redshift(zkey, None)
        base = LikelihoodAngBase(names, realistic_Nz=True, hmf_choice=hmf_model, z=z_eff,
                                 exact_specs=True, fixed_Mknee=cfg.get("fixed_Mknee", False))
        thet, w_obs, sig_obs = (np.asarray(a, dtype=float) for a in getattr(obs, getter)())
        lnl_lik = base.call_likelihood(row, obs=key, thet=thet, w=w_obs, sig_w=sig_obs)
        ang = base.angular_gal
        p1 = base.p1_z7 if zkey == "z7" else base.p1

        fstar_lin, alpha_star = shmr_params_at_z(
            10 ** dic["fstar_norm"], dic["alpha_star_low"], z_eff,
            alpha_fstar_z=dic.get("alpha_fstar_z", 0.0), alpha_star_z=dic.get("alpha_star_z", 0.0))
        M_knee = 10 ** dic["M_knee"] if "M_knee" in dic else 2.6e11
        hod_args = (fstar_lin, alpha_star, M_knee, dic["sigma_SHMR"], ms_min)
        if dic.get("a_sig_SHMR", 0.0) != 0.0 or cfg.get("sigma_shmr_z_dependent"):
            print("  WARNING: SHMR-scatter extensions are not reproduced here (fiducial HOD only)")

        theta_hm = np.asarray(ang.theta)                   # radians, increasing
        sel = theta_hm <= THETA_MAX_RAD
        theta = theta_hm[sel]
        w_A = np.asarray(ang.angular_corr_gal)[sel]
        print(f"\n=== {key}: z_eff = {z_eff}, M*_min = 1e{ms_min}, "
              f"halomod hc_spectrum = {getattr(ang, 'hc_spectrum', '?')}, "
              f"likelihood lnL = {lnl_lik:.3f}")

        # --- HOD, number density and bias vs halomod ------------------------
        mf = mass_function(z_eff, hmf_model)
        ngal, beff = ngal_beff(mf, hod_args)
        ngal_cen = trapz(mf.dndlnm * occupation(mf.m, *hod_args)[0], np.log(mf.m))
        hm_ngal = float(getattr(ang, "mean_tracer_den", np.nan))
        hm_beff = float(getattr(ang, "bias_effective_tracer", np.nan))
        print(f"  n_gal [h^3/Mpc^3]: mine {ngal:.4e} (centrals {ngal_cen:.4e}), "
              f"halomod {hm_ngal:.4e}, ratio {hm_ngal / ngal:.4f}")
        print(f"  b_eff:             mine {beff:.4f}, halomod {hm_beff:.4f}, "
              f"ratio {hm_beff / beff:.4f}")
        m_test = np.array([1e10, 3e10, 1e11, 3e11, 1e12, 3e12]) * H   # Msun/h
        my_c, my_s = occupation(m_test, *hod_args)
        hm_m = np.asarray(ang.m)
        try:
            hm_c_all, hm_s_all = np.asarray(ang.central_occupation), np.asarray(ang.satellite_occupation)
        except AttributeError:
            hm_c_all, hm_s_all = ang.hod.central_occupation(hm_m), ang.hod.satellite_occupation(hm_m)
        hm_c = np.interp(np.log(m_test), np.log(hm_m), hm_c_all)
        hm_s = np.interp(np.log(m_test), np.log(hm_m), hm_s_all)
        print("  log10 Mh[Msun]   N_cen mine / halomod      N_sat mine / halomod")
        for mt, a, b, c, d in zip(m_test / H, my_c, hm_c, my_s, hm_s):
            print(f"     {np.log10(mt):5.2f}      {a:9.3e} / {b:9.3e}     {c:9.3e} / {d:9.3e}")

        z, pz = nz_grid(p1)
        pk_lin = np.exp(np.interp(np.log(k), np.log(mf.k), np.log(mf.power)))
        pk_nl = np.exp(np.interp(np.log(k), np.log(mf.k), np.log(mf.nonlinear_power)))

        # --- B: my projection of halomod's xi_gg ----------------------------
        r_hm = np.asarray(ang.r)
        xi_hm = np.asarray(ang.corr_auto_tracer)
        r_far = np.logspace(np.log10(r_hm[-1]), np.log10(450.0), 120)[1:]
        xi_far = hm_beff ** 2 * xi_from_pk(r_far, k_xi, np.exp(
            np.interp(np.log(k_xi), np.log(mf.k), np.log(mf.power))))
        r_all = np.concatenate([r_hm, r_far])
        xi_all = np.concatenate([xi_hm, xi_far])

        def xi_fn(r):
            return np.interp(np.log(r), np.log(r_all), xi_all, left=xi_all[0], right=0.0)

        w_B = limber_xi(theta, z, pz, xi_fn)

        # --- C: b_eff^2 w_mm at fixed z (linear and halofit) -----------------
        w_C_lin = beff ** 2 * limber_k(theta, z, pz, k, lambda i: pk_lin)
        w_C_nl = beff ** 2 * limber_k(theta, z, pz, k, lambda i: pk_nl)

        # --- D: evolving b_eff(z), D(z) across N(z), linear ------------------
        support = z[pz > 1e-3 * pz.max()]
        z_nodes = np.arange(np.floor(support.min() * 4) / 4, support.max() + 0.25, 0.25)
        b_nodes, g_nodes = [], []
        for zn in z_nodes:
            mfz = mass_function(zn, hmf_model)
            b_nodes.append(ngal_beff(mfz, hod_args)[1])
            g_nodes.append(mfz.growth_factor)
        b_z = np.interp(z, z_nodes, b_nodes)
        g_z = np.interp(z, z_nodes, g_nodes) / mf.growth_factor
        w_D = limber_k(theta, z, pz, k, lambda i: (b_z[i] * g_z[i]) ** 2 * pk_lin)

        # --- integral constraint and likelihood for every variant ------------
        variants = {"A": w_A, "B": w_B, "C_lin": w_C_lin, "C_halofit": w_C_nl, "D_evolving": w_D}
        labels = {"A": "A halomod", "B": "B proj(xi_hm)", "C_lin": "C lin",
                  "C_halofit": "C halofit", "D_evolving": "D lin evolv"}
        use = thet > THETA_CUT_DEG
        th_rad = np.deg2rad(thet[use])
        print(f"  b_eff(z) across N(z): {np.round(b_nodes, 3)} at z = {z_nodes}")
        ics, at_data = {}, {}
        for name, wv in variants.items():
            ics[name] = integral_constraint_rectangle(theta, wv, *FIELD_DEG)
            at_data[name] = np.interp(th_rad, theta, wv - ics[name])
        print("\n  theta[deg]   w_obs +- sig     " + "  ".join(f"{labels[v]:>14s}" for v in variants))
        for j, t in enumerate(thet[use]):
            print(f"  {t:9.5f}  {w_obs[use][j]:8.4f} +- {sig_obs[use][j]:6.4f}  " +
                  "  ".join(f"{at_data[v][j]:14.5f}" for v in variants))
        print("  ratio to A at the data theta (before IC):")
        for name, wv in variants.items():
            ratio = np.interp(th_rad, theta, wv) / np.interp(th_rad, theta, w_A)
            print(f"    {labels[name]:>14s}: " + " ".join(f"{x:6.3f}" for x in ratio))
        print("  IC: " + ", ".join(f"{labels[n]} {ic:.5f}" for n, ic in ics.items()))
        print("  chi^2 (fitted points): " + ", ".join(
            f"{labels[n]} {chi2(at_data[n], w_obs[use], sig_obs[use]):.3f}" for n in variants)
            + f"   [likelihood -2 lnL = {-2 * lnl_lik:.3f}]")
        if hasattr(ang, "corr_1h_auto_tracer"):
            xi1 = np.asarray(ang.corr_1h_auto_tracer)
            print("  1-halo share of xi_gg at r = 0.5, 1, 2 Mpc/h: " + ", ".join(
                f"{np.interp(np.log(rr), np.log(r_hm), xi1 / xi_hm):.3f}" for rr in (0.5, 1, 2)))

        results[key] = dict(z_eff=z_eff, theta=theta, thet=thet, w_obs=w_obs, sig_obs=sig_obs,
                            ngal=ngal, beff=beff, hm_ngal=hm_ngal, hm_beff=hm_beff,
                            z_nodes=z_nodes, b_nodes=np.array(b_nodes),
                            **{f"w_{n}": v for n, v in variants.items()},
                            **{f"ic_{n}": v for n, v in ics.items()})

    out = args.out or os.path.join(args.run_dir, f"check_acf_{args.which}.npz")
    np.savez(out, params=row, names=names,
             **{f"{key}__{q}": v for key, d in results.items() for q, v in d.items()})
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
