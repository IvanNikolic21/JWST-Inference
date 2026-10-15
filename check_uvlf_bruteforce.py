"""Brute-force Monte Carlo check of the UV LF likelihood model (likelihood
validation, step 2a).

For one parameter set (MAP or median of a run) and each Willott+24 redshift
used in the fit, draws individual galaxies and histograms their M_UV:

  halo      log Mh uniform in [LMH_LO, LMH_HI], weight dn/dlog10Mh (with the
            same exp(-5e8/M) factor as mcmc.py) * range / N
  M*        lognormal SHMR, truncated at the baryon fraction (inverse CDF)
  SFR       lognormal SFMS with sigma_SFMS evaluated at the drawn M*
  burst     x = log10(SFR_10/SFR) ~ N(0, sigma_sfr_10)
  M_UV      bpass_loader.get_UV_sfr10 called for every galaxy (no tables),
            plus the 0.1 mag BPASS scatter

Two versions of the last step:
  physical  metallicity and stellar mass of the galaxy itself
  map       metallicity and stellar mass of the mean-relation galaxy with the
            same SFR, which is what the likelihood's M_UV(SFR, x) table assumes

The likelihood's own prediction (uvlf.UV_calc_numba_sfr10, seed 0 as in
mcmc.py, plus seeds 1-4 to measure its fixed Monte Carlo error) is computed on
the same grid. Output: a table at the data points, Delta lnL, and an npz with
the fine-binned luminosity functions for bin-averaging offline.

Cluster only (BPASS spectra). With MPI, e.g. 40 ranks:
  mpirun -np 40 python check_uvlf_bruteforce.py --run_dir <run> --which map
"""
import argparse
import json
import os

import numpy as np
import hmf
from astropy.cosmology import Planck18 as cosmo
from mpi4py import MPI
from scipy.special import ndtr, ndtri

from cosmology_settings import HMF_COSMO_KW
from effective_redshift import uvlf_redshift
from likelihood_terms import asymmetric_gaussian_lnl
from observations import Observations
from uvlf import (SFH_sampler, bpass_loader, ms_mh_flattening, SFMS_slope,
                  sigma_SFR_variable, metalicity_from_FMR, DeltaZ_z, Muv_Luv,
                  UV_calc_numba_sfr10)

H = cosmo.h
M_TURN = 5e8                    # Msun, exp(-M_TURN/M) in mcmc.py
LMH_LO, LMH_HI = 6.0, 14.0      # log10 Mh [Msun] sampled
SIGMA_BPASS = 0.1               # mag, sigma_muv_const in the likelihood
MUV_EDGES = np.round(np.arange(-24.0, -13.95, 0.1), 2)
HALF_WIN = 0.05                 # mag, window around each data point
FAINT_CUT = -20.0               # --use_only_faint_end
DATASETS = {                    # likelihood name -> (nominal z, data getter)
    "UVLF_z8_Willot23": (8, "get_obs_uvlf_z8_Willot23"),
    "UVLF_z10_Willot23": (10, "get_obs_uvlf_z10_Willot23"),
    "UVLF_z12_Willot23": (12, "get_obs_uvlf_z12_Willot23"),
}
CLUSTER_BPASS = "/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00."


def pick_params(run_dir, which):
    names = list(json.load(open(os.path.join(run_dir, "priors_params.json"))))
    post = np.loadtxt(os.path.join(run_dir, "post_equal_weights.dat"))
    if which == "map":
        row = post[np.argmax(post[:, -1]), :len(names)]
    else:
        row = np.median(post[:, :len(names)], axis=0)
    return names, row


def model_kwargs(dic):
    return dict(
        f_star_norm=10 ** dic["fstar_norm"],
        alpha_star=dic["alpha_star_low"],
        sigma_SHMR=dic["sigma_SHMR"],
        sigma_SFMS_norm=dic["sigma_SFMS_norm"],
        t_star=dic["t_star"],
        a_sig_SFR=dic["a_sig_SFR"],
        M_knee=10 ** dic["M_knee"] if "M_knee" in dic else 2.6e11,
        sigma_sfr10=dic.get("sigma_sfr_10", 0.2),
    )


def hmf_tables(z, hmf_model):
    mf = hmf.MassFunction(z=z, Mmin=5, Mmax=19, dlog10m=0.05, hmf_model=hmf_model, **HMF_COSMO_KW)
    log_m = np.log10(mf.m / H)                                     # Msun
    dndlog10m = mf.dndlog10m * H ** 3 * np.exp(-M_TURN / (mf.m / H))  # Mpc^-3 dex^-1
    return log_m, dndlog10m


def draw_galaxies(rng, n, z, kw, log_m, dndlog10m):
    """Halo weights and (log M*, log SFR, x) for n galaxies."""
    log_mh = rng.uniform(LMH_LO, LMH_HI, n)
    weight = np.exp(np.interp(log_mh, log_m, np.log(dndlog10m))) * (LMH_HI - LMH_LO)
    mu = np.log10(ms_mh_flattening(10 ** log_mh, cosmo, fstar_norm=kw["f_star_norm"],
                                   alpha_star_low=kw["alpha_star"], M_knee=kw["M_knee"]))
    top = np.log10(cosmo.Ob0 / cosmo.Om0) + log_mh
    cdf_top = ndtr((top - mu) / kw["sigma_SHMR"])
    u = rng.uniform(0.0, 1.0, n) * cdf_top
    log_ms = mu + kw["sigma_SHMR"] * ndtri(np.clip(u, 1e-300, 1.0))
    log_ms = np.minimum(log_ms, top)
    sfms = np.log10(SFMS_slope(10 ** log_ms, SFR_norm=kw["t_star"], z=z))
    sig = sigma_SFR_variable(10 ** log_ms, norm=kw["sigma_SFMS_norm"], a_sig_SFR=kw["a_sig_SFR"])
    log_sfr = sfms + sig * rng.standard_normal(n)
    x = kw["sigma_sfr10"] * rng.standard_normal(n)
    return weight, log_ms, log_sfr, x


def mean_relation(z, kw, log_m):
    """log M* and log SFR along the mean SHMR+SFMS on the halo grid, as in the
    likelihood's M_UV(SFR, x) table."""
    ms = ms_mh_flattening(10 ** log_m, cosmo, fstar_norm=kw["f_star_norm"],
                          alpha_star_low=kw["alpha_star"], M_knee=kw["M_knee"])
    sfr = SFMS_slope(ms, SFR_norm=kw["t_star"], z=z)
    return np.log10(ms), np.log10(sfr)


def galaxy_muv(bp, samp, z, log_ms, log_sfr, x, rng):
    ms, sfr = 10 ** log_ms, 10 ** log_sfr
    metal = metalicity_from_FMR(ms, sfr) + DeltaZ_z(z)
    out = np.empty(len(ms))
    for i in range(len(ms)):
        lum = bp.get_UV_sfr10(metal[i], ms[i], sfr[i], z, SFH_samp=samp, sfr_10=sfr[i] * 10 ** x[i])
        out[i] = Muv_Luv(lum * 3.846e33)
    return out + SIGMA_BPASS * rng.standard_normal(len(ms))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--which", choices=["map", "median"], default="map")
    ap.add_argument("--n_gal", type=float, default=1e7, help="galaxies per redshift (all ranks)")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--n_seeds", type=int, default=5, help="likelihood seeds 0..n-1")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    cfg = json.load(open(os.path.join(args.run_dir, "run_config.json")))
    names, row = pick_params(args.run_dir, args.which)
    dic = dict(zip(names, row))
    kw = model_kwargs(dic)
    extra = set(names) - {"fstar_norm", "sigma_SHMR", "t_star", "alpha_star_low",
                          "sigma_SFMS_norm", "a_sig_SFR", "M_knee", "sigma_sfr_10"}
    if extra or cfg.get("mass_dependent_sfr10"):
        raise NotImplementedError(f"fiducial model only; this run also has {extra}")
    hmf_model = cfg.get("hmf", "Tinker08")

    bp = bpass_loader(filename=CLUSTER_BPASS) if os.path.exists(CLUSTER_BPASS + "zem5.dat") \
        else bpass_loader()
    vect_func = np.vectorize(bp.get_UV_sfr10)
    obs = Observations(True, True)

    data, zs = {}, {}
    for name, (z_nom, getter) in DATASETS.items():
        muv, phi, (sp, sm) = getattr(obs, getter)()
        keep = muv >= FAINT_CUT
        data[name] = (muv[keep], phi[keep], sp[keep], sm[keep])
        zs[name] = uvlf_redshift(name, z_nom)
    if rank == 0:
        print(f"run {args.run_dir} ({args.which}); " + ", ".join(f"{n}={v:.4f}" for n, v in dic.items()))
        print("redshifts: " + ", ".join(f"{n} z={z}" for n, z in zs.items()), flush=True)

    # --- likelihood predictions: (dataset, seed) jobs spread over ranks -----
    centres = 0.5 * (MUV_EDGES[1:] + MUV_EDGES[:-1])
    jobs = [(name, seed) for name in DATASETS for seed in range(args.n_seeds)]
    my_lik = {}
    samplers = {}
    for j, (name, seed) in enumerate(jobs):
        if j % size != rank:
            continue
        z = zs[name]
        samplers.setdefault(z, SFH_sampler(z=z))
        log_m, dndlog10m = hmf_tables(z, hmf_model)
        muv_eval = np.concatenate([data[name][0], centres])
        my_lik[(name, seed)] = UV_calc_numba_sfr10(
            muv_eval, log_m, dndlog10m, z=z, vect_func=vect_func, bpass_read=bp,
            SFH_samp=samplers[z], mass_dependent_sfr10=False, seed=seed, **kw)
    lik = {}
    for part in comm.gather(my_lik, root=0) or []:
        lik.update(part)

    # --- brute-force Monte Carlo ---------------------------------------------
    n_per_rank = int(np.ceil(args.n_gal / size))
    mc = {}
    for name in DATASETS:
        z = zs[name]
        samp = samplers.get(z) or samplers.setdefault(z, SFH_sampler(z=z))
        log_m, dndlog10m = hmf_tables(z, hmf_model)
        mean_ms, mean_sfr = mean_relation(z, kw, log_m)
        rng = np.random.default_rng([rank, int(100 * z)])
        muv_pts = data[name][0]
        acc = {mode: dict(h=np.zeros(len(centres)), h2=np.zeros(len(centres)),
                          p=np.zeros(len(muv_pts)), p2=np.zeros(len(muv_pts)))
               for mode in ("physical", "map")}
        done = 0
        while done < n_per_rank:
            n = min(args.chunk, n_per_rank - done)
            w, log_ms, log_sfr, x = draw_galaxies(rng, n, z, kw, log_m, dndlog10m)
            w = w / (n_per_rank * size)
            # the likelihood drops galaxies outside its sample ranges
            inside = (log_sfr > -5) & (log_sfr < 5) & (log_ms > 2) & (log_ms < 12)
            log_ms_map = np.interp(log_sfr, mean_sfr, mean_ms)
            for mode, lms in (("physical", log_ms), ("map", log_ms_map)):
                muv = galaxy_muv(bp, samp, z, lms[inside], log_sfr[inside], x[inside], rng)
                wi = w[inside]
                a = acc[mode]
                a["h"] += np.histogram(muv, MUV_EDGES, weights=wi)[0]
                a["h2"] += np.histogram(muv, MUV_EDGES, weights=wi ** 2)[0]
                near = np.abs(muv[:, None] - muv_pts[None, :]) < HALF_WIN
                a["p"] += (wi[:, None] * near).sum(0)
                a["p2"] += (wi[:, None] ** 2 * near).sum(0)
            done += n
        for mode in acc:
            for q in acc[mode]:
                acc[mode][q] = comm.reduce(acc[mode][q], op=MPI.SUM, root=0)
        mc[name] = acc
        if rank == 0:
            print(f"  MC done for {name}", flush=True)

    if rank != 0:
        return

    dmag = np.diff(MUV_EDGES)
    out = dict(params=row, names=names, muv_edges=MUV_EDGES, centres=centres)
    dlnl = {}
    print("\n M_UV     phi_obs (+/-)        likelihood   seed spread   MC map       MC physical    "
          "lik/map  lik/phys  map/phys")
    for name in DATASETS:
        muv_pts, phi, sp, sm = data[name]
        npt = len(muv_pts)
        lik_all = np.array([lik[(name, s)] for s in range(args.n_seeds)])
        lik_pts, lik_fine = lik_all[:, :npt], lik_all[:, npt:]
        a_map, a_phys = mc[name]["map"], mc[name]["physical"]
        mc_map = a_map["p"] / (2 * HALF_WIN)
        mc_phys = a_phys["p"] / (2 * HALF_WIN)
        err_map = np.sqrt(a_map["p2"]) / (2 * HALF_WIN)
        err_phys = np.sqrt(a_phys["p2"]) / (2 * HALF_WIN)
        print(f" --- {name} (z = {zs[name]})")
        for j in range(npt):
            spread = np.std(lik_pts[:, j]) / np.mean(lik_pts[:, j])
            print(f" {muv_pts[j]:6.2f}  {phi[j]:.2e} (+{sp[j]:.1e}/-{sm[j]:.1e})  {lik_pts[0, j]:.3e}"
                  f"   {100 * spread:5.1f}%     {mc_map[j]:.3e}±{100 * err_map[j] / mc_map[j]:.1f}%"
                  f"  {mc_phys[j]:.3e}±{100 * err_phys[j] / mc_phys[j]:.1f}%"
                  f"   {lik_pts[0, j] / mc_map[j]:6.3f}  {lik_pts[0, j] / mc_phys[j]:6.3f}"
                  f"  {mc_map[j] / mc_phys[j]:6.3f}")
        lnl = {lab: sum(asymmetric_gaussian_lnl(p, o, s1, s2) for p, o, s1, s2 in zip(pred, phi, sp, sm))
               for lab, pred in (("likelihood", lik_pts[0]), ("MC map", mc_map), ("MC physical", mc_phys))}
        dlnl[name] = lnl
        print("   lnL: " + ", ".join(f"{k} {v:.3f}" for k, v in lnl.items()))
        out.update({f"{name}__z": zs[name], f"{name}__muv_obs": muv_pts, f"{name}__phi_obs": phi,
                    f"{name}__lik_pts": lik_pts, f"{name}__lik_fine": lik_fine,
                    f"{name}__mc_map_pts": mc_map, f"{name}__mc_phys_pts": mc_phys,
                    f"{name}__mc_map_fine": a_map["h"] / dmag,
                    f"{name}__mc_map_fine_err": np.sqrt(a_map["h2"]) / dmag,
                    f"{name}__mc_phys_fine": a_phys["h"] / dmag,
                    f"{name}__mc_phys_fine_err": np.sqrt(a_phys["h2"]) / dmag})
    tot = {k: sum(d[k] for d in dlnl.values()) for k in ("likelihood", "MC map", "MC physical")}
    print("\n total lnL (UV LF part): " + ", ".join(f"{k} {v:.3f}" for k, v in tot.items()))
    print(" Delta lnL (likelihood - MC physical) = "
          f"{tot['likelihood'] - tot['MC physical']:+.3f}; (likelihood - MC map) = "
          f"{tot['likelihood'] - tot['MC map']:+.3f}")
    path = os.path.join(args.run_dir, f"check_uvlf_bruteforce_{args.which}.npz")
    np.savez(path, **out)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
