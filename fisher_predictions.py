"""Model predictions for the Fisher-sensitivity figure and the parameter-
response appendix figure (Sec. 3.x, Appendix par_sweep of gal_inf.tex).

Evaluates the UV LF and the ACF with exactly the forward model of the
inference (LikelihoodUVLFBase / LikelihoodAngBase from mcmc.py: effective
redshifts, M_turn suppression of the HMF, SFR10 burstiness, sigma_BPASS,
realistic N(z), integral constraint, satellites) at

  fid             the posterior median of --run_dir (v4 main_joint)
  <p>+-h, +-h/2   central-difference steps, h = --step_frac * posterior sd
                  (0.1 dex for the fixed M_knee)
  <p>+-sw         the appendix sweep: the 2.5th / 97.5th posterior percentiles
                  (--sweep_percentiles; M_knee +- --mknee_sweep dex)

and saves everything fisher_analysis.py needs (predictions, data, prior,
posterior samples) to --out. Fisher matrices and figures are made locally
by fisher_analysis.py from that file.

The ACF depends only on the SHMR parameters (the HOD in LikelihoodAngBase
uses f*, alpha*, sigma_SHMR, M_knee), so for t*, sigma_SFMS,0, a_sigma and
sigma_SFR10 the ACF is not recomputed and the fiducial one is stored.

CANNOT BE RUN LOCALLY -- needs halomod and the BPASS spectra.

  mpirun -np 40 python fisher_predictions.py --run_dir <main_joint>/ \\
      --uvlf_run_dir <main_uvlf>/ --out <main_joint>/fisher_raw.npz
"""
import argparse
import json
import os

import numpy as np

from mcmc import LikelihoodAngBase, LikelihoodUVLFBase
from observations import Observations
from uvlf import SFH_sampler, bpass_loader
from ulty import w_IC

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None
rank = comm.Get_rank() if comm else 0
size = comm.Get_size() if comm else 1

ACF_PARAMS = {"fstar_norm", "sigma_SHMR", "alpha_star_low", "M_knee"}
UVLF_DATA = {"UVLF_z8_Willot23": "get_obs_uvlf_z8_Willot23",
             "UVLF_z9_Willot23": "get_obs_uvlf_z9_Willot23",
             "UVLF_z10_Willot23": "get_obs_uvlf_z10_Willot23",
             "UVLF_z12_Willot23": "get_obs_uvlf_z12_Willot23"}
ACF_DATA = {"z7": ("Ang_z7_m9", "get_obs_z7_m90"), "z9": ("Ang_z9_m9", "get_obs_z9_m90")}
# M_UV values evaluated besides the data points: idealized bins and the appendix curves
MUV_EXTRA = np.round(np.concatenate([np.arange(-21.0, -17.49, 0.5), np.arange(-23.0, -15.99, 0.25)]), 3)
BPASS_CLUSTER = "/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00."


def load_posterior(run_dir, n_par):
    return np.genfromtxt(os.path.join(run_dir, "post_equal_weights.dat"))[:, :n_par]


def build_points(params, fid, sd, limits, step_frac, sweep_lo, sweep_hi, mknee_sweep):
    """name -> parameter vector, and name -> varied parameter (None for fid).
    Derivative points at fid +- h, +- h/2; sweep points at sweep_lo / sweep_hi
    (posterior percentiles), M_knee at fid +- mknee_sweep."""
    points, varied = {"fid": fid.copy()}, {"fid": None}
    for i, p in enumerate(params):
        h = 0.1 if p == "M_knee" else step_frac * sd[i]
        if p == "M_knee":
            lo, hi = fid[i] - mknee_sweep, fid[i] + mknee_sweep
        else:
            lo, hi = sweep_lo[i], sweep_hi[i]
        steps = [("+h", fid[i] + h), ("-h", fid[i] - h), ("+h2", fid[i] + h / 2), ("-h2", fid[i] - h / 2),
                 ("+sw", hi), ("-sw", lo)]
        for tag, val in steps:
            v = fid.copy()
            v[i] = np.clip(val, limits[i][0], limits[i][1])
            points[p + tag] = v
            varied[p + tag] = p
    return points, varied


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="joint run: fiducial and posterior sd")
    ap.add_argument("--uvlf_run_dir", default=None, help="UV-LF-only run, stored for validation")
    ap.add_argument("--out", default=None)
    ap.add_argument("--step_frac", type=float, default=0.2)
    ap.add_argument("--sweep_percentiles", type=float, nargs=2, default=[2.5, 97.5],
                    help="posterior percentiles used as low / high in the appendix sweep")
    ap.add_argument("--mknee_sweep", type=float, default=0.7)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run_dir, "fisher_raw.npz")

    with open(os.path.join(args.run_dir, "run_config.json")) as f:
        cfg = json.load(f)
    params = cfg["params"]
    with open(cfg["prior_file"]) as f:
        prior = json.load(f)
    assert prior["params"] == params, (prior["params"], params)
    limits = np.array(prior["sampling_limits"], dtype=float)

    post = load_posterior(args.run_dir, len(params))
    fid, sd = np.median(post, axis=0), np.std(post, axis=0)
    sweep_lo, sweep_hi = np.percentile(post, args.sweep_percentiles, axis=0)
    points, varied = build_points(params, fid, sd, limits, args.step_frac, sweep_lo, sweep_hi, args.mknee_sweep)
    names = list(points)

    obs = Observations(True, True)
    uvlf_keys = [k for k in cfg["likelihoods"] if k in UVLF_DATA]
    uvlf_z = {k: cfg["uvlf_redshifts"][k] for k in uvlf_keys}
    data_uv = {k: getattr(obs, UVLF_DATA[k])() for k in uvlf_keys}
    muv_all = np.unique(np.round(np.concatenate([MUV_EXTRA] + [d[0] for d in data_uv.values()]), 3))
    acf_keys = [k for k, (o, _) in ACF_DATA.items() if o in cfg["likelihoods"]]
    data_acf = {k: getattr(obs, ACF_DATA[k][1])() for k in acf_keys}

    tasks = [("uvlf", k, n) for k in uvlf_keys for n in names]
    tasks += [("acf", k, n) for k in acf_keys for n in names
              if varied[n] is None or varied[n] in ACF_PARAMS]
    my_tasks = tasks[rank::size]

    bp = vect_func = None
    uv_bases, sfh_samps, ang_bases = {}, {}, {}
    results = {}
    for kind, key, name in my_tasks:
        p = points[name]
        if kind == "uvlf":
            if bp is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                bp = bpass_loader(filename=BPASS_CLUSTER) if script_dir.startswith("/groups/astro") else bpass_loader()
                vect_func = np.vectorize(bp.get_UV_sfr10)
            if key not in uv_bases:
                uv_bases[key] = LikelihoodUVLFBase(
                    params, z=uvlf_z[key], hmf_choice=cfg["hmf"],
                    sigma_sfr_10_explicit=cfg["sigma_sfr_10_explicit"], sigma_uv=cfg["sigma_uv"],
                    mass_dependent_sigma_uv=cfg["mass_dependent_sigma_uv"],
                    mass_dependent_sigma_shmr=cfg["mass_dependent_sigma_shmr"],
                    sigma_shmr_z_dependent=cfg["sigma_shmr_z_dependent"], fixed_Mknee=cfg["fixed_Mknee"])
                sfh_samps[key] = SFH_sampler(z=uvlf_z[key])
            ones = np.ones(len(muv_all))
            uv_bases[key].call_likelihood(p, muvs_o=muv_all, uvlf_o=ones, sig_o=(ones, ones),
                                          sfr_samp_inst=sfh_samps[key], bpass_read=bp, vect_func=vect_func)
            results[(kind, key, name)] = uv_bases[key].last_preds.copy()
        else:
            if key not in ang_bases:
                ang_bases[key] = LikelihoodAngBase(
                    params, realistic_Nz=True, hmf_choice=cfg["hmf"], z=cfg["acf_redshifts"][key],
                    fixed_Mknee=cfg["fixed_Mknee"], mass_dependent_sigma_shmr=cfg["mass_dependent_sigma_shmr"],
                    sigma_shmr_z_dependent=cfg["sigma_shmr_z_dependent"])
            thet, w, sig = data_acf[key]
            base = ang_bases[key]
            base.call_likelihood(p, obs=ACF_DATA[key][0], thet=thet, w=w, sig_w=sig)
            ag = base.angular_gal
            ang = np.asarray(ag.angular_corr_gal)
            results[(kind, key, name)] = ang - w_IC(ag.theta, ang, 41.5 / 60, 46.6 / 60, 940.29997)
            results[("theta", key, "")] = np.asarray(ag.theta) * 180 / np.pi
        print(f"[rank {rank}] {kind} {key} {name} done", flush=True)

    gathered = comm.gather(results, root=0) if comm else [results]
    if rank != 0:
        return
    allres = {}
    for r in gathered:
        allres.update(r)

    uv = np.array([[allres[("uvlf", k, n)] for k in uvlf_keys] for n in names])
    theta = np.array([allres[("theta", k, "")] for k in acf_keys])
    acf = np.array([[allres.get(("acf", k, n), allres[("acf", k, "fid")]) for k in acf_keys] for n in names])

    save = dict(
        params=np.array(params), fid=fid, sd_post=sd, prior_mean=np.array(prior["mean"]),
        prior_cov=np.array(prior["cov_effective"]), limits=limits,
        names=np.array(names), varied=np.array([varied[n] or "" for n in names]), points=np.array([points[n] for n in names]),
        step_frac=args.step_frac, sweep_percentiles=np.array(args.sweep_percentiles), mknee_sweep=args.mknee_sweep,
        uvlf_keys=np.array(uvlf_keys), uvlf_z=np.array([uvlf_z[k] for k in uvlf_keys]),
        muv=muv_all, uvlf=uv,
        acf_keys=np.array(acf_keys), acf_z=np.array([cfg["acf_redshifts"][k] for k in acf_keys]),
        theta=theta, acf=acf, post_joint=post,
    )
    for k in uvlf_keys:
        m, phi, (sp, sm) = data_uv[k]
        save[f"data_{k}"] = np.vstack([m, phi, sp, sm])
    for k in acf_keys:
        save[f"data_acf_{k}"] = np.vstack([np.asarray(x, dtype=float) for x in data_acf[k]])
    if args.uvlf_run_dir:
        save["post_uvlf"] = load_posterior(args.uvlf_run_dir, len(params))
    np.savez(out, **save)
    print(f"saved {out}: {len(names)} points, {len(tasks)} model evaluations", flush=True)


if __name__ == "__main__":
    main()
