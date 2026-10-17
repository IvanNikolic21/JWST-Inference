"""
Posterior-predictive angular correlation functions for any run, including
UV-LF-only runs that never evaluated the ACF during sampling.

For every sample of <dir>/post_equal_weights.dat this evaluates the z~7 and
z~9 (M* > 10^9) ACF models exactly as the likelihood does (LikelihoodAngBase
from mcmc.py: realistic N(z), effective redshifts, the run's HMF and SHMR
extensions, integral constraint subtracted) and also returns the ACF lnL
against the data, i.e. how well a UV-LF-only posterior predicts the clustering.

Usage (cluster):
    mpirun -n <N> python acf_a_post.py --directory_of_posteriors <dir> \\
        [--sample_start 0] [--n_samples -1] [--no-write_txt]

Saves to <dir>/:
    acf_a_post.npz -- theta_deg, w_z7[sample, theta], w_z9[sample, theta],
                      lnL_z7, lnL_z9 (per sample), samples, param_names
    ang_z7_fs*_sig*_al*.txt and fs*_sig*_al*.txt (unless --no-write_txt):
        the same per-sample files a joint run writes during sampling, so
        notebooks that glob for them work unchanged.
"""

import os
import json
import argparse
import numpy as np
from mpi4py import MPI
from mcmc import LikelihoodAngBase
from effective_redshift import acf_redshift
from observations import Observations
from ulty import w_IC

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory_of_posteriors", type=str, required=True)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=-1, help="-1 = all samples")
    parser.add_argument("--write_txt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exact_specs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    directory = os.path.join(args.directory_of_posteriors, "")

    with open(os.path.join(directory, "run_config.json")) as f:
        run_config = json.load(f)
    param_names = run_config["params"]
    hmf_choice = run_config.get("hmf", "Tinker08")
    base_kw = dict(
        realistic_Nz=True,
        hmf_choice=hmf_choice,
        exact_specs=args.exact_specs,
        fixed_Mknee=run_config.get("fixed_Mknee", False),
        mass_dependent_sigma_shmr=run_config.get("mass_dependent_sigma_shmr", False),
        sigma_shmr_z_dependent=run_config.get("sigma_shmr_z_dependent", False),
    )

    posteriors_all = np.atleast_2d(np.genfromtxt(os.path.join(directory, "post_equal_weights.dat")))
    stop = None if args.n_samples < 0 else args.sample_start + args.n_samples
    posteriors = posteriors_all[args.sample_start:stop, :len(param_names)]
    my_idx = np.arange(len(posteriors))[rank::size]

    if rank == 0:
        print(f"[acf_a_post] {directory}: {len(posteriors)} samples, {size} ranks, "
              f"hmf={hmf_choice}, params={param_names}", flush=True)

    obs = Observations(True, True)
    data = {"z7": ("Ang_z7_m9", obs.get_obs_z7_m90()),
            "z9": ("Ang_z9_m9", obs.get_obs_z9_m90())}
    bases = {
        "z7": LikelihoodAngBase(param_names, z=acf_redshift("z7", 7.0), **base_kw),
        "z9": LikelihoodAngBase(param_names, z=acf_redshift("z9", 9.25), **base_kw),
    }
    theta_deg = np.asarray(bases["z9"].angular_gal.theta) / 2 / np.pi * 360
    n_th = len(theta_deg)

    local = {k: np.full((len(my_idx), n_th), np.nan) for k in data}
    local_lnL = {k: np.full(len(my_idx), np.nan) for k in data}

    for i_local, i in enumerate(my_idx):
        p = posteriors[i]
        for k, (obs_name, (thet, w, wsig)) in data.items():
            try:
                local_lnL[k][i_local] = bases[k].call_likelihood(
                    p, obs=obs_name, thet=thet, w=w, sig_w=wsig,
                    savedir=directory if args.write_txt else False,
                )
            except ZeroDivisionError:
                print(f"ZeroDivisionError in {obs_name} for {p}", flush=True)
                continue
            ag = bases[k].angular_gal
            ang = np.asarray(ag.angular_corr_gal)
            # same integral constraint as LikelihoodAngBase.call_likelihood
            local[k][i_local] = ang - w_IC(ag.theta, ang, 41.5 / 60, 46.6 / 60, 940.29997)
        if rank == 0 and (i_local + 1) % 10 == 0:
            print(f"  rank 0: {i_local + 1}/{len(my_idx)} done", flush=True)

    gathered = comm.gather((my_idx, local, local_lnL), root=0)

    if rank == 0:
        order = np.concatenate([g[0] for g in gathered])
        srt = np.argsort(order)
        out = {}
        for k in data:
            out[f"w_{k}"] = np.concatenate([g[1][k] for g in gathered])[srt]
            out[f"lnL_{k}"] = np.concatenate([g[2][k] for g in gathered])[srt]
        out_path = os.path.join(directory, "acf_a_post.npz")
        np.savez(out_path, theta_deg=theta_deg, samples=posteriors,
                 param_names=np.array(param_names), **out)
        lnl = out["lnL_z7"] + out["lnL_z9"]
        print(f"Saved: {out_path}", flush=True)
        print(f"ACF lnL (z7+z9) over the posterior: median {np.nanmedian(lnl):.2f}, "
              f"max {np.nanmax(lnl):.2f}, {np.sum(np.isnan(lnl))} failed", flush=True)
