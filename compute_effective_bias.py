"""
Effective galaxy bias b_eff(z) from the posterior, independent of the UVLF
fit -- for comparing against other clustering works.

    b_eff(z) = [ int dMh dn/dMh(Mh,z) N_gal(Mh) b_h(Mh,z) ]
             / [ int dMh dn/dMh(Mh,z) N_gal(Mh) ]

with N_gal(Mh) = N_c(Mh) + N_s(Mh) from the HOD (My_HOD in ulty.py -- the
same N_c we rewrote with the SHMR-truncation erfc-ratio form) and b_h(Mh,z)
the halo bias (Tinker10, same bias_model already used for the ACF fit).

Rather than re-deriving the HOD/halomod wiring, this reuses
LikelihoodAngBase from mcmc.py directly (its self.angular_gal is an
AngularCF_NL/halomod.TracerHaloModel instance) so the hod_params mapping
from posterior samples is guaranteed identical to what the actual ACF
likelihood uses. halomod exposes `.bias_effective_tracer` as a built-in
property computing exactly the integral above once hod_params/z are set.

Usage (cluster):
    mpirun -n <N> python compute_effective_bias.py --directory_of_posteriors <dir> \\
        [--z_list 6,7,8,9,10] [--stellar_mass_min 9.3] \\
        [--z_mthresh_overrides 9:8.75,7:8.7] [--sample_start 0] [--n_samples 500]

Reads `post_equal_weights.dat` and `run_config.json` from <dir>.

Saves to <dir>/:
    effective_bias.npz -- b_eff[sample, z], z_list, mthresh_per_z
    effective_bias.pdf -- median +/- 16-84th percentile b_eff vs z
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
from mcmc import LikelihoodAngBase

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory_of_posteriors", type=str, required=True)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--z_list", type=str, default="6,7,8,9,10")
    parser.add_argument("--stellar_mass_min", type=float, default=9.3,
                        help="log10(M*/Msun) threshold applied at every z, "
                             "unless overridden per-z via --z_mthresh_overrides.")
    parser.add_argument("--z_mthresh_overrides", type=str, default="",
                        help="Comma-separated z:Mthresh pairs, e.g. '9:8.75,7:8.7', "
                             "matching the per-obs thresholds in "
                             "LikelihoodAngBase.call_likelihood, to line b_eff up "
                             "with a specific ACF dataset.")
    parser.add_argument("--exact_specs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    directory = args.directory_of_posteriors
    z_list = [float(z) for z in args.z_list.split(",")]

    mthresh_per_z = {z: args.stellar_mass_min for z in z_list}
    if args.z_mthresh_overrides:
        for pair in args.z_mthresh_overrides.split(","):
            z_str, m_str = pair.split(":")
            mthresh_per_z[float(z_str)] = float(m_str)

    with open(os.path.join(directory, "run_config.json")) as f:
        run_config = json.load(f)
    param_names  = run_config["params"]
    fixed_Mknee  = run_config.get("fixed_Mknee", False)

    posteriors_all = np.genfromtxt(os.path.join(directory, "post_equal_weights.dat"))
    posteriors = posteriors_all[args.sample_start: args.sample_start + args.n_samples]
    my_posteriors = posteriors[rank::size]

    if rank == 0:
        print(f"[effective_bias] {len(posteriors)} samples total, {size} ranks, "
              f"{len(my_posteriors)} per rank (rank 0), z={z_list}, "
              f"M*_min={mthresh_per_z}", flush=True)

    # one LikelihoodAngBase (and its angular_gal halo model) per redshift,
    # built once per rank and reused (via .update()) across all samples --
    # mirrors how mcmc.py itself amortises the halo-model setup cost.
    bases = {
        z: LikelihoodAngBase(params=param_names, z=z, exact_specs=args.exact_specs,
                              fixed_Mknee=fixed_Mknee)
        for z in z_list
    }

    n_local = len(my_posteriors)
    n_z = len(z_list)
    local_b_eff = np.full((n_local, n_z), np.nan)

    for i_local, post_sample in enumerate(my_posteriors):
        dic_params = dict(zip(param_names, post_sample))

        alpha          = dic_params.get("alpha", 1.0)
        M_0            = dic_params.get("M_0", 11.65)
        M_1            = dic_params.get("M_1", 12.3)
        fstar_norm     = dic_params.get("fstar_norm", 0.0)
        sigma_SHMR     = dic_params.get("sigma_SHMR", 0.3)
        alpha_star_low = dic_params.get("alpha_star_low", 0.5)
        M_knee = 10 ** dic_params["M_knee"] if "M_knee" in dic_params else 2.6e11

        for i_z, z in enumerate(z_list):
            base = bases[z]
            base.angular_gal.hod_params = {
                "stellar_mass_min": mthresh_per_z[z],
                "stellar_mass_sigma": sigma_SHMR,
                "fstar_norm": 10 ** fstar_norm,
                "alpha": alpha,
                "alpha_star_low": alpha_star_low,
                "M1": M_1,
                "M_0": M_0,
                "M_knee": M_knee,
            }
            base.angular_gal.update(p1=base.p1)
            local_b_eff[i_local, i_z] = base.angular_gal.bias_effective_tracer

        if rank == 0 and (i_local + 1) % 10 == 0:
            print(f"  rank 0: {i_local + 1}/{n_local} done", flush=True)

    gathered = comm.gather(local_b_eff, root=0)

    if rank == 0:
        b_eff = np.concatenate(gathered, axis=0)  # (Nsample, Nz)

        out_path = os.path.join(directory, "effective_bias.npz")
        np.savez(
            out_path, b_eff=b_eff, z_list=np.array(z_list),
            mthresh_per_z=np.array([mthresh_per_z[z] for z in z_list]),
        )
        print(f"Saved: {out_path}", flush=True)

        med = np.nanmedian(b_eff, axis=0)
        lo  = np.nanpercentile(b_eff, 16, axis=0)
        hi  = np.nanpercentile(b_eff, 84, axis=0)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(z_list, med, color="k", lw=2.5)
        ax.fill_between(z_list, lo, hi, color="k", alpha=0.25, lw=0)
        ax.set_xlabel(r"$z$", fontsize=16)
        ax.set_ylabel(r"$b_{\rm eff}$", fontsize=16)
        ax.tick_params(labelsize=14)
        plt.tight_layout()

        plot_path = os.path.join(directory, "effective_bias.pdf")
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Saved: {plot_path}", flush=True)
