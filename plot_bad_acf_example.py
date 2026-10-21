"""Fig. ang_elp (bad_ang_example.pdf, Sec. 3 "Why do we need clustering?"):
one sample of the UV-LF-only posterior that fits the UV LF but not the ACF.

Top left: UV LF at z_eff = 11.73 (Willott+23 z~12 bin, fitted) with the other
z~12 estimates. Top right: z~7 ACF (M* > 10^9) against Paquereau+25. Bottom:
the sample's SHMR and SFMS (mean and 1 sigma) against First Light at z = 10.

Sample choice (unless --sample): among UV-LF-only samples whose UV LF lnL is
above the posterior median, the one with the lowest z~7 ACF lnL whose ACF
lies below the data. Those misfits are mild (chi2 <= 6.4 for 4 points); the
large ones are ACFs that are too HIGH (low, tight SHMR compensated by a high
or broad SFMS, ~1% of the posterior). The paper uses --sample 1304
(f* -3.44, sigma_SHMR 0.06, t* 0.09, sigma_SFMS 0.57; chi2_z7 = 18.5).

Inputs, all in <main_uvlf>/:
  post_equal_weights.dat   (last column = UV LF lnL)
  acf_a_post.npz           (acf_a_post.py; w_z7, lnL_z7 per sample)
  UVLFs_<f*>.txt           (uvlf_a_post.py, default z_list; row 4 = z 11.73)
and the First Light database (--firstlight_db, as in prior_configs/fiducial.json).
Everything the plot needs is written to bad_ang_example_data.npz, so the figure
can be remade anywhere with --data.

  python plot_bad_acf_example.py <main_uvlf>/ --firstlight_db <FirstLight_database_DR3.1.dat>
  python plot_bad_acf_example.py --data bad_ang_example_data.npz [--out bad_ang_example.pdf]
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo

from observations import Observations
from uvlf import ms_mh_flattening, SFMS, sigma_SFR_variable

MUV_GRID = np.linspace(-25, -16, 20)          # uvlf_a_post.py
Z_LIST = [6.0, 7.99, 10.23, 11.0, 11.73, 12.5, 14.0]
Z_UVLF = 11.73
Z_SFMS = 10.0                                 # First Light snapshot
M_KNEE_FIXED = 2e12
THETA_MIN = 0.005                             # deg, likelihood scale cut
FB = cosmo.Ob0 / cosmo.Om0

C_MODEL = "#4daf4a"                           # UV-LF-only colour of Figs. UVLFs / clustering
C_FL = "cornflowerblue"


def select_sample(post, lnl_uv, acf, sample=None):
    w, lnl7, theta = acf["w_z7"], acf["lnL_z7"], acf["theta_deg"]
    if sample is not None:
        return sample
    thet, wd, _ = Observations(True, True).get_obs_z7_m90()
    thet, wd = np.asarray(thet), np.asarray(wd)
    m = thet > THETA_MIN
    w_at = np.array([np.interp(thet[m], theta, wi) for wi in w])
    below = np.sum(w_at - wd[m], axis=1) < 0
    ok = np.isfinite(lnl7) & (lnl_uv >= np.median(lnl_uv)) & below
    return int(np.flatnonzero(ok)[np.argmin(lnl7[ok])])


def build(run_dir, firstlight_db, sample):
    post = np.genfromtxt(os.path.join(run_dir, "post_equal_weights.dat"))
    acf = np.load(os.path.join(run_dir, "acf_a_post.npz"))
    names = [str(n) for n in acf["param_names"]]
    assert np.allclose(acf["samples"], post[:len(acf["samples"]), :len(names)])
    lnl_uv = post[:len(acf["samples"]), -1]
    has_uv = np.array([os.path.exists(os.path.join(run_dir, f"UVLFs_{p[0]:.8f}.txt")) for p in post])
    lnl_uv = np.where(has_uv[:len(lnl_uv)], lnl_uv, -np.inf)
    i = select_sample(post, lnl_uv, acf, sample)
    p = post[i, :len(names)]
    uv = np.loadtxt(os.path.join(run_dir, f"UVLFs_{p[0]:.8f}.txt"))[Z_LIST.index(Z_UVLF)]

    from prior_pipeline.simulations import load_firstlight
    fl = load_firstlight("FirstLight", firstlight_db, target_z=(Z_SFMS,)).snapshots[0]

    lnl7 = acf["lnL_z7"]
    print(f"sample {i}: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, p)))
    print(f"  UV LF lnL {lnl_uv[i]:.2f} (posterior median {np.median(post[:, -1]):.2f}, max {post[:, -1].max():.2f})")
    print(f"  ACF z7 lnL {lnl7[i]:.2f} (UV-LF-only posterior median {np.nanmedian(lnl7):.2f}, "
          f"max {np.nanmax(lnl7):.2f}); z9 lnL {acf['lnL_z9'][i]:.2f}")
    return dict(names=np.array(names), params=p, index=i, lnl_uv=lnl_uv[i], lnl_z7=lnl7[i],
                lnl_z7_median=np.nanmedian(lnl7), muv=MUV_GRID, uvlf=uv,
                theta_deg=acf["theta_deg"], w_z7=acf["w_z7"][i],
                fl_mh=fl.mh, fl_ms=fl.ms, fl_sfr=fl.sfr)


def errorbars(ax, data, **kw):
    m, phi, (sp, sm) = data
    ax.errorbar(m, phi, yerr=[sm, sp], fmt=kw.pop("marker", "o"), capsize=3, **kw)


def plot(d, out):
    par = dict(zip([str(n) for n in d["names"]], d["params"]))
    obs = Observations(True, True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6))
    (ax_uv, ax_acf), (ax_shmr, ax_sfms) = axes
    fs, fs_leg = 16, 11.5

    # UV LF
    ax_uv.plot(d["muv"], d["uvlf"], color=C_MODEL, ls="--", lw=2.5, label="UVLF-only sample")
    grey = dict(color="grey", ms=7)
    errorbars(ax_uv, obs.get_obs_uvlf_z12_5_Donnan24(), marker="<", label=r"Donnan+24, $z=12.5$", **grey)
    errorbars(ax_uv, obs.get_obs_uvlf_z12_8_Whitler25(), marker=">", label=r"Whitler+25, $z=12.8$", **grey)
    errorbars(ax_uv, obs.get_obs_uvlf_z12_Willot23(), marker="^", color="red", ms=11, zorder=5,
              label=r"Willott+23, $z=12$")
    ax_uv.set_yscale("log")
    ax_uv.set_xlim(-22, -17)
    ax_uv.set_ylim(1e-7, 3e-3)
    ax_uv.set_xlabel(r"$M_{\rm UV}$ [mag]", fontsize=fs)
    ax_uv.set_ylabel(r"${\rm d}n/{\rm d}M_{\rm UV}\,[{\rm mag^{-1}\,Mpc^{-3}}]$", fontsize=fs)
    ax_uv.text(0.04, 0.9, r"$z=12$", transform=ax_uv.transAxes, fontsize=fs, va="top")
    ax_uv.legend(fontsize=fs_leg, loc="lower right", frameon=False)

    # ACF
    th, w = d["theta_deg"], d["w_z7"]
    # IC-subtracted model turns negative at large theta: end the line there
    neg = np.flatnonzero((th > THETA_MIN) & (w <= 0))
    m = (th > THETA_MIN) & (th < (th[neg[0]] if len(neg) else np.inf))
    ax_acf.plot(th[m], w[m], color=C_MODEL, ls="--", lw=2.5, label="UVLF-only sample")
    thet, wd, ws = (np.asarray(x) for x in obs.get_obs_z7_m90(positive_only=True))
    ax_acf.errorbar(thet, wd, yerr=ws, fmt=".", color="black", capsize=4, ms=8, label="Paquereau+25")
    ax_acf.set_xscale("log")
    ax_acf.set_yscale("log")
    ax_acf.set_xlim(THETA_MIN, 0.25)
    ax_acf.set_ylim(1e-3, 1)
    ax_acf.set_xlabel(r"$\theta$ [deg]", fontsize=fs)
    ax_acf.set_ylabel(r"$\omega(\theta)$", fontsize=fs)
    ax_acf.text(0.96, 0.9, r"$z=6$–$8$", transform=ax_acf.transAxes, fontsize=fs, va="top", ha="right")
    ax_acf.legend(fontsize=fs_leg, loc="lower left", frameon=False)

    # SHMR: mean and 1 sigma, truncated at f_b M_h as in the model
    lmh = np.linspace(8.5, 12, 200)
    lms = np.log10(ms_mh_flattening(10 ** lmh, cosmo, fstar_norm=10 ** par["fstar_norm"],
                                    alpha_star_low=par["alpha_star_low"], M_knee=M_KNEE_FIXED))
    cap = np.log10(FB) + lmh
    s = par["sigma_SHMR"]
    ax_shmr.fill_between(lmh, lms - s, np.minimum(lms + s, cap), color=C_MODEL, alpha=0.25,
                         label=r"$1\sigma$ scatter")
    ax_shmr.plot(lmh, lms, color=C_MODEL, ls="--", lw=2.5, label="UVLF-only sample")
    ax_shmr.scatter(np.log10(d["fl_mh"]), np.log10(d["fl_ms"]), color=C_FL, alpha=0.4, s=15,
                    label=rf"First Light, $z={Z_SFMS:.0f}$")
    ax_shmr.set_xlim(9, 11)
    ax_shmr.set_ylim(4, 9)
    ax_shmr.set_xlabel(r"$\log M_{\rm h}\ [{\rm M_\odot}]$", fontsize=fs)
    ax_shmr.set_ylabel(r"$\log M_\ast\ [{\rm M_\odot}]$", fontsize=fs)
    ax_shmr.legend(fontsize=fs_leg, loc="lower right", frameon=False)

    # SFMS: mean and 1 sigma at z = 10
    lmsg = np.linspace(5.5, 9.5, 200)
    lsfr = np.log10(SFMS(10 ** lmsg, SFR_norm=par["t_star"], z=Z_SFMS))
    sig = sigma_SFR_variable(10 ** lmsg, norm=par["sigma_SFMS_norm"], a_sig_SFR=par["a_sig_SFR"])
    ax_sfms.fill_between(lmsg, lsfr - sig, lsfr + sig, color=C_MODEL, alpha=0.25, label=r"$1\sigma$ scatter")
    ax_sfms.plot(lmsg, lsfr, color=C_MODEL, ls="--", lw=2.5, label="UVLF-only sample")
    ax_sfms.scatter(np.log10(d["fl_ms"]), np.log10(d["fl_sfr"]), color=C_FL, alpha=0.4, s=15,
                    label=rf"First Light, $z={Z_SFMS:.0f}$")
    ax_sfms.set_xlim(6, 9)
    ax_sfms.set_ylim(-3.5, 1.5)
    ax_sfms.set_xlabel(r"$\log M_\ast\ [{\rm M_\odot}]$", fontsize=fs)
    ax_sfms.set_ylabel(r"$\log {\rm SFR}_{100}\ [{\rm M_\odot\,yr^{-1}}]$", fontsize=fs)
    ax_sfms.legend(fontsize=fs_leg, loc="lower right", frameon=False)

    for ax in axes.flat:
        ax.tick_params(labelsize=fs - 2)
    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="UV-LF-only run (main_uvlf)")
    ap.add_argument("--firstlight_db", default=None)
    ap.add_argument("--sample", type=int, default=None, help="posterior row to show instead of the automatic choice")
    ap.add_argument("--data", default=None, help="replot from a saved bad_ang_example_data.npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.data:
        d = dict(np.load(args.data))
        out = args.out or os.path.join(os.path.dirname(args.data), "bad_ang_example.pdf")
    else:
        d = build(args.run_dir, args.firstlight_db, args.sample)
        np.savez(os.path.join(args.run_dir, "bad_ang_example_data.npz"), **d)
        out = args.out or os.path.join(args.run_dir, "bad_ang_example.pdf")
    plot(d, out)


if __name__ == "__main__":
    main()
