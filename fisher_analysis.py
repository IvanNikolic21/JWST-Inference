"""Fisher sensitivity figures and the parameter-response appendix figure from
the predictions saved by fisher_predictions.py (runs locally).

Two Fisher versions, both shown as the dimensionless
    S_i^(o) = sigma_ref,i * sqrt(F_ii^(o)),
the shift of observable o, in units of its uncertainty, produced by a
1-sigma prior change of parameter i (sigma_ref = prior sd; 1 dex for the
fixed M_knee). S > 1: observable o is informative about p_i beyond the prior.

  data       the actual likelihood data: Willott UV LF points (M_UV >= -20,
             error sqrt(sigma+ sigma-), Barlow's variance at the data point)
             and Paquereau ACF points at theta > 0.005 deg (Gaussian in w)
  idealized  every observable "measured" to 20% in 8 bins: UV LF at
             z = 7.99 (-21 <= M_UV <= -17.5), ACF at z = 7.29 (0.005-0.05
             deg), SHMR mean/scatter (10^9.5-10^11.5 Msun), SFMS
             mean/scatter at z = 7.99 (10^8-10^10 Msun)

Derivatives: central differences at h and h/2, Richardson-combined; the
change between the two step sizes is printed as a convergence check.
Validation: (F_data + F_prior)^-1 against the posterior sd and the
f*-sigma_SHMR correlation of the joint and UV-LF-only runs.

  python fisher_analysis.py fisher_raw.npz [--outdir figs/]
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from astropy.cosmology import Planck18 as cosmo

from uvlf import ms_mh_flattening, SFMS, sigma_SFR_variable

LABELS = {
    "fstar_norm": r"$\log_{10} f_{\ast,0}$",
    "sigma_SHMR": r"$\sigma_{\rm SHMR}$",
    "t_star": r"$t_{\ast}$",
    "alpha_star_low": r"$\alpha_{\ast}$",
    "sigma_SFMS_norm": r"$\sigma_{{\rm SFMS},0}$",
    "a_sig_SFR": r"$a_{\sigma,{\rm SFMS}}$",
    "M_knee": r"$\log_{10} M_{\rm knee}$ (fixed)",
    "sigma_sfr_10": r"$\sigma_{{\rm SFR},10}$",
}
ROW_ORDER = ["fstar_norm", "sigma_SHMR", "alpha_star_low", "M_knee",
             "t_star", "sigma_SFMS_norm", "a_sig_SFR", "sigma_sfr_10"]
MKNEE_REF = 1.0                       # dex, reference "prior sd" of the fixed M_knee
SIG20 = np.log10(1.2)                 # 20% in log10
MUV_IDEAL = np.arange(-21.0, -17.49, 0.5)
THETA_IDEAL = np.logspace(np.log10(0.005), np.log10(0.05), 8)
MH_IDEAL = np.logspace(9.5, 11.5, 8)
MS_IDEAL = np.logspace(8.0, 10.0, 8)
Z_SCALING = 7.99
FAINT_CUT = -20.0
ACF_CUT = 0.005


class Raw:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.d = d
        self.params = [str(p) for p in d["params"]]
        self.names = [str(n) for n in d["names"]]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.points = d["points"]
        self.uvlf_keys = [str(k) for k in d["uvlf_keys"]]
        self.acf_keys = [str(k) for k in d["acf_keys"]]
        self.muv, self.uvlf = d["muv"], d["uvlf"]          # (Npt, Nz, Nmuv)
        self.theta, self.acf = d["theta"], d["acf"]        # (Nzacf, Nth), (Npt, Nzacf, Nth)
        self.prior_sd = np.sqrt(np.diag(d["prior_cov"]))
        self.sig_ref = np.where(np.array(self.params) == "M_knee", MKNEE_REF, self.prior_sd)

    def p(self, name):
        return dict(zip(self.params, self.points[self.idx[name]]))


def scaling_vector(pd):
    """log10 of SHMR mean, SHMR scatter, SFMS mean, SFMS scatter on the idealized bins."""
    ms = ms_mh_flattening(MH_IDEAL, cosmo, fstar_norm=10 ** pd["fstar_norm"],
                          alpha_star_low=pd["alpha_star_low"], M_knee=10 ** pd["M_knee"])
    return {
        "SHMR mean": np.log10(ms),
        "SHMR scatter": np.log10(np.full(len(MH_IDEAL), pd["sigma_SHMR"])),
        "SFMS mean": np.log10(SFMS(MS_IDEAL, SFR_norm=pd["t_star"], z=Z_SCALING)),
        "SFMS scatter": np.log10(sigma_SFR_variable(MS_IDEAL, norm=pd["sigma_SFMS_norm"],
                                                    a_sig_SFR=pd["a_sig_SFR"])),
    }


def observables(raw, name):
    """name -> {block: (model vector, sigma vector)} for both Fisher versions."""
    i = raw.idx[name]
    data, ideal = {}, {}
    for iz, k in enumerate(raw.uvlf_keys):
        m, phi, sp, sm = raw.d[f"data_{k}"]
        sel = m >= FAINT_CUT
        model = np.interp(m[sel], raw.muv, raw.uvlf[i, iz])
        data[k] = (model, np.sqrt(sp[sel] * sm[sel]))
    for iz, k in enumerate(raw.acf_keys):
        th, w, sig = raw.d[f"data_acf_{k}"]
        sel = th > ACF_CUT
        model = np.interp(th[sel], raw.theta[iz], raw.acf[i, iz])
        data[f"ACF_{k}"] = (model, sig[sel])
    k8 = raw.uvlf_keys.index("UVLF_z8_Willot23")
    ideal["UV LF"] = (np.log10(np.interp(MUV_IDEAL, raw.muv, raw.uvlf[i, k8])), SIG20)
    w = np.interp(THETA_IDEAL, raw.theta[0], raw.acf[i, 0])
    if np.any(w <= 0):
        raise ValueError(f"{name}: w - w_IC <= 0 in the idealized ACF range")
    ideal["ACF"] = (np.log10(w), SIG20)
    for k, v in scaling_vector(raw.p(name)).items():
        ideal[k] = (v, SIG20)
    return data, ideal


def jacobians(raw):
    """block -> (J (Ndata, Npar) Richardson-combined, sigma, max rel. change h vs h/2 in S)."""
    obs = {n: observables(raw, n) for n in raw.names}
    out = {}
    for version in (0, 1):
        blocks = obs["fid"][version].keys()
        for b in blocks:
            sig = obs["fid"][version][b][1]
            cols, conv = [], []
            for ip, p in enumerate(raw.params):
                def deriv(tag):
                    yp, ym = obs[p + "+" + tag][version][b][0], obs[p + "-" + tag][version][b][0]
                    dp = raw.points[raw.idx[p + "+" + tag], ip] - raw.points[raw.idx[p + "-" + tag], ip]
                    return (yp - ym) / dp
                d1, d2 = deriv("h"), deriv("h2")
                cols.append((4 * d2 - d1) / 3)
                s1, s2 = np.sqrt(np.sum((d1 / sig) ** 2)), np.sqrt(np.sum((d2 / sig) ** 2))
                conv.append(abs(s1 - s2) / max(s2, 1e-300) if s2 > 0 else 0.0)
            out[(version, b)] = (np.column_stack(cols), sig, np.array(conv))
    return out


def fisher(J, sig):
    Jw = J / np.broadcast_to(sig, (J.shape[0],))[:, None]
    return Jw.T @ Jw


def combine(jac, version, blocks):
    return sum(fisher(*jac[(version, b)][:2]) for b in blocks)


def heatmap(S, rows, cols, fname, title, sep=None, group_labels=None):
    fig, ax = plt.subplots(figsize=(1.35 * len(cols) + 3, 0.62 * len(rows) + 1.8))
    Sm = np.ma.masked_where(S <= 1e-6, S)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("black")
    vmax = max(np.nanmax(Sm), 10)
    im = ax.imshow(Sm, cmap=cmap, aspect="auto", norm=LogNorm(vmin=1e-2, vmax=vmax))
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i, j] > 1e-6:
                txt = f"{S[i, j]:.2f}" if S[i, j] < 10 else f"{S[i, j]:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                        color="black" if im.norm(S[i, j]) > 0.6 else "white")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right", fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[r] for r in rows], fontsize=12)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for x in (sep or []):
        ax.axvline(x, color="white", lw=6)
    for x, lab in (group_labels or []):
        ax.text(x, 1.02, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(r"$S_i = \sigma_{{\rm prior},i}\sqrt{F_{ii}}$", fontsize=12)
    ax.set_title(title, fontsize=13, pad=28 if group_labels else 8)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def response_figure(raw, fname):
    """Appendix figure: low / fiducial / high (fid -+ sweep) for each parameter."""
    rows = [p for p in ROW_ORDER if p in raw.params]
    COL = {"-sw": "royalblue", "fid": "black", "+sw": "crimson"}
    mh = np.logspace(8, 12.5, 200)
    ms = np.logspace(6, 10.5, 200)
    k8 = raw.uvlf_keys.index("UVLF_z8_Willot23")
    m_d, phi_d, sp_d, sm_d = raw.d["data_UVLF_z8_Willot23"]
    th_d, w_d, s_d = raw.d[f"data_acf_{raw.acf_keys[0]}"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(15.5, 2.25 * len(rows)), sharex="col")
    for r, p in enumerate(rows):
        ip = raw.params.index(p)
        vals = {}
        for tag in ("-sw", "+sw", "fid"):
            name = "fid" if tag == "fid" else p + tag
            i, pd, c = raw.idx[name], raw.p(name), COL[tag]
            vals[tag] = raw.points[i, ip]
            ax = axes[r, 0]
            ax.plot(raw.theta[0], raw.acf[i, 0], color=c, lw=1.6)
            ax = axes[r, 1]
            sel = (raw.muv >= -23) & (raw.muv <= -16)
            ax.plot(raw.muv[sel], raw.uvlf[i, k8][sel], color=c, lw=1.6)
            m_star = ms_mh_flattening(mh, cosmo, fstar_norm=10 ** pd["fstar_norm"],
                                      alpha_star_low=pd["alpha_star_low"], M_knee=10 ** pd["M_knee"])
            axes[r, 2].fill_between(mh, m_star * 10 ** -pd["sigma_SHMR"], m_star * 10 ** pd["sigma_SHMR"],
                                    color=c, alpha=0.2, lw=0)
            axes[r, 2].plot(mh, m_star, color=c, lw=1.3)
            sfr = SFMS(ms, SFR_norm=pd["t_star"], z=Z_SCALING)
            sg = sigma_SFR_variable(ms, norm=pd["sigma_SFMS_norm"], a_sig_SFR=pd["a_sig_SFR"])
            axes[r, 3].fill_between(ms, sfr * 10 ** -sg, sfr * 10 ** sg, color=c, alpha=0.2, lw=0)
            axes[r, 3].plot(ms, sfr, color=c, lw=1.3)
        axes[r, 0].errorbar(th_d, w_d, s_d, fmt=".", color="gray", ms=4, capsize=2)
        axes[r, 0].axvspan(1e-4, ACF_CUT, color="gray", alpha=0.15, lw=0)
        axes[r, 1].errorbar(m_d, phi_d, (sm_d, sp_d), fmt=".", color="gray", ms=4, capsize=2)
        fmt = (lambda v: f"{v:.2f}") if p != "M_knee" else (lambda v: f"{v:.1f}")
        axes[r, 0].text(-0.42, 0.5, f"{LABELS[p]}\n[{', '.join(fmt(vals[t]) for t in ('-sw', 'fid', '+sw'))}]",
                        transform=axes[r, 0].transAxes, rotation=90, va="center", ha="center", fontsize=11)
        axes[r, 0].set(xscale="log", yscale="log", xlim=(1e-3, 0.3), ylim=(1e-3, 1e2))
        axes[r, 1].set(yscale="log", ylim=(1e-8, 1e-1))
        axes[r, 2].set(xscale="log", yscale="log", ylim=(1e4, 1e11))
        axes[r, 3].set(xscale="log", yscale="log", ylim=(1e-4, 1e3))
        axes[r, 0].set_ylabel(r"$\omega(\theta)$")
        axes[r, 1].set_ylabel(r"$\phi\,[{\rm Mpc^{-3}\,mag^{-1}}]$")
        axes[r, 2].set_ylabel(r"$M_\ast\,[M_\odot]$")
        axes[r, 3].set_ylabel(r"${\rm SFR}_{100}\,[M_\odot\,{\rm yr}^{-1}]$")
    titles = [rf"ACF, $z={float(raw.d['acf_z'][0]):.2f}$, $M_\ast>10^9M_\odot$",
              rf"UV LF, $z={float(raw.d['uvlf_z'][k8]):.2f}$", "SHMR", rf"SFMS, $z={Z_SCALING}$"]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=13)
    for j, lab in enumerate([r"$\theta\,[{\rm deg}]$", r"$M_{\rm UV}$", r"$M_{\rm h}\,[M_\odot]$",
                             r"$M_\ast\,[M_\odot]$"]):
        axes[-1, j].set_xlabel(lab, fontsize=12)
    handles = [Line2D([0], [0], color=COL[t], lw=2, label=l)
               for t, l in [("-sw", "low"), ("fid", "fiducial"), ("+sw", "high")]]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.52, 1.0),
               fontsize=16)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.05, hspace=0.2, wspace=0.28)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def validation(raw, jac, lines):
    free = [i for i, p in enumerate(raw.params) if p != "M_knee"]
    Fp = np.linalg.inv(raw.d["prior_cov"][np.ix_(free, free)])
    names = [raw.params[i] for i in free]
    ifs, isg = names.index("fstar_norm"), names.index("sigma_SHMR")
    uv_blocks = list(raw.uvlf_keys)
    acf_blocks = [f"ACF_{k}" for k in raw.acf_keys]
    for label, blocks, post_key in [("joint", uv_blocks + acf_blocks, "post_joint"),
                                    ("UV LF only", uv_blocks, "post_uvlf")]:
        F = combine(jac, 0, blocks)[np.ix_(free, free)]
        C = np.linalg.inv(F + Fp)
        sd_f = np.sqrt(np.diag(C))
        corr_f = C[ifs, isg] / (sd_f[ifs] * sd_f[isg])
        lines.append(f"\nValidation, {label}: Fisher+prior vs posterior (prior sd untruncated)")
        lines.append(f"  {'param':16s} {'prior sd':>9s} {'Fisher sd':>10s} {'post sd':>9s}  ratio")
        post = raw.d[post_key][:, free] if post_key in raw.d.files else None
        for j, n in enumerate(names):
            ps = np.std(post[:, j]) if post is not None else np.nan
            lines.append(f"  {n:16s} {raw.prior_sd[free[j]]:9.3f} {sd_f[j]:10.3f} {ps:9.3f}  {sd_f[j] / ps:5.2f}")
        pc = np.corrcoef(post[:, ifs], post[:, isg])[0, 1] if post is not None else np.nan
        lines.append(f"  corr(f*, sigma_SHMR): Fisher {corr_f:+.2f}, posterior {pc:+.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.raw))
    os.makedirs(outdir, exist_ok=True)

    raw = Raw(args.raw)
    jac = jacobians(raw)
    rows = [p for p in ROW_ORDER if p in raw.params]
    ridx = [raw.params.index(p) for p in rows]
    sref = raw.sig_ref[ridx]
    lines = [f"fiducial (posterior median): " + ", ".join(f"{p}={v:.3f}" for p, v in zip(raw.params, raw.d['fid'])),
             f"steps h = {float(raw.d['step_frac'])} posterior sd (M_knee 0.1 dex); S uses sigma_ref = prior sd "
             f"(M_knee {MKNEE_REF} dex)"]

    # --- real data ---
    zlab = {k: f"UV LF $z={z:.1f}$" for k, z in zip(raw.uvlf_keys, raw.d["uvlf_z"])}
    alab = {f"ACF_{k}": f"ACF $z={z:.1f}$" for k, z in zip(raw.acf_keys, raw.d["acf_z"])}
    uvb, acb = list(raw.uvlf_keys), list(alab)
    cols = [(zlab[k], [k]) for k in uvb] + [("UV LF (all)", uvb)] + [(alab[k], [k]) for k in acb] + \
           [("ACF (all)", acb), ("UV LF + ACF", uvb + acb)]
    S = np.column_stack([np.sqrt(np.clip(np.diag(combine(jac, 0, bl)), 0, None))[ridx] * sref for _, bl in cols])
    heatmap(S, rows, [c for c, _ in cols], os.path.join(outdir, "fisher_sensitivity_data.pdf"),
            "Sensitivity to the data used in the inference", sep=[len(uvb) + 0.5, len(uvb) + len(acb) + 1.5])
    lines.append("\nS (data): " + " | ".join(c for c, _ in cols))
    for r, p in enumerate(rows):
        lines.append(f"  {p:16s} " + " ".join(f"{v:8.3f}" for v in S[r]))

    # --- idealized ---
    icol = ["UV LF", "ACF", "SHMR mean", "SFMS mean", "SHMR scatter", "SFMS scatter"]
    Si = np.column_stack([np.sqrt(np.clip(np.diag(combine(jac, 1, [b])), 0, None))[ridx] * sref for b in icol])
    heatmap(Si, rows, icol, os.path.join(outdir, "fisher_sensitivity_idealized.pdf"),
            "Sensitivity for 20% measurements in 8 bins", sep=[1.5],
            group_labels=[(0.5, "Observables"), (3.5, "Galaxy scalings")])
    lines.append("\nS (idealized 20%, 8 bins): " + " | ".join(icol))
    for r, p in enumerate(rows):
        lines.append(f"  {p:16s} " + " ".join(f"{v:8.3f}" for v in Si[r]))

    # --- convergence of the derivatives ---
    lines.append("\nmax |S(h) - S(h/2)| / S(h/2) per block (rows with S > 0.01 only):")
    for (version, b), (J, sig, conv) in jac.items():
        Fd = np.sqrt(np.clip(np.diag(fisher(J, sig)), 0, None)) * raw.sig_ref
        ok = Fd > 0.01
        lines.append(f"  {'data' if version == 0 else 'ideal'} {b:18s} {np.max(conv[ok]) if ok.any() else 0:.3f}")

    validation(raw, jac, lines)
    response_figure(raw, os.path.join(outdir, "parameter_response_appendix.pdf"))

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(outdir, "fisher_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
