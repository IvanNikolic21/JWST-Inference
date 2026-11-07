"""Fig. uvlf_post: posterior predictive UV LFs of the UVLF-only and UVLF + ACF runs.

    python plot_uvlfs_all.py RUNS_DIR --out UVLFs.pdf                    # main_uvlf, main_joint
    python plot_uvlfs_all.py RUNS_DIR --pair willott_z9 --out UVLFs.pdf  # runs with Willott z = 9

RUNS_DIR holds the run folders (best_prior_v4/). Each run's UVLFs_*.txt files have
one ROW per redshift of the --z_list that uvlf_a_post.py was run with, and 20 columns
on muv_ar = linspace(-25, -16, 20). The main runs used the default 7-redshift list,
the z=9 runs an 8-redshift list with 8.99 inserted, so rows are selected by redshift,
never by position. main_joint also holds UVLFs_highz*.txt files, which are skipped.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from observations import Observations

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("runs", help="directory holding the run folders")
parser.add_argument("--pair", choices=("willott_z9", "main"), default="main",
                    help="which UVLF-only / UVLF + ACF pair to plot")
parser.add_argument("--out", default="UVLFs.pdf")
args = parser.parse_args()
RUNS = args.runs
OUT = args.out


def gimme_dust(Muv, beta=None):
    """A_UV for the beta(M_UV) of uvlf.gimme_dust, or a fixed beta, or 'Hubble'."""
    if beta is None:
        beta = -0.17 * Muv - 5.40
    elif str(beta) == 'Hubble':
        beta = -0.2 * (Muv + 19.5) - 2
    Auv = 4.43 + 1.99 * beta
    return np.clip(Auv, 0, 5)


Z_MAIN = [6.0, 7.99, 10.23, 11.0, 11.73, 12.5, 14.0]          # uvlf_a_post.py default
Z_WZ9 = [6.0, 7.99, 8.99, 10.23, 11.0, 11.73, 12.5, 14.0]     # run_post_wz9_v4.sh
RUN_ZLIST = {f"{args.pair}_uvlf": Z_MAIN if args.pair == "main" else Z_WZ9,
             f"{args.pair}_joint": Z_MAIN if args.pair == "main" else Z_WZ9}

# Model redshift shown in each panel. Panel 5 holds the Willott z~12 bin, whose
# effective redshift is 11.73; use 12.5 instead to match the other z~12.5 data.
PANEL_Z = {"z6": 6.0, "z8": 7.99, "z10": 10.23, "z11": 11.0, "z12_5": 11.73, "z14": 14.0}
PANEL_TEXT = {"z6": "z=6", "z8": "z=8", "z10": "z=10", "z11": "z=11", "z12_5": "z=12", "z14": "z=14"}

PCT = (2.5, 97.5)   # 95% credible interval, as in the paper caption

muv_ar = np.linspace(-25, -16, 20)


def load_run(name):
    """(N_samples, N_z, N_muv) array of predictive UV LFs for one run."""
    files = sorted(f for f in glob.glob(os.path.join(RUNS, name, "UVLFs_*.txt"))
                   if "highz" not in os.path.basename(f))
    if not files:
        raise FileNotFoundError(f"no UVLFs_*.txt in {RUNS}/{name}")
    arr = np.array([np.loadtxt(f) for f in files])
    nz = len(RUN_ZLIST[name])
    if arr.shape[1] != nz:
        raise ValueError(f"{name}: files have {arr.shape[1]} redshift rows, expected {nz}")
    print(f"{name}: {arr.shape[0]} samples, {nz} redshifts")
    return arr


def band(name, arr, z):
    """Lower/upper percentile of the UV LF at redshift z (row picked by value)."""
    zl = np.array(RUN_ZLIST[name])
    i = int(np.argmin(np.abs(zl - z)))
    if abs(zl[i] - z) > 1e-6:
        raise ValueError(f"{name}: z={z} not in its z_list {list(zl)}")
    lo, hi = np.percentile(arr[:, i, :], PCT, axis=0)
    return lo, hi


data = {name: load_run(name) for name in RUN_ZLIST}
bands = {name: {p: band(name, data[name], z) for p, z in PANEL_Z.items()} for name in RUN_ZLIST}

observations_inst = Observations(True, True)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
    3, 2, figsize=(10, 12), sharey=True, sharex=True, gridspec_kw={"hspace": 0}
)
axes = {"z6": ax1, "z8": ax2, "z10": ax3, "z11": ax4, "z12_5": ax5, "z14": ax6}

ax1.errorbar(
    *observations_inst.get_obs_uvlf_z6_Bouwens21(),
    color='gray', capsize=10, ms=15, label='HST, Bouwens+21, z=6', marker='.', ls='none'
)
ax2.errorbar(
    *observations_inst.get_obs_uvlf_z8_Bouwens21(),
    color='gray', capsize=10, ms=15, label='HST, Bouwens+21, z=8', marker='.', ls='none'
)
ax3.errorbar(
    *observations_inst.get_obs_uvlf_z10_Bouwens21(),
    color='gray', capsize=10, ms=15, label='HST, Bouwens+21, z=10', marker='.', ls='none'
)

ax1_leg = ax1.legend(fontsize=10, loc='lower right')
ax2_leg = ax2.legend(fontsize=10, loc='upper left')
ax3_leg = ax3.legend(fontsize=10, loc='upper left')
ax1.add_artist(ax1_leg)
ax2.add_artist(ax2_leg)
ax3.add_artist(ax3_leg)

don_10 = ax3.errorbar(
    observations_inst.get_obs_uvlf_z10_Donnan24()[0],
    observations_inst.get_obs_uvlf_z10_Donnan24()[1],
    observations_inst.get_obs_uvlf_z10_Donnan24()[2][::-1],
    color='gray', capsize=10, ms=8, label=r'Donnan+24, z=10', marker='<', ls='none')
don_11 = ax4.errorbar(
    observations_inst.get_obs_uvlf_z11_Donnan24()[0],
    observations_inst.get_obs_uvlf_z11_Donnan24()[1],
    observations_inst.get_obs_uvlf_z11_Donnan24()[2][::-1],
    color='gray', capsize=10, ms=8, label=r'Donnan+24, z=11', marker='<', ls='none')
don_12 = ax5.errorbar(
    observations_inst.get_obs_uvlf_z12_5_Donnan24()[0],
    observations_inst.get_obs_uvlf_z12_5_Donnan24()[1],
    observations_inst.get_obs_uvlf_z12_5_Donnan24()[2][::-1],
    color='gray', capsize=10, ms=8, label=r'Donnan+24, z=12.5', marker='<', ls='none')

handles_2 = []
handles_3 = [don_10]
handles_4 = [don_11]
handles_5 = [don_12]
handles_6 = []

fink6 = ax6.errorbar(
    observations_inst.get_obs_uvlf_z14_Finkelstein24()[0],
    observations_inst.get_obs_uvlf_z14_Finkelstein24()[1],
    observations_inst.get_obs_uvlf_z14_Finkelstein24()[2][::-1],
    color='gray', ms=8, capsize=5, label=r'Finkelstein+24, z=14', marker='^', ls='none')
fink4 = ax4.errorbar(
    observations_inst.get_obs_uvlf_z11_Finkelstein24()[0],
    observations_inst.get_obs_uvlf_z11_Finkelstein24()[1],
    observations_inst.get_obs_uvlf_z11_Finkelstein24()[2][::-1],
    color='gray', ms=8, capsize=5, label=r'Finkelstein+24, z=11', marker='^', ls='none')
handles_6.append(fink6)
handles_4.append(fink4)

whit6 = ax6.errorbar(
    observations_inst.get_obs_uvlf_z14_3_Whitler25()[0],
    observations_inst.get_obs_uvlf_z14_3_Whitler25()[1],
    observations_inst.get_obs_uvlf_z14_3_Whitler25()[2][::-1],
    color='gray', ms=8, capsize=5, label=r'Whitler+25, z=14.3', marker='>', ls='none')
whit5 = ax5.errorbar(
    observations_inst.get_obs_uvlf_z12_8_Whitler25()[0],
    observations_inst.get_obs_uvlf_z12_8_Whitler25()[1],
    observations_inst.get_obs_uvlf_z12_8_Whitler25()[2][::-1],
    color='gray', ms=8, capsize=5, label=r'Whitler+25, z=12.8', marker='>', ls='none')
whit3 = ax3.errorbar(
    observations_inst.get_obs_uvlf_z9_8_Whitler25()[0],
    observations_inst.get_obs_uvlf_z9_8_Whitler25()[1],
    observations_inst.get_obs_uvlf_z9_8_Whitler25()[2][::-1],
    color='gray', ms=8, capsize=5, label=r'Whitler+25, z=9.8', marker='>', ls='none')
handles_6.append(whit6)
handles_5.append(whit5)
handles_3.append(whit3)


def plot_harikane(ax, obs, label, handles):
    """Harikane+24 points; negative errors mark lower limits. The legend handle
    is appended once, after the first point is drawn (the original loops
    referenced har3 before assignment and never added the z=14 handle)."""
    for muvi_har, uvlf_har, sig_har_p, sig_har_m in zip(obs[0], obs[1], obs[2][0], obs[2][1]):
        if sig_har_p > 0:
            h = ax.errorbar(
                muvi_har, uvlf_har, np.array([sig_har_m, sig_har_p]).reshape(2, 1),
                color='gray', ms=8, capsize=5, label=label, marker='o')
        else:
            h = ax.errorbar(
                muvi_har, uvlf_har, xerr=0, yerr=np.array([-sig_har_m, -sig_har_p]).reshape(2, 1),
                color='gray', ms=8, capsize=5, label=label, marker='o', lolims=True)
        if label is not None:
            handles.append(h)
        label = None


plot_harikane(ax2, observations_inst.get_obs_uvlf_z8_Harikane24(), r'Harikane+24, z=8', handles_2)
plot_harikane(ax3, observations_inst.get_obs_uvlf_z10_Harikane24(), r'Harikane+24, z=10', handles_3)
plot_harikane(ax5, observations_inst.get_obs_uvlf_z12_Harikane24(), r'Harikane+24, z=12', handles_5)
plot_harikane(ax6, observations_inst.get_obs_uvlf_z14_Harikane24(), r'Harikane+24, z=14', handles_6)

# ── model bands: rows picked by redshift; same dust conventions as before ──
dust_beta = {"z6": 'Hubble', "z8": 'Hubble', "z10": None, "z11": -2.5, "z12_5": -2.5, "z14": -2.5}


def muv_dusty(p):
    b = dust_beta[p]
    return muv_ar + (gimme_dust(muv_ar) if b is None else gimme_dust(muv_ar, beta=b))


STYLE = {f"{args.pair}_uvlf": dict(color='#4daf4a', ls='-', fill=True),
         f"{args.pair}_joint": dict(color='#e41a1c', ls='-', fill=True)}
for name, st in STYLE.items():
    for p, ax in axes.items():
        lo, hi = bands[name][p]
        x = muv_dusty(p)
        if st["fill"]:
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.4)
        ax.plot(x, lo, color=st["color"], lw=2, ls=st["ls"])
        ax.plot(x, hi, color=st["color"], lw=2, ls=st["ls"])

for ax in axes.values():
    ax.set_yscale('log')

wil_obs = {
    ax2: (observations_inst.get_obs_uvlf_z8_Willot23(), r'Willott+24, z=8'),
    ax3: (observations_inst.get_obs_uvlf_z10_Willot23(), r'Willott+24, z=10'),
    ax5: (observations_inst.get_obs_uvlf_z12_Willot23(), r'Willott+24, z=12'),
}
wil_handles = {}
for ax, (obs, label) in wil_obs.items():
    m_used = obs[0] >= -20
    err = obs[2][::-1]
    wil_handles[ax] = ax.errorbar(
        obs[0][m_used], obs[1][m_used], (err[0][m_used], err[1][m_used]),
        color='red', capsize=5, label=label, marker='^', ls='none', ms=15)
    m_cut = ~m_used
    ax.errorbar(
        obs[0][m_cut], obs[1][m_cut], (err[0][m_cut], err[1][m_cut]),
        color='gray', capsize=5, ms=8, marker='^', ls='none', alpha=0.5)

mcleod4 = ax4.errorbar(
    observations_inst.get_obs_uvlf_z11_McLeod23()[0],
    observations_inst.get_obs_uvlf_z11_McLeod23()[1],
    observations_inst.get_obs_uvlf_z11_McLeod23()[2],
    color='gray', ms=8, capsize=5, label=r'McLeod+24, z=11', marker='v', ls='none')
obs = observations_inst.get_obs_uvlf_z12_Weibel25()
weibel5 = ax5.errorbar(obs[0], obs[1], obs[2][::-1], color='gray', ms=8, capsize=5,
                       label=r'Weibel+25, $z\sim13$', marker='x', ls='none')
obs = observations_inst.get_obs_uvlf_z13_5_McLeod26()
mcleod6 = ax6.errorbar(obs[0], obs[1], obs[2][::-1], color='gray', ms=8, capsize=5,
                       label=r'McLeod+26, $z\sim13.5$', marker='v', ls='none')
m_lim, phi_lim = observations_inst.get_obs_uvlf_z13_5_McLeod26_limit()
ax6.errorbar(m_lim, phi_lim, yerr=0.5 * phi_lim, uplims=True, color='gray', capsize=5, fmt='.')

handles_2.append(wil_handles[ax2])
handles_3.append(wil_handles[ax3])
handles_4.append(mcleod4)
handles_5.append(wil_handles[ax5])
handles_5.append(weibel5)
handles_6.append(mcleod6)

fig.subplots_adjust(wspace=0)
LEG = dict(fontsize=10, labelspacing=0.2, borderpad=0.3, handletextpad=0.4)
ax2.legend(handles=handles_2, loc='lower right', **LEG)
ax3.legend(handles=handles_3, loc='lower right', **LEG)
ax4.legend(handles=handles_4, loc='lower right', **LEG)
ax5.legend(handles=handles_5, loc='lower right', **LEG)
ax6.legend(handles=handles_6, loc='upper right', **LEG)

text_y = {"z6": 10**-3.2, "z8": 10**-3.2, "z10": 10**-3.5, "z11": 10**-3.2, "z12_5": 10**-3.2, "z14": 10**-2.5}
for p, ax in axes.items():
    ax.text(-21.8, text_y[p], PANEL_TEXT[p], fontsize=18)
    ax.tick_params(labelsize=18)
    ax.set_ylim(10**-6.8, 1e-2)
    ax.set_xlim(-22.2, -16.4)
ax1.set_xlim(-22.5, -16.4)
ax1.set_yticks([10.0**e for e in range(-6, -2)])   # shared y; a 10^-2 label would hit the panel above
for ax in (ax5, ax6):
    ax.set_xlabel(r'$M_{\rm UV}$ [mag]', fontsize=18)
fig.supylabel(r'$\mathrm{d}n/\mathrm{d}M_{\rm UV}$ [mag$^{-1}$ Mpc$^{-3}$]', fontsize=24, x=0.01)

legend_handles = [mlines.Line2D([], [], color='#4daf4a', label='UVLF only'),
                  mlines.Line2D([], [], color='#e41a1c', label='UVLF + ACF')]
ax1.add_artist(ax1.legend(handles=legend_handles, fontsize=14, frameon=True,
                          loc='lower right', bbox_to_anchor=(0.98, 0.14)))
plt.savefig(OUT, bbox_inches='tight')
print(f"Saved: {OUT}")
