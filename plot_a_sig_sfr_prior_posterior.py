"""Prior vs. posterior for a_sig_SFR (mass-slope of the SFMS scatter).

The prior is the simulation-derived Gaussian used by mcmc.py's prior()
(cov_matr_uv.txt / 5, means_uv.txt), truncated to the sampling limits.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, gaussian_kde

script_dir = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(script_dir, '..', '..')
i_par = 5  # a_sig_SFR position in the params list
limits = (-1.0, 0.5)

cov = np.loadtxt(os.path.join(script_dir, 'cov_matr_uv.txt')) / 5
mu = np.loadtxt(os.path.join(script_dir, 'means_uv.txt'))
m, s = mu[i_par], np.sqrt(cov[i_par, i_par])
prior = truncnorm((limits[0] - m) / s, (limits[1] - m) / s, loc=m, scale=s)

runs = {
    'UVLF': ('Willot_faint', 'tab:green'),
    'UVLF + ACF': ('Willot_clustering_faint', 'tab:red'),
}

x = np.linspace(-0.3, 0.15, 600)
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.plot(x, prior.pdf(x), color='k', ls='--', lw=2, label='Prior')
print(f'Prior:      P(a<0) = {prior.cdf(0):.3f}   mean={m:.3f} sd={s:.3f}')

for label, (d, c) in runs.items():
    a = np.loadtxt(os.path.join(base, d, 'post_equal_weights.dat'))[:, i_par]
    ax.plot(x, gaussian_kde(a)(x), color=c, lw=2, label=label)
    lo, med, hi = np.percentile(a, [16, 50, 84])
    print(f'{label:11s} P(a<0) = {np.mean(a < 0):.3f}   '
          f'median={med:.3f} [{lo:.3f}, {hi:.3f}]  sd={a.std():.3f}  N={len(a)}')

ax.axvline(0, color='gray', lw=1)
ax.set_xlim(x[0], x[-1])
ax.set_ylim(bottom=0)
ax.set_xlabel(r'$a_{\sigma,\rm SFMS}$')
ax.set_ylabel('Probability density')
ax.legend(frameon=False)
fig.tight_layout()
out = os.path.join(script_dir, 'a_sig_SFR_prior_posterior')
fig.savefig(out + '.pdf')
fig.savefig(out + '.png', dpi=150)
