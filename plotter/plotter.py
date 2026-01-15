"""
    Set of functions that were used for plotting.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from uvlf import UV_calc_BPASS, bpass_loader, SFH_sampler
import hmf

def get_highphotoz_plot(posterior):
    muv_bins = np.linspace(-23, -17, 10)
    hmf_loc_17 = hmf.MassFunction(z=17, Mmin=5, Mmax=15, dlog10m=0.05)
    hmf_loc_23 = hmf.MassFunction(z=23, Mmin=5, Mmax=15, dlog10m=0.05)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir == "/groups/astro/ivannik/programs/JWST-Inference":
        bpass_read = bpass_loader(
            filename='/groups/astro/ivannik/programs/Stochasticity_sampler/BPASS/spectra-bin-imf135_300.a+00.',
        )
    else:
        bpass_read = bpass_loader()
    vect_func = np.vectorize(bpass_read.get_UV)
    SFR_samp_17 = SFH_sampler(z=17)
    SFR_samp_23 = SFH_sampler(z=23)
    uvlf_z17_uvonly = [UV_calc_BPASS(
        muv_bins,
        np.log10(hmf_loc_17.m),
        hmf_loc_17.dndlnm,
        f_star_norm=10 ** params[0],
        alpha_star=params[3],
        sigma_SHMR=params[1],
        sigma_SFMS_norm=params[4],
        t_star=params[2],
        a_sig_SFR=params[5],
        z=17,
        vect_func = vect_func,
        bpass_read = bpass_read,
        SFH_samp = SFR_samp_17,
    ) for params in posterior]

    uvlf_z23_uvonly = [UV_calc_BPASS(
        muv_bins,
        np.log10(hmf_loc_23.m),
        hmf_loc_23.dndlnm,
        f_star_norm=10 ** params[0],
        alpha_star=params[3],
        sigma_SHMR=params[1],
        sigma_SFMS_norm=params[4],

        t_star=params[2],
        a_sig_SFR=params[5],
        z=23,
        vect_func = vect_func,
        bpass_read = bpass_read,
        SFH_samp = SFR_samp_23,
    ) for params in posterior]

    Muv_Cast_15_20 = np.array([-22.5, -21.5, -20.5, -19.5, -18.5])
    Muv_Cast_15_20_uvlf = np.array([0.06, 0.14, 0.50, 1.2, 5.4]) * 1e-5
    Muv_Cast_15_20_uvlf_p = np.array([0.14, 0.18, 0.4, 1.6, -0.1]) * 1e-5
    Muv_Cast_15_20_uvlf_m = np.array([0.05,0.09,0.25, 0.8, -0.1]) * 1e-5
    Muv_Cast_20_28 = np.array([-22.5, -21.5, -20.5, -19.5, -18.5])
    Muv_Cast_20_28_uvlf = np.array([0.14,0.19,0.55,2.8,9.5]) * 1e-5
    Muv_Cast_20_28_uvlf_p = np.array([-0.1, -0.1, -0.1, -0.1, -0.1]) * 1e-5
    Muv_Cast_20_28_uvlf_m = np.array([-0.1,-0.1,-0.1, -0.1, -0.1]) * 1e-5
    Muv_Per_16_20 = np.array([-19.79, -19.04, -18.29, -17.54])
    Muv_Per_16_20_uvlf = 10**np.array([-4.8, -4.93, -4.26, -3.96])
    Muv_Per_16_20_uvlf_p = 10**np.array([-1,-4.93 + 0.34, -4.26+0.28, -3.96+0.28]) - Muv_Per_16_20_uvlf
    Muv_Per_16_20_uvlf_p[0]=-0.1
    Muv_Per_16_20_uvlf_m = Muv_Per_16_20_uvlf - 10**np.array([-1,- 4.93-1.38,-4.26-0.63,-3.96-0.58])
    Muv_Per_16_20_uvlf_m[0]=-0.1
    Muv_Per_20_24 = np.array([-18.99, -18.24, -17.49])
    Muv_Per_20_24_uvlf = 10**np.array([-4.6, -4.82, -4.66])
    Muv_Per_20_24_uvlf_p = 10**np.array([-1,-4.82+ 0.31, -4.66+0.33]) - Muv_Per_20_24_uvlf
    Muv_Per_20_24_uvlf_p[0]=-1

    Muv_Per_20_24_uvlf_m = Muv_Per_20_24_uvlf - 10**np.array([-1,- 4.82-0.75,-4.66-1.11])
    Muv_Per_20_24_uvlf_m[0]=-1

    fig, ((ax5, ax6)) = plt.subplots(
        1, 2, figsize=(10, 5), sharey=True, sharex=True, gridspec_kw={"hspace": 0}
    )

    # ax5.fill_between(
    #     muv_bins,
    #     uv025_z17,uv095_z17, color='green', alpha=0.3, label=r'posterior Harikane+24')
    # ax5.fill_between(
    #     muv_bins,
    #     uv025_z17_all,uv095_z17_all, color='midnightblue', alpha=0.3, label=r'posterior all')

    label = 'Castellano+25, z=15-20'
    for (muvi_cas, uvlf_cas, sig_cas_p, sig_cas_m) in zip(
            Muv_Cast_15_20,
            Muv_Cast_15_20_uvlf,
            Muv_Cast_15_20_uvlf_p,
            Muv_Cast_15_20_uvlf_m
    ):

        if sig_cas_p > 0:
            ax5.errorbar(
                muvi_cas, uvlf_cas, xerr=np.array([0.5, 0.5]).reshape(2, 1),
                yerr=np.array([sig_cas_m, sig_cas_p]).reshape(2, 1),
                color='red', capsize=5, label=label, marker='<'
            )

            label = None
        else:
            ax5.errorbar(
                muvi_cas, uvlf_cas, xerr=np.array([0.5, 0.5]).reshape(2, 1),
                yerr=np.array([uvlf_cas - uvlf_cas / 5, -sig_cas_p]).reshape(2, 1),
                color='red', capsize=5, label=label, marker='<',
                uplims=True
            )

            label = None
    label = 'Perez-Gonzalez+25, z=16-20'
    label_alt = 'PG+25,reduced by a factor of 3'

    for (muvi_per, uvlf_per, sig_per_p, sig_per_m) in zip(
            Muv_Per_16_20,
            Muv_Per_16_20_uvlf,
            Muv_Per_16_20_uvlf_p,
            Muv_Per_16_20_uvlf_m
    ):
        if sig_per_p > 0:
            ax5.errorbar(
                muvi_per, uvlf_per, xerr=np.array([0.5, 0.5]).reshape(2, 1),
                yerr=np.array([sig_per_m, sig_per_p]).reshape(2, 1),
                color='brown', capsize=5, label=label, marker='>'
            )
            ax5.scatter(
                muvi_per, uvlf_per / 3,  # xerr=np.array([0.3,0.3]).reshape(2,1),
                # yerr = np.array([sig_per_m, sig_per_p]).reshape(2,1),
                color='black', label=label_alt, marker='>', s=100
            )
            label = None
            label_alt = None

        else:
            ax5.errorbar(
                muvi_per, uvlf_per, xerr=np.array([0.3, 0.3]).reshape(2, 1),
                yerr=np.array([uvlf_per - uvlf_per / 5, -sig_per_p]).reshape(2, 1),
                color='brown', capsize=5, label=label, marker='>',
                uplims=True
            )
            label = None

        # ax6.fill_between(
    #     muv_bins,
    #     uv025_z23,uv095_z23, color='green', alpha=0.3, label=r'posterior Harikane+24')
    # ax6.fill_between(
    #     muv_bins,
    #     uv025_z23_all,uv095_z23_all, color='midnightblue', alpha=0.3, label=r'posterior all')

    label = 'Castellano+25, z=20-28'

    for (muvi_cas, uvlf_cas, sig_cas_p, sig_cas_m) in zip(
            Muv_Cast_20_28,
            Muv_Cast_20_28_uvlf,
            Muv_Cast_20_28_uvlf_p,
            Muv_Cast_20_28_uvlf_m
    ):
        print(muvi_cas, uvlf_cas, sig_cas_m, sig_cas_p)
        if sig_cas_p > 0:
            ax6.errorbar(
                muvi_cas, uvlf_cas, xerr=np.array([0.5, 0.5]).reshape(2, 1),
                yerr=np.array([sig_cas_m, sig_cas_p]).reshape(2, 1),
                color='red', capsize=5, label=label, marker='<'
            )
            label = None
        else:
            ax6.errorbar(
                muvi_cas, uvlf_cas, xerr=np.array([0.5, 0.5]).reshape(2, 1),
                yerr=np.array([uvlf_cas - uvlf_cas / 5, -sig_cas_p]).reshape(2, 1),
                color='red', capsize=5, label=label, marker='<',
                uplims=True
            )
            label = None

    label = 'Perez-Gonzalez+25, z=20-24'
    label_alt = 'PG+25,reduced by a factor of 2'

    for (muvi_per, uvlf_per, sig_per_p, sig_per_m) in zip(
            Muv_Per_20_24,
            Muv_Per_20_24_uvlf,
            Muv_Per_20_24_uvlf_p,
            Muv_Per_20_24_uvlf_m
    ):

        if sig_per_p > 0:
            ax6.errorbar(
                muvi_per, uvlf_per, xerr=np.array([0.3, 0.3]).reshape(2, 1),
                yerr=np.array([sig_per_m, sig_per_p]).reshape(2, 1),
                color='brown', capsize=5, label=label, marker='>'
            )
            ax6.scatter(
                muvi_per, uvlf_per / 2,  # xerr=np.array([0.3,0.3]).reshape(2,1),
                # yerr = np.array([sig_per_m, sig_per_p]).reshape(2,1),
                color='black', label=label_alt, marker='>', s=100
            )
            label = None
            label_alt = None
        else:
            ax6.errorbar(
                muvi_per, uvlf_per, xerr=np.array([0.3, 0.3]).reshape(2, 1),
                yerr=np.array([uvlf_per - uvlf_per / 5, -sig_per_p]).reshape(2, 1),
                color='brown', capsize=5, label=label, marker='>',
                uplims=True
            )
            label = None

    fig.subplots_adjust(wspace=0)
    ax5.legend(fontsize=12, loc='lower right')
    ax6.legend(fontsize=12, loc='lower right')

    ax5.text(-22.5, 1e-3, 'z=14-20', fontsize=18)
    ax6.text(-22.5, 1e-3, 'z=20-30', fontsize=18)

    ax5.set_yscale('log')

    ax5.set_xlabel(r'M$_{\rm UV}$[mag]', fontsize=18)
    ax6.set_xlabel(r'M$_{\rm UV}$[mag]', fontsize=18)

    ax5.set_ylabel(r'$\Phi$[mag$^{-1}$Mpc$^{-3}$]', fontsize=18)

    ax5.tick_params(labelsize=18)
    ax6.tick_params(labelsize=18)

    ax5.set_ylim(1e-8, 1e-2)
    ax6.set_ylim(1e-8, 1e-2)

    plt.savefig(
        '/home/inikolic/projects/UVLF_FMs/run_speed/analysis_post/diff_num_PG25.pdf',
        bbox_inches='tight')

def corner_for_the_paper():
    params = [
        r"f$_{\ast}$",
        r"$\sigma_{\rm SHMR}$",
        r"$t_{\ast}$",
        r"$\alpha_{\ast}$",
        r"$\sigma_{\rm SFMS, 0}$",
        r"$a_{\sigma, \rm SFR}$",
        r"M$_{\rm knee}$"]
    # priors = [(-2.0, 1.0), (0.001, 1.0), (0.001, 1.0), (0.0, 1.0), (0.001, 1.2),
    #               (-1.0, 0.5)]

    sampls_diag = np.random.multivariate_normal(mean,
                                                cov_matr_diag_changed * 6 / 5,
                                                size=40000)
    priors = [(-6.0, 1.0), (0.001, 2.0), (0.001, 1.0), (0.0, 2.0),
              (0.001, 1.5), (-1.0, 0.5), (11.5, 16.0)]
    mins = np.array([b[0] for b in priors])
    maxs = np.array([b[1] for b in priors])
    mask = np.all((sampls_diag >= mins) & (sampls_diag <= maxs), axis=1)
    samps_diag_Mknee_done = sampls_diag[mask]
    dataset2_padded = np.hstack(
        [
            np.random.random((post_FLARES_transormed.shape[1], 2)) - 10,
            (post_FLARES_transormed[:1, :]).T,
            np.random.random((post_FLARES_transormed.shape[1], 1)) - 10,
            (post_FLARES_transormed[1:, :]).T,
            np.random.random((post_FLARES_transormed.shape[1], 1)) - 10
        ]
    )
    import matplotlib.lines as mlines

    fig = corner.corner(
        post_FL_transormed.T,
        bins=40,
        smooth=True,
        labels=params,
        range=[(-6.0, 0.0), (0.05, 0.8), (0.01, 1.0), (0.1, 1.0), (0.0, 1.0),
               (-0.3, 0.3), (11.0, 16.0)],

        # levels= (1-np.exp(-2./2),1-np.exp(-4./2.)),#, 1-np.exp(-5/2.)),
        levels=(0.68, 0.95),
        # quantiles=(0.16, 0.84),
        plot_datapoints=False, plot_density=False,
        weights=np.ones(len(post_FL_transormed.T)) / len(post_FL_transormed.T),
        label_kwargs={'fontsize': 18},
        color='#a6cee3',
        label='FirstLight',
        contour_kwargs={"linewidths": 3}, hist_kwargs={
            "linewidth": 2.5,
            "histtype": "step",
        },
    )

    fig2 = corner.corner(
        post_ASTRID_transormed.T,
        bins=40,
        fig=fig,
        smooth=True,
        labels=params,
        range=[(-6.0, 0.0), (0.05, 0.8), (0.01, 1.0), (0.1, 1.0), (0.0, 1.0),
               (-0.3, 0.3), (11.0, 16.0)],

        # levels= (1-np.exp(-2./2),1-np.exp(-4./2.)),#, 1-np.exp(-5/2.)),
        levels=(0.68, 0.95),
        # quantiles=(0.16, 0.84),
        # range=ranges,
        plot_datapoints=False, plot_density=False,
        weights=np.ones(len(post_ASTRID_transormed.T)) / len(
            post_ASTRID_transormed.T),
        label_kwargs={'fontsize': 18},
        color='#1f78b4',
        label='ASTRID', contour_kwargs={"linewidths": 3}, hist_kwargs={
            "linewidth": 2.5,
            "histtype": "step",
        },
    )

    fig3 = corner.corner(
        post_SERRA_transormed.T,
        bins=40,
        smooth=True,
        labels=params,
        fig=fig,
        range=[(-6.0, 0.0), (0.05, 0.8), (0.01, 1.0), (0.1, 1.0), (0.0, 1.0),
               (-0.3, 0.3), (11.0, 16.0)],

        # levels= (1-np.exp(-2./2),1-np.exp(-4./2.)),#, 1-np.exp(-5/2.)),
        levels=(0.68, 0.95),
        # quantiles=(0.16, 0.84),
        # range=ranges,
        plot_datapoints=False, plot_density=False,
        weights=np.ones(len(post_SERRA_transormed.T)) / len(
            post_SERRA_transormed.T),
        label_kwargs={'fontsize': 18},
        color='#b2df8a',
        label='SERRA', contour_kwargs={"linewidths": 3}, hist_kwargs={
            "linewidth": 2.5,
            "histtype": "step",
        },
    )

    fig3 = corner.corner(
        post_FIRE_transormed.T,
        bins=40,
        smooth=True,
        labels=params,
        fig=fig,
        range=[(-6.0, 0.0), (0.05, 0.8), (0.01, 1.0), (0.1, 1.0), (0.0, 1.0),
               (-0.3, 0.3), (11.0, 16.0)],

        # levels= (1-np.exp(-2./2),1-np.exp(-4./2.)),#, 1-np.exp(-5/2.)),
        levels=(0.68, 0.95),
        # quantiles=(0.16, 0.84),
        # range=ranges,
        plot_datapoints=False, plot_density=False,
        weights=np.ones(len(post_FIRE_transormed.T)) / len(
            post_FIRE_transormed.T),
        label_kwargs={'fontsize': 18},
        color='#33a02c',
        label='FIRE', contour_kwargs={"linewidths": 3}, hist_kwargs={
            "linewidth": 2.5,
            "histtype": "step",
        },
    )
    fig5 = corner.corner(
        samps_diag_Mknee_done,
        bins=40,
        smooth=True,
        labels=params,
        fig=fig,
        range=[(-6.0, 0.0), (0.05, 0.8), (0.01, 1.0), (0.1, 1.0), (0.0, 1.0),
               (-0.3, 0.3), (11.0, 16.0)],

        # levels= (1-np.exp(-2./2),1-np.exp(-4./2.)),#, 1-np.exp(-5/2.)),
        levels=(0.68, 0.95),
        # quantiles=(0.16, 0.84),
        # range=ranges,
        plot_datapoints=False, plot_density=False,
        weights=np.ones(len(samps_diag_Mknee_done)) / len(
            samps_diag_Mknee_done),
        label_kwargs={'fontsize': 18},
        color='black', zorder=0,
        label='Combined', contour_kwargs={"linewidths": 2.0}, hist_kwargs={
            "linewidth": 2.0,
            "histtype": "step",
        },
    )

    fig4 = corner.corner(
        dataset2_padded,
        bins=40,
        smooth=True,
        labels=params,
        fig=fig,
        # levels= (1-np.exp(-2./2),1-np.exp(-4./2.)),#, 1-np.exp(-5/2.)),
        levels=(0.68, 0.95),
        range=[(-10, -9), (-10, -9), (0.01, 1.0), (-10, -9), (0.0, 1.0),
               (-0.3, 0.3), (-10, -9)],

        # range=[(-1.2,0.6), (0.05,0.6), (0.01,0.6), (0.1,1.0),(0.0,0.8), (-0.3,0.3)],
        # quantiles=(0.16, 0.84),
        # range=ranges,
        plot_datapoints=False, plot_density=False, plot_contours=True,
        weights=np.ones(len(dataset2_padded)) / len(dataset2_padded),
        label_kwargs={'fontsize': 18},
        color='#fb9a99',
        label='FLARES',
        zorder=-1, lw=3, contour_kwargs={"linewidths": 3}, hist_kwargs={
            "linewidth": 2.5,
            "histtype": "step",
        },
    )

    # fig3 = corner.corner(
    #     sampls_diag,
    #     bins=40 ,
    #     smooth=True,
    #     labels = params,
    #     fig=fig,
    #     range=[(-6.0,0.0), (0.05,1.0), (0.01,1.0), (0.1,1.0),(0.0,1.0), (-0.3,0.3),(11.0,16.0)],

    #     #levels= (1-np.exp(-2./2),1-np.exp(-4./2.)),#, 1-np.exp(-5/2.)),
    #     levels = (0.68,0.95),
    #     #quantiles=(0.16, 0.84),
    #     #range=ranges,
    #     plot_datapoints=False, plot_density=False,
    #     weights=np.ones(len(sampls_diag))/len(sampls_diag),
    #     label_kwargs = {'fontsize':18},
    #     color='gray',
    #     label='Combined'
    # )

    def set_corner_ranges(fig, ranges):
        """
        Force all axes in an existing corner figure to use the provided ranges.

        ranges: list of (min,max) with length ndim, in the same order as params.
        """
        ndim = len(ranges)

        # corner uses ndim*ndim axes; this reshape matches corner's internal layout
        axes = np.array(fig.get_axes()).reshape((ndim, ndim))

        # Diagonal: 1D histograms
        for i in range(ndim):
            axes[i, i].set_xlim(ranges[i])

        # Lower triangle: 2D panels
        for y in range(1, ndim):
            for x in range(0, y):
                ax = axes[y, x]
                ax.set_xlim(ranges[x])
                ax.set_ylim(ranges[y])

    # ---- call this right before savefig ----
    ranges = [(-5.0, 0.0), (0.05, 0.8), (0.01, 1.0), (0.1, 1.0),
              (0.0, 1.0), (-0.3, 0.3), (11.0, 16.0)]

    set_corner_ranges(fig, ranges)

    def style_corner_axes(fig, labelsize=22, ticksize=16, tickwidth=1.8,
                          ticklength=6):
        for ax in fig.get_axes():
            # Axis labels (parameter names)
            ax.xaxis.label.set_size(labelsize)
            ax.yaxis.label.set_size(labelsize)

            # Tick labels + ticks
            ax.tick_params(
                axis='both',
                which='major',
                labelsize=ticksize,
                width=tickwidth,
                length=ticklength,
                direction='in'
            )

    style_corner_axes(
        fig,
        labelsize=22,  # parameter names
        ticksize=16,  # numbers on axes
        tickwidth=1.8,
        ticklength=6
    )
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', 'gray', ]
    sample_labels = ['First Light', 'ASTRID', 'SERRA', 'FIRE', 'FLARES',
                     'Combined']
    plt.legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=sample_labels[i], lw=3)
            for i in range(len(colors))
        ],
        fontsize=40, frameon=True,
        bbox_to_anchor=(1, len(post_FL_transormed)), loc="upper right"
    )
    plt.savefig(
        '/home/inikolic/projects/UVLF_FMs/priors/prior_for_the_paper.pdf',
        bbox_inches='tight')