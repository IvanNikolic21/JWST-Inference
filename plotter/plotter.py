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