"""
"""
import numpy as np
import pylab as plt
from pspipe_utils import simulation, best_fits, pspipe_list
from pspy import pspy_utils, so_spectra, sph_tools, so_map, so_cov, so_window, so_mcm, so_dict
from pixell import curvedsky
import time, sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

tuto_data_dir = "tuto_data"
result_dir = "result_simulation"
pspy_utils.create_directory(result_dir)

ncomp, niter = 3, 0
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
type = "Dl"

surveys = d["surveys"]
n_splits = d["n_splits"]
binned_mcm = d["binned_mcm"]
lmax = d["lmax"]
lmax_sim = lmax + 500
n_sims = d["n_sims"]

binning_file = f"{tuto_data_dir}/binning.dat"

spec_name_list = pspipe_list.get_spec_name_list(d)
freq_list = pspipe_list.get_freq_list(d)

f_name_cmb = tuto_data_dir + "/cmb.dat"
f_name_noise = tuto_data_dir + "/mean_{}x{}_{}_noise.dat"
f_name_fg = tuto_data_dir + "/fg_{}x{}.dat"
f_name_beam = tuto_data_dir + "/beam_{}_{}.dat"

arrays, bl, nu_eff, window = {}, {}, {}, {}
for sv in surveys:
    arrays[sv] = d[f"arrays_{sv}"]
    for ar in arrays[sv]:
        l, bl[sv, ar] = np.loadtxt(f"{f_name_beam.format(sv, ar)}", unpack=True)
        nu_eff[sv, ar] = d[f"nu_eff_{sv}_{ar}"]
        window[sv, ar] = so_map.read_map(f"{tuto_data_dir}/window_{sv}_{ar}.fits")

ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax_sim, spectra)
l, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, freq_list, lmax_sim, spectra)
noise_mat = {}
for sv in surveys:
    l, noise_mat[sv] = simulation.noise_matrix_from_files(f_name_noise,
                                                          sv,
                                                          arrays[sv],
                                                          lmax_sim,
                                                          n_splits[sv],
                                                          spectra)

template = so_map.read_map(f"{tuto_data_dir}/template.fits")

ps_all = {}
for iii in range(n_sims):
    t = time.time()
    alms_cmb = curvedsky.rand_alm(ps_mat, lmax=lmax_sim, dtype="complex64")
    fglms = simulation.generate_fg_alms(fg_mat, freq_list, lmax_sim)
    sim_alm = {}

    for sv in surveys:
        signal_alms = {}
        for ar in arrays[sv]:
            signal_alms[ar] = alms_cmb + fglms[nu_eff[sv, ar]]
            for i in range(3):
                signal_alms[ar][i] = curvedsky.almxfl(signal_alms[ar][i], bl[sv, ar])

        for k in range(n_splits[sv]):
            noise_alms = simulation.generate_noise_alms(noise_mat[sv], arrays[sv], lmax_sim)
            for ar in arrays[sv]:
                split = sph_tools.alm2map(signal_alms[ar] + noise_alms[ar], template)

                sim_alm[sv, ar, k] = sph_tools.get_alms(split, (window[sv, ar], window[sv, ar]), niter, lmax)

    for spec_name in spec_name_list:
        na, nb = spec_name.split("x")
        sv_a, ar_a = na.split("&")
        sv_b, ar_b = nb.split("&")

        spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
        mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{tuto_data_dir}/{spec_name}", spin_pairs=spin_pairs)

        ps_dict = {}
        for spec in spectra:
            ps_dict[spec, "auto"] = []
            ps_dict[spec, "cross"] = []
                    
        for s1 in range(n_splits[sv_a]):
            for s2 in range(n_splits[sv_b]):
                if (sv_a == sv_b) & (ar_a == ar_b) & (s1 > s2) : continue

                l, ps_master = so_spectra.get_spectra_pixell(sim_alm[sv_a, ar_a, s1],
                                                             sim_alm[sv_b, ar_b, s2],
                                                             spectra=spectra)

                lb, ps = so_spectra.bin_spectra(l,
                                                ps_master,
                                                binning_file,
                                                lmax,
                                                type=type,
                                                mbb_inv=mbb_inv,
                                                spectra=spectra,
                                                binned_mcm=binned_mcm)
                                                            
                for count, spec in enumerate(spectra):
                    if (s1 == s2) & (sv_a == sv_b): ps_dict[spec, "auto"] += [ps[spec]]
                    else: ps_dict[spec, "cross"] += [ps[spec]]

        ps_dict_auto_mean, ps_dict_cross_mean, ps_dict_noise_mean = {}, {}, {}
        for spec in spectra:
            ps_dict_cross_mean[spec] = np.mean(ps_dict[spec, "cross"], axis=0)
            if ar_a == ar_b and sv_a == sv_b:
                # Average TE / ET so that for same array same season TE = ET
                ps_dict_cross_mean[spec] = (np.mean(ps_dict[spec, "cross"], axis=0) + np.mean(ps_dict[spec[::-1], "cross"], axis=0)) / 2.
                            
            if sv_a == sv_b:
                ps_dict_auto_mean[spec] = np.mean(ps_dict[spec, "auto"], axis=0)
                ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / n_splits[sv_a]

        spec_name_cross = f"{type}_{spec_name}_cross_%05d" % iii
        so_spectra.write_ps(f"{result_dir}/{spec_name_cross}.dat", lb, ps_dict_cross_mean, type, spectra=spectra)
        if sv_a == sv_b:
            spec_name_auto = f"{type}_{spec_name}_auto_%05d" % iii
            spec_name_noise = f"{type}_{spec_name}_noise_%05d" % iii
            so_spectra.write_ps(f"{result_dir}/{spec_name_auto}.dat", lb, ps_dict_auto_mean, type, spectra=spectra)
            so_spectra.write_ps(f"{result_dir}/{spec_name_noise}.dat", lb, ps_dict_noise_mean, type, spectra=spectra)

        if iii == 0:
            ps_all[spec_name, "cross"] = []
            ps_all[spec_name, "noise"] = []

        ps_all[spec_name, "cross"] += [ps_dict_cross_mean]
        ps_all[spec_name, "noise"] += [ps_dict_noise_mean]

    print("sim %05d/%05d took %.02f s to compute" % (iii, n_sims, time.time() - t))

##########################################################################
# Let's look at the spectra
##########################################################################

l_cmb, cmb_dict = best_fits.cmb_dict_from_file(f_name_cmb, lmax, spectra)
l_fg, fg_dict = best_fits.fg_dict_from_files(f_name_fg, freq_list, lmax, spectra)
l_noise, nl_dict = best_fits.noise_dict_from_files(f_name_noise,  surveys, arrays, lmax, spectra)
l_beam, bl_dict = best_fits.beam_dict_from_files(f_name_beam, surveys, arrays, lmax)

l_cmb, ps_all_th, nl_all_th = best_fits.get_all_best_fit(spec_name_list,
                                                         l_cmb,
                                                         cmb_dict,
                                                         fg_dict,
                                                         nu_eff,
                                                         spectra,
                                                         nl_dict=nl_dict)

n_bins = len(lb)
for spec_name in spec_name_list:
    na, nb = spec_name.split("x")
    sv_a, ar_a = na.split("&")
    sv_b, ar_b = nb.split("&")

    for spec_type in ["cross", "noise"]:
        mean_a, _, mc_cov = so_cov.mc_cov_from_spectra_list(ps_all[spec_name, spec_type],
                                                            ps_all[spec_name, spec_type],
                                                            spectra=spectra)

        plt.figure(figsize=(18, 14))

        for count, spec in enumerate(spectra):
            mc_cov_select = so_cov.selectblock(mc_cov, spectra, n_bins, block=spec+spec)
            std = np.sqrt(mc_cov_select.diagonal())
            plt.subplot(3, 3, count + 1)
            plt.errorbar(lb, mean_a[spec], std, fmt=".")

            if spec_type=="cross":
                if  spec == "TT":
                    plt.semilogy()
                plt.plot(l_cmb, cmb_dict[spec], label="CMB")
                plt.plot(l_cmb, ps_all_th[f"{sv_a}&{ar_a}", f"{sv_b}&{ar_b}", spec], label="CMB+fg")
                plt.xlabel("$\ell$", fontsize=16)
                plt.ylabel("$D^{%s}_\ell$" % spec, fontsize=16)
                plt.title(spec_name)
                plt.legend(fontsize=14)
            else:
                plt.plot(l_cmb, nl_all_th[f"{sv_a}&{ar_a}", f"{sv_b}&{ar_b}", spec] / (bl_dict[sv_a, ar_a] * bl_dict[sv_b, ar_b]))
                plt.xlabel("$\ell$", fontsize=14)
                plt.ylabel("$DN^{%s}_\ell$" % spec, fontsize=16)
                
        plt.suptitle(spec_name)
        plt.savefig(f"{result_dir}/spectra_{spec_name}_{spec_type}.png", bbox_inches="tight")
        plt.clf()
        plt.close()
