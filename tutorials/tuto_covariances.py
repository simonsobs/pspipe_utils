"""
This script generate covariance matrices, combine them in a x_ar covariance matrix
and do a ML combination from the x_ar to x_freq and from x_freq to final specta.
We then compare the analytic covariance to montecarlo ones
"""
import numpy as np
import pylab as plt
from pspipe_utils import simulation, best_fits, covariance, pspipe_list
from pspy import pspy_utils, so_spectra, sph_tools, so_map, so_cov, so_window, so_mcm, so_dict
from itertools import combinations, product
from itertools import combinations_with_replacement as cwr
from pixell import curvedsky
import time, sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

tuto_data_dir = "tuto_data"
sim_dir = "result_simulation"

result_dir = "result_covariances"
pspy_utils.create_directory(result_dir)

ncomp, niter = 3, 0
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

type = "Dl"
binned_mcm = d["binned_mcm"]
surveys = d["surveys"]
n_splits = d["n_splits"]
lmax = d["lmax"]
n_sims = d["n_sims"]
binning_file = f"{tuto_data_dir}/binning.dat"

_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

spec_name_list, nu_tag_list = pspipe_list.get_spec_name_list(d, return_nu_tag=True)
spec_name_list_ET = pspipe_list.get_spec_name_list(d, remove_same_ar_and_sv=True)

narrays, sv_list, ar_list = pspipe_list.get_arrays_list(d)
arrays_list = [f"{sv}_{ar}" for (sv, ar) in zip(sv_list, ar_list)]
freq_list = pspipe_list.get_freq_list(d)

#arrays = {}
#for sv in surveys:
#    arrays[sv] = d[f"arrays_{sv}"]
#    for ar in arrays[sv]:
#        nu_eff[sv, ar] = d[f"nu_eff_{sv}_{ar}"]
arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}

f_name_cmb = tuto_data_dir + "/cmb.dat"
f_name_noise = tuto_data_dir + "/mean_{}x{}_{}_noise.dat"
f_name_fg = tuto_data_dir + "/fg_{}x{}.dat"
f_name_beam = tuto_data_dir + "/beam_{}_{}.dat"

l_cmb, cmb_dict = best_fits.cmb_dict_from_file(f_name_cmb, lmax, spectra)
l_fg, fg_dict = best_fits.fg_dict_from_files(f_name_fg, arrays_list, lmax, spectra)
l_noise, nl_dict = best_fits.noise_dict_from_files(f_name_noise,  surveys, arrays, lmax, spectra, n_splits=n_splits)
l_beam, bl_dict = best_fits.beam_dict_from_files(f_name_beam, surveys, arrays, lmax)

l_cmb, ps_all_th, nl_all_th = best_fits.get_all_best_fit(spec_name_list,
                                                         l_cmb,
                                                         cmb_dict,
                                                         fg_dict,
                                                         spectra,
                                                         nl_dict=nl_dict,
                                                         bl_dict=bl_dict)

for sid1, spec1 in enumerate(spec_name_list):
    for sid2, spec2 in enumerate(spec_name_list):
        if sid1 > sid2: continue
        na, nb = spec1.split("x")
        nc, nd = spec2.split("x")

        print(f"cov element ({na} x {nb}, {nc} x {nd})")
        coupling = so_cov.fast_cov_coupling_spin0and2(tuto_data_dir,
                                                      [na, nb, nc, nd],
                                                      lmax)

        sv_a, ar_a = na.split("&")
        sv_b, ar_b = nb.split("&")
        sv_c, ar_c = nc.split("&")
        sv_d, ar_d = nd.split("&")

        # These objects are symmetric in (sv_a, ar_a) and (sv_b, ar_b)
        try: mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(prefix=f"{tuto_data_dir}/{na}x{nb}", spin_pairs=spin_pairs)
        except: mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(prefix=f"{tuto_data_dir}/{nb}x{na}", spin_pairs=spin_pairs)

        try: mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(prefix=f"{tuto_data_dir}/{nc}x{nd}", spin_pairs=spin_pairs)
        except: mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(prefix=f"{tuto_data_dir}/{nd}x{nc}", spin_pairs=spin_pairs)

        analytic_cov = so_cov.generalized_cov_spin0and2(coupling,
                                                       [na, nb, nc, nd],
                                                       n_splits,
                                                       ps_all_th,
                                                       nl_all_th,
                                                       lmax,
                                                       binning_file,
                                                       mbb_inv_ab,
                                                       mbb_inv_cd,
                                                       binned_mcm=binned_mcm)

        np.save(f"{result_dir}/analytic_cov_{spec1}_{spec2}.npy", analytic_cov)

cov = {}
combin_level = ["xar", "xfreq", "final"]
cov["xar"] = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                          result_dir,
                                                          "analytic_cov",
                                                          spectra_order=["TT", "TE", "ET", "EE"],
                                                          remove_doublon=True,
                                                          check_pos_def=True)

inv_cov_xar = np.linalg.inv(cov["xar"])

x_ar_list = pspipe_list.x_ar_cov_order(spec_name_list)
#x_freq_cov_order = pspipe_list.x_freq_cov_order(freq_list)

P_mat = covariance.get_x_ar_to_x_freq_P_mat(freq_list, spec_name_list, nu_tag_list, binning_file, lmax)
P_mat_cross = covariance.get_x_ar_to_x_freq_P_mat_cross(freq_list, spec_name_list, nu_tag_list, binning_file, lmax)

P = covariance.combine_P_mat(P_mat, P_mat_cross)

cov["xfreq"] = covariance.get_max_likelihood_cov(P, inv_cov_xar, force_sim = True, check_pos_def = True)
inv_cov_xfreq = np.linalg.inv(cov["xfreq"])

P_final = covariance.get_x_freq_to_final_P_mat(freq_list, binning_file, lmax)
cov["final"]  = covariance.get_max_likelihood_cov(P_final, inv_cov_xfreq, force_sim = True, check_pos_def = True)

for comb in combin_level:
    corr = so_cov.cov2corr(cov[comb], remove_diag=True)
    #so_cov.plot_cov_matrix(corr, file_name=f"{result_dir}/sigurd_plot_corr_{comb}")

#so_cov.plot_cov_matrix(P, file_name=f"{result_dir}/sigurd_plot_Pmat")
#so_cov.plot_cov_matrix(P_final, file_name=f"{result_dir}/sigurd_plot_Pmat_final")


vec_list = {}
for comb in combin_level:
    vec_list[comb] = []

for iii in range(n_sims):

    vec_xar = covariance.read_x_ar_spectra_vec(sim_dir,
                                               spec_name_list,
                                               "cross_%05d" % iii,
                                                spectra_order = ["TT", "TE", "ET", "EE"],
                                                type="Dl")

    vec_xfreq = covariance.max_likelihood_spectra(cov["xfreq"], inv_cov_xar, P, vec_xar)
    vec_final = covariance.max_likelihood_spectra(cov["final"], inv_cov_xfreq, P_final, vec_xfreq)
    vec_list["xar"] += [vec_xar]
    vec_list["xfreq"] += [vec_xfreq]
    vec_list["final"] += [vec_final]

mean = {}
for comb in combin_level:
    mean[comb] = np.mean(vec_list[comb], axis=0)
    mc_cov = np.cov(vec_list[comb], rowvar=False)

    plt.figure(figsize=(12,8))
    plt.semilogy()
    plt.plot(mc_cov.diagonal(), label="MC")
    plt.plot(cov[comb].diagonal(), label="analytic")
    plt.savefig(f"{result_dir}/plot_{comb}_covariance_diag.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    plt.figure(figsize=(12,8))
    plt.plot(mc_cov.diagonal()/cov[comb].diagonal(), label="MC/analytic")
    plt.legend()
    plt.savefig(f"{result_dir}/plot_{comb}_covariance_ratio.png", bbox_inches="tight")
    plt.clf()
    plt.close()

spectra_order = ["TT", "TE", "ET", "EE"]
xar_block_order = {}
xar_block_order["TT"] = spec_name_list
xar_block_order["TE"] = spec_name_list
xar_block_order["ET"] = spec_name_list_ET
xar_block_order["EE"] = spec_name_list
lb, spec_dict_xar, std_dict_xar = covariance.from_vector_and_cov_to_ps_and_std_dict(mean["xar"],
                                                                                    cov["xar"],
                                                                                    spectra_order,
                                                                                    xar_block_order,
                                                                                    binning_file,
                                                                                    lmax)
spectra_order = ["TT", "TE", "EE"]
xfreq_block_order = {}
xfreq_block_order["TT"] = [f"{f0}x{f1}" for f0, f1 in cwr(freq_list, 2)]
xfreq_block_order["TE"] = [f"{f0}x{f1}" for f0, f1 in product(freq_list, freq_list)]
xfreq_block_order["EE"] = [f"{f0}x{f1}" for f0, f1 in cwr(freq_list, 2)]
lb, spec_dict_xfreq, std_dict_xfreq = covariance.from_vector_and_cov_to_ps_and_std_dict(mean["xfreq"],
                                                                                        cov["xfreq"],
                                                                                        spectra_order,
                                                                                        xfreq_block_order,
                                                                                        binning_file,
                                                                                        lmax)
spectra_order = ["TT", "TE", "EE"]
final_block_order = {}
final_block_order["TT"] = [f"{f0}x{f1}" for f0, f1 in cwr(freq_list, 2)]
final_block_order["TE"] = ["final"]
final_block_order["EE"] = ["final"]
lb, spec_dict_final, std_dict_final = covariance.from_vector_and_cov_to_ps_and_std_dict(mean["final"],
                                                                                        cov["final"],
                                                                                        spectra_order,
                                                                                        final_block_order,
                                                                                        binning_file,
                                                                                        lmax)

for spec in ["TT", "TE", "EE"]:
    plt.figure(figsize=(12,8))
    for xar in xar_block_order[spec]:
        na, nb = xar.split("x")
        sv_a, ar_a = na.split("&")
        sv_b, ar_b = nb.split("&")

        plt.errorbar(lb, spec_dict_xar[spec, xar], std_dict_xar[spec, xar], fmt=".", label=f"{spec} {xar}")

    if spec == "TE":
        for xar in xar_block_order["ET"]:
            na, nb = xar.split("x")
            sv_a, ar_a = na.split("&")
            sv_b, ar_b = nb.split("&")
            plt.errorbar(lb, spec_dict_xar["ET", xar], std_dict_xar["ET", xar], fmt=".", label=f"{spec} {xar}")

    for xf in xfreq_block_order[spec]:
        plt.errorbar(lb, spec_dict_xfreq[spec, xf], std_dict_xfreq[spec, xf], fmt=".", label=f"{spec} {xf}")

    if spec == "TE" or spec == "EE":
        plt.errorbar(lb, spec_dict_final[spec, "final"], std_dict_final[spec, "final"], fmt=".", label=f"{spec} final")

    plt.xlabel(r"$\ell$",fontsize=22)
    plt.ylabel(r"$D_\ell$",fontsize=22)
    plt.legend()
    plt.savefig(f"{result_dir}/plot_spectra_{spec}.png", bbox_inches="tight")
    plt.clf()
    plt.close()

for spec in ["TT", "TE", "EE"]:
    plt.figure(figsize=(12,8))
    plt.semilogy()
    for xar in xar_block_order[spec]:
        na, nb = xar.split("x")
        sv_a, ar_a = na.split("&")
        sv_b, ar_b = nb.split("&")

        plt.errorbar(lb, std_dict_xar[spec, xar], fmt="-", label=f"{spec} {xar}")

    if spec == "TE":
        for xar in xar_block_order["ET"]:
            na, nb = xar.split("x")
            sv_a, ar_a = na.split("&")
            sv_b, ar_b = nb.split("&")

            plt.errorbar(lb, std_dict_xar["ET", xar], fmt="-", label=f"{spec} {xar}")

    for xf in xfreq_block_order[spec]:
        plt.errorbar(lb, std_dict_xfreq[spec, xf], fmt="-", label=f"{spec} {xf}")

    if spec == "TE" or spec == "EE":
        plt.errorbar(lb, std_dict_final[spec, "final"], fmt="-", label=f"{spec} final")

    plt.xlabel(r"$\ell$",fontsize=22)
    plt.ylabel(r"$\sigma_\ell$",fontsize=22)
    plt.legend()
    plt.savefig(f"{result_dir}/plot_std_{spec}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
