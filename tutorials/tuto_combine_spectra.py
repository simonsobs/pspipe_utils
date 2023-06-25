"""
This script combine covariance matrices in a x_ar covariance matrix
and do a ML combination from the x_ar to x_freq and from x_freq to final specta.
We then compare the analytic covariance to montecarlo ones
"""
import numpy as np
import pylab as plt
from pspipe_utils import covariance, pspipe_list
from pspy import pspy_utils, so_cov, so_dict
from itertools import product
from itertools import combinations_with_replacement as cwr
import time, sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

tuto_data_dir = "tuto_data"
sim_dir = "result_simulation"
cov_dir = "result_covariances"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

type = "Dl"
lmax = d["lmax"]
n_sims = d["n_sims"]
binning_file = f"{tuto_data_dir}/binning.dat"

_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

spec_name_list, nu_tag_list = pspipe_list.get_spec_name_list(d, return_nu_tag=True)

freq_list = pspipe_list.get_freq_list(d)
if d["cov_T_E_only"] == True:
    modes_for_xar_cov = ["TT", "TE", "ET", "EE"]
    modes_for_xfreq_cov = ["TT", "TE", "EE"]
else:
    modes_for_xar_cov = spectra
    modes_for_xfreq_cov = ["TT", "TE", "TB", "EE", "EB", "BB"]

cov = {}
combin_level = ["xar", "xfreq", "final"]
cov["xar"] = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                          cov_dir,
                                                          "analytic_cov",
                                                          spectra_order=modes_for_xar_cov,
                                                          remove_doublon=True,
                                                          check_pos_def=True)

inv_cov_xar = np.linalg.inv(cov["xar"])

x_ar_cov_list = pspipe_list.x_ar_cov_order(spec_name_list, nu_tag_list, spectra_order=modes_for_xar_cov)
x_freq_cov_list = pspipe_list.x_freq_cov_order(freq_list, spectra_order=modes_for_xfreq_cov)
final_cov_list = pspipe_list.final_cov_order(freq_list, spectra_order=modes_for_xfreq_cov)

print("x_array list:")
print(x_ar_cov_list)
print("x_freq list:")
print(x_freq_cov_list)
print("final_cov_list:")
print(final_cov_list)

P_mat = covariance.get_x_ar_to_x_freq_P_mat(x_ar_cov_list, x_freq_cov_list, binning_file, lmax)
cov["xfreq"] = covariance.get_max_likelihood_cov(P_mat, inv_cov_xar, force_sim = True, check_pos_def = True)
inv_cov_xfreq = np.linalg.inv(cov["xfreq"])

P_final = covariance.get_x_freq_to_final_P_mat(x_freq_cov_list, final_cov_list, binning_file, lmax)
cov["final"]  = covariance.get_max_likelihood_cov(P_final, inv_cov_xfreq, force_sim = True, check_pos_def = True)

covariance.plot_P_matrix(P_mat, x_freq_cov_list, x_ar_cov_list, file_name=f"{cov_dir}/P_mat_x_ar_to_x_freq")
covariance.plot_P_matrix(P_final, final_cov_list, x_freq_cov_list, file_name=f"{cov_dir}/P_mat_x_freq_to_final")


    
for iii in range(n_sims):
    if iii == 0:
        vec_list = {}
        for comb in combin_level:
            vec_list[comb] = []

    vec_xar = covariance.read_x_ar_spectra_vec(sim_dir,
                                               spec_name_list,
                                               "cross_%05d" % iii,
                                               spectra_order = modes_for_xar_cov,
                                               type="Dl")
                                               
    vec_xfreq = covariance.max_likelihood_spectra(cov["xfreq"], inv_cov_xar, P_mat, vec_xar)
    vec_final = covariance.max_likelihood_spectra(cov["final"], inv_cov_xfreq, P_final, vec_xfreq)
    vec_list["xar"] += [vec_xar]
    vec_list["xfreq"] += [vec_xfreq]
    vec_list["final"] += [vec_final]
    
mean = {}
for comb in combin_level:

    mean[comb] = np.mean(vec_list[comb], axis=0)
    mc_cov = np.cov(vec_list[comb], rowvar=False)

    plt.figure(figsize=(12, 8))
    plt.semilogy()
    plt.plot(mc_cov.diagonal(), label="MC")
    plt.plot(cov[comb].diagonal(), label="analytic")
    plt.legend()
    plt.savefig(f"{cov_dir}/plot_{comb}_covariance_diag.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(mc_cov.diagonal() / cov[comb].diagonal(), label="MC / analytic")
    plt.legend()
    plt.savefig(f"{cov_dir}/plot_{comb}_covariance_ratio.png", bbox_inches="tight")
    plt.clf()
    plt.close()
    
    corr = so_cov.cov2corr(cov[comb], remove_diag=True)
    so_cov.plot_cov_matrix(corr, file_name=f"{cov_dir}/corr_{comb}")

    
bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(bin_hi)

def select_spec(spec_vec, cov, id, n_bins):
    mean = spec_vec[id * n_bins: (id + 1) * n_bins]
    cov_block = cov[id * n_bins: (id + 1) * n_bins, id * n_bins: (id + 1) * n_bins]
    std = np.sqrt(cov_block.diagonal())
    return mean, std

lscaling = {}
lscaling["TT"] = 2
lscaling["TE"] = 1
lscaling["TB"] = 0
lscaling["EE"] = 1
lscaling["EB"] = 0
lscaling["BB"] = 0

for spec in modes_for_xfreq_cov:
        x_freq_list = []
        if spec[0] == spec[1]:
            x_freq_list += [(f0, f1) for f0, f1 in cwr(freq_list, 2)]
        else:
            x_freq_list +=  [(f0, f1) for f0, f1 in product(freq_list, freq_list)]
        for x_freq in x_freq_list:
            plt.figure(figsize=(12,8))
    
            for id_ar, x_ar_cov_el in enumerate(x_ar_cov_list):
                spec1, name, nu_pair1 = x_ar_cov_el
                if (spec1 == spec) and (spec1[0] == spec1[1]):
                    if (x_freq == nu_pair1) or (x_freq == nu_pair1[::-1]):
                        mean, std = select_spec(vec_xar, cov["xar"], id_ar, n_bins)
                        plt.errorbar(lb, mean * lb ** lscaling[spec], std * lb ** lscaling[spec], label=x_ar_cov_el , fmt=".")
                if (spec1 == spec) and (spec1[0] != spec1[1]) :
                    if (x_freq == nu_pair1):
                        mean, std = select_spec(vec_xar, cov["xar"], id_ar, n_bins)
                        plt.errorbar(lb, mean * lb ** lscaling[spec], std * lb ** lscaling[spec], label=x_ar_cov_el, fmt=".")
                if (spec1[0] != spec1[1]) and (spec1 == spec[::-1]):
                    if (x_freq == nu_pair1[::-1]):
                        mean, std = select_spec(vec_xar, cov["xar"], id_ar, n_bins)
                        plt.errorbar(lb, mean * lb ** lscaling[spec], std * lb ** lscaling[spec], label=x_ar_cov_el, fmt=".")

            for id_freq, x_freq_cov_el in enumerate(x_freq_cov_list):
                spec2, nu_pair2 = x_freq_cov_el
                if (spec2 == spec) and (x_freq == nu_pair2):
                    mean, std = select_spec(vec_xfreq, cov["xfreq"], id_freq, n_bins)
                    plt.errorbar(lb, mean * lb ** lscaling[spec], std * lb ** lscaling[spec], label=x_freq_cov_el)

            plt.legend()
            plt.xlabel(r"$\ell$", fontsize=12)
            plt.ylabel(r"$\ell^{%.1f} D^{%s}_\ell$" % (lscaling[spec], spec), fontsize=20)
            plt.savefig(f"{cov_dir}/{spec}_xar_and_xfreq_{x_freq[0]}x{x_freq[1]}.png", bbox_inches="tight")
            plt.clf()
            plt.close()

        if spec == "TT": continue
        plt.figure(figsize=(12,8))
        for id_freq, x_freq_cov_el in enumerate(x_freq_cov_list):
            spec2, nu_pair2 = x_freq_cov_el
            if (spec2 == spec):
                for x_freq in x_freq_list:
                    if (x_freq == nu_pair2):
                    
                        mean, std = select_spec(vec_xfreq, cov["xfreq"], id_freq, n_bins)
                        plt.errorbar(lb, mean * lb ** lscaling[spec], std * lb ** lscaling[spec], label=x_freq_cov_el)

        for id_final, final_cov_el in enumerate(final_cov_list):
            spec3, _ = final_cov_el
            if spec3 == spec:
                mean, std = select_spec(vec_final, cov["final"], id_final, n_bins)
                plt.errorbar(lb, mean * lb ** lscaling[spec], std * lb ** lscaling[spec], label=spec + " final")

        plt.legend()
        
        plt.xlabel(r"$\ell$", fontsize=12)
        plt.ylabel(r"$\ell^{%.1f} D^{%s}_\ell$" % (lscaling[spec], spec), fontsize=20)
        plt.savefig(f"{cov_dir}/{spec}_xfreq_and_final.png", bbox_inches="tight")
        plt.clf()
        plt.close()
    
