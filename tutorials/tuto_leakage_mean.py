"""
This script tests our analytical prediction for the effect of the leakage on all power spectra
We generate a bunch of alms and apply a random realisation of the leakage model
We then check that the recovered power spectra agree with our expectation
"""
from pspy import pspy_utils, so_spectra, so_cov
from pspipe_utils import simulation, leakage
from pixell import curvedsky
import numpy as np
import pylab as plt
import time

cosmo_params = {"cosmomc_theta":0.0104085, "logA": 3.044, "ombh2": 0.02237,
                "omch2": 0.1200,  "ns": 0.9649, "tau": 0.0544}
lmax = 4000
n_sims = 300
type = "Dl"
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

tuto_data_dir = "tuto_leakage"
pspy_utils.create_directory(tuto_data_dir)
binning_file = "../pspipe_utils/data/binning_files/BIN_ACTPOL_50_4_SC_large_bin_at_low_ell"

l_th, ps_th_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax)
f_name_cmb = f"{tuto_data_dir}/cmb.dat"
so_spectra.write_ps(f_name_cmb, l_th, ps_th_dict, type, spectra=spectra)
ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)

leakage_file_dir = "../pspipe_utils/data/beams/"
arrays = ["pa5_f090", "pa5_f150", "pa6_f090", "pa6_f150"]

gamma, var, err_m = {}, {}, {}

plt.figure(figsize=(12,8))
for ar in arrays:
    gamma[ar], var[ar], err_m[ar] = {}, {}, {}
    l, gamma[ar]["TE"], err_m[ar]["TE"], gamma[ar]["TB"], err_m[ar]["TB"] = leakage.read_leakage_model(leakage_file_dir,
                                                                                                       f"gamma_mp_uranus_{ar}.txt",
                                                                                                       lmax)
                                                                                             
    var[ar]["TETE"] = leakage.error_modes_to_cov(err_m[ar]["TE"]).diagonal()
    var[ar]["TBTB"] = leakage.error_modes_to_cov(err_m[ar]["TB"]).diagonal()
    var[ar]["TETB"] = var[ar]["TETE"] * 0

    plt.subplot(2,1,1)
    plt.errorbar(l, gamma[ar]["TE"], np.sqrt(var[ar]["TETE"]), fmt=".", label=ar)
    plt.ylabel(r"$\gamma^{TE}_{\ell}$", fontsize=17)
    plt.legend()
    plt.subplot(2,1,2)
    plt.ylabel(r"$\gamma^{TB}_{\ell}$", fontsize=17)
    plt.xlabel(r"$\ell$", fontsize=17)
    plt.errorbar(l, gamma[ar]["TB"], np.sqrt(var[ar]["TBTB"]), fmt=".", label=ar)
    plt.legend()
plt.savefig(f"{tuto_data_dir}/beam_leakage.png", bbox_inches="tight")
plt.clf()
plt.close()
    
ps_all = {}
for iii in range(n_sims):
    t = time.time()
    alms_cmb = curvedsky.rand_alm(ps_mat, lmax=lmax, dtype="complex64")
    
    alms_leak = {}
    for ar in arrays:
        gamma_TE_sim = leakage.leakage_beam_sim(gamma[ar]["TE"], err_m[ar]["TE"])
        gamma_TB_sim = leakage.leakage_beam_sim(gamma[ar]["TB"], err_m[ar]["TB"])

        alms_leak[ar] = leakage.apply_leakage_model_to_alm(alms_cmb.copy(),
                                                           gamma_TE_sim,
                                                           gamma_TB_sim)
        
    for id_ar1, ar1 in enumerate(arrays):
        for id_ar2, ar2 in enumerate(arrays):
            if  (id_ar1 > id_ar2) : continue
            if iii == 0: ps_all[ar1, ar2] = []

            l, ps = so_spectra.get_spectra_pixell(alms_leak[ar1],
                                                  alms_leak[ar2],
                                                  spectra=spectra)

            lb, psb = so_spectra.bin_spectra(l,
                                             ps,
                                             binning_file,
                                             lmax,
                                             "Dl",
                                             spectra=spectra)
        
            ps_all[ar1, ar2] += [psb]
    print(f"time to do sim {iii}", time.time() - t)

# the :2 is because the alms start at l=0, therefore we had to use gamma starting at 0
# when applied to spectra we want it to start at 2 since ps[0]=ps[l=2]
for ar in arrays:
    gamma[ar]["TE"], gamma[ar]["TB"] = gamma[ar]["TE"][2:], gamma[ar]["TB"][2:]
    var[ar]["TETE"] = var[ar]["TETE"][2:]
    var[ar]["TBTB"] = var[ar]["TBTB"][2:]
    var[ar]["TETB"] = var[ar]["TETB"][2:]

n_bins = len(lb)
for id_ar1, ar1 in enumerate(arrays):
    for id_ar2, ar2 in enumerate(arrays):
        if  (id_ar1 > id_ar2) : continue
    
        mean, _, mc_cov = so_cov.mc_cov_from_spectra_list(ps_all[ar1, ar2], ps_all[ar1, ar2], spectra=spectra)

        l, psb_th_dict = leakage.leakage_correction(l_th,
                                                    ps_th_dict,
                                                    gamma[ar1],
                                                    var[ar1],
                                                    lmax,
                                                    gamma_beta=gamma[ar2],
                                                    binning_file=binning_file)
    
        for count, spec in enumerate(spectra):
            sigma = so_cov.get_sigma(mc_cov, spectra, n_bins, spec)
        
            plt.figure(figsize=(15,10))
            plt.suptitle(f"{ar1}x{ar2}", fontsize=18)
            plt.subplot(2,1,1)
            plt.ylabel(r"$D^{%s}_{\ell}$" % spec, fontsize=18)
            plt.xlabel(r"$\ell$", fontsize=18)
            plt.errorbar(l, mean[spec], sigma, fmt=".", label="mean spectra with leakage", color="black")
            plt.plot(l, psb_th_dict[spec], label="theory prediction", color="black")
            plt.legend()
            plt.subplot(2,1,2)
            plt.errorbar(l, mean[spec] - psb_th_dict[spec], sigma, fmt=".")
            plt.ylabel(r"$D^{%s}_{\ell}- D^{%s, th}_{\ell}$" % (spec, spec), fontsize=18)
            plt.xlabel(r"$\ell$", fontsize=18)
            plt.savefig(f"{tuto_data_dir}/spectra_{ar1}x{ar2}_{spec}.png", bbox_inches="tight")
            plt.clf()
            plt.close()
