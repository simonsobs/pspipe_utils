"""
This simulate the error propagation of the beam leakage measurement to spectra measurement 
"""
from pspy import pspy_utils, so_cov
from pspipe_utils import leakage
import numpy as np
import pylab as plt

cosmo_params = {"cosmomc_theta":0.0104085, "logA": 3.044, "ombh2": 0.02237,
                "omch2": 0.1200,  "ns": 0.9649, "tau": 0.0544}
lmax = 4000
n_sims = 10000
type = "Dl"
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

tuto_data_dir = "tuto_leakage_cov"
pspy_utils.create_directory(tuto_data_dir)
binning_file = "../pspipe_utils/data/binning_files/BIN_ACTPOL_50_4_SC_large_bin_at_low_ell"

l_th, ps_th_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax)

leakage_file_dir = "../pspipe_utils/data/beams/"

arrays = ["pa5_f090", "pa5_f150"]

gamma, err_m, var = {}, {}, {}
plt.figure(figsize=(12,8))
for ar in arrays:
    gamma[ar], err_m[ar], var[ar] = {}, {}, {}

    l, gamma[ar]["TE"], err_m[ar]["TE"], gamma[ar]["TB"], err_m[ar]["TB"] = leakage.read_leakage_model(leakage_file_dir,
                                                                                                       f"gamma_mp_uranus_{ar}.txt",
                                                                                                       lmax,
                                                                                                       lmin=2)

    cov_TE = leakage.error_modes_to_cov(err_m[ar]["TE"])
    corr_TE = so_cov.cov2corr(cov_TE)

    cov_TB = leakage.error_modes_to_cov(err_m[ar]["TB"])
    corr_TB = so_cov.cov2corr(cov_TB)

    var[ar]["TETE"] = cov_TE.diagonal()
    var[ar]["TBTB"] = cov_TB.diagonal()
    var[ar]["TETB"] = var[ar]["TETE"] * 0

    plt.figure(figsize=(12, 8))
    plt.imshow(corr_TE)
    plt.title(f"gamma TE correlation {ar}", fontsize=12)
    plt.colorbar()
    plt.savefig(f"{tuto_data_dir}/gammaTE_correlation_{ar}.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.imshow(corr_TB)
    plt.title(f"gamma TB correlation {ar}", fontsize=12)
    plt.colorbar()
    plt.savefig(f"{tuto_data_dir}/gammaTB_correlation_{ar}.png", bbox_inches="tight")
    plt.clf()
    plt.close()

residual = {}
spec_name_list = []
for id_ar1, ar1 in enumerate(arrays):
    for id_ar2, ar2 in enumerate(arrays):
        if  (id_ar1 > id_ar2) : continue
        
        l, residual[ar1, ar2] = leakage.leakage_correction(l_th,
                                                           ps_th_dict,
                                                           gamma[ar1],
                                                           var[ar1],
                                                           lmax,
                                                           return_residual=True,
                                                           gamma_beta=gamma[ar2],
                                                           binning_file=binning_file)
        spec_name_list += [f"{ar1}x{ar2}"]

ps_all = {}
ps_all_corrected = {}
for iii in range(n_sims):
    print(iii)
    gamma_sim = {}
    for ar in arrays:
        gamma_sim[ar] = {}
        gamma_sim[ar]["TE"] = leakage.leakage_beam_sim(gamma[ar]["TE"], err_m[ar]["TE"])
        gamma_sim[ar]["TB"] = leakage.leakage_beam_sim(gamma[ar]["TB"], err_m[ar]["TB"])

    for id_ar1, ar1 in enumerate(arrays):
        for id_ar2, ar2 in enumerate(arrays):
            if  (id_ar1 > id_ar2) : continue
            
            if iii == 0:
                ps_all[ar1, ar2] = []
                ps_all_corrected[ar1, ar2] = []
                

            l, psb_sim = leakage.leakage_correction(l_th,
                                                    ps_th_dict,
                                                    gamma_sim[ar1],
                                                    var[ar1],
                                                    lmax,
                                                    gamma_beta=gamma_sim[ar2],
                                                    binning_file=binning_file)
                                                    
            ps_all[ar1, ar2] += [psb_sim]
            
            psb_sim_corrected = {}
            
            for spec in spectra:
                psb_sim_corrected[spec] = psb_sim[spec] - residual[ar1, ar2][spec]
                
            ps_all_corrected[ar1, ar2] += [psb_sim_corrected]

nbins = len(l)
for sid1, spec1 in enumerate(spec_name_list):
    for sid2, spec2 in enumerate(spec_name_list):
        if sid1 > sid2: continue
        ar1, ar2 = spec1.split("x")
        ar3, ar4 = spec2.split("x")
    
        mean_corrected_a, _, mc_cov = so_cov.mc_cov_from_spectra_list(ps_all_corrected[ar1, ar2], ps_all_corrected[ar3, ar4], spectra=spectra)

        if sid1 == sid2:
            mean_a, _, _ = so_cov.mc_cov_from_spectra_list(ps_all[ar1, ar2], ps_all[ar3, ar4], spectra=spectra)

            for my_id, spec in enumerate(spectra):
                sub_cov =  mc_cov[my_id * nbins: (my_id + 1) * nbins, my_id * nbins: (my_id + 1) * nbins]
                std = np.sqrt(sub_cov.diagonal())
                plt.figure(figsize=(12,8))
                plt.title(f"{ar1}x{ar2} {spec}")
                plt.errorbar(l, mean_a[spec], label="pre-correction")
                plt.errorbar(l, mean_corrected_a[spec], std, fmt=".", label="corrected")
                plt.plot(l_th, ps_th_dict[spec], color="gray")
                plt.legend()
                plt.savefig(f"{tuto_data_dir}/spectrum_{ar1}x{ar2}_{spec}.png", bbox_inches="tight")
                plt.clf()
                plt.close()

            nbins = len(l)
            mc_cov = mc_cov[nbins:, nbins:] #remove TT
            mc_corr = so_cov.cov2corr(mc_cov)
            plt.figure(figsize=(12,8))
            plt.title(f"cov({ar1}x{ar2}, {ar3}x{ar4})")
            plt.imshow(mc_corr)
            plt.savefig(f"{tuto_data_dir}/mc_corr_{ar1}x{ar2}_{ar3}x{ar4}.png", bbox_inches="tight")
            plt.clf()
            plt.close()

