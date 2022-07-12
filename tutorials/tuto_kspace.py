"""
Some test of the kspace filter
"""

from pspy import so_map, so_window, sph_tools, so_spectra, pspy_utils, so_map_preprocessing, so_mcm, so_cov
import pylab as plt
import numpy as np
from pspipe_utils import simulation, best_fits, kspace
from pixell import curvedsky, enmap
import time

test_dir = "result_leakage"
pspy_utils.create_directory(test_dir)

########################################################
# let's first specify the parameters used in the analysis
########################################################

ncomp = 3
niter = 0
n_sims = 40
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
binning_file = "data/BIN_ACTPOL_50_4_SC_large_bin_at_low_ell"
vk_mask = [-90, 90]
hk_mask = [-50, 50]
binned_mcm = False
type = "Dl"
cosmo_params = {"cosmomc_theta":0.0104085,
                "logA": 3.044,
                "ombh2": 0.02237,
                "omch2": 0.1200,
                "ns": 0.9649,
                "tau": 0.0544}

fg_components = {"tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
                 "te": ["radio", "dust"],
                 "ee": ["radio", "dust"],
                 "bb": ["radio", "dust"],
                 "tb": ["radio", "dust"],
                 "eb": []}

fg_params = {"a_tSZ": 3.30, "a_kSZ": 1.60,"a_p": 6.90, "beta_p": 2.08, "a_c": 4.90,
             "beta_c": 2.20, "a_s": 3.10, "xi": 0.1,  "T_d": 9.60,  "a_gtt": 18.7,
             "a_gte": 0.36, "a_pste": 0.05,
             "a_gee": 0.13, "a_psee": 0.05,
             "a_gbb": 0.13, "a_psbb": 0.05,
             "a_gtb": 0.36, "a_pstb": 0.05}


survey = "dr6"
arrays = ["pa4_f150"]#, "pa4_f220", "pa5_f090", "pa5_f150", "pa6_f090", "pa6_f150"]
nu_effs = {}
nu_effs["dr6", "pa4_f150"] = 150
nu_effs["dr6", "pa4_f220"] = 220
nu_effs["dr6", "pa5_f090"] = 90
nu_effs["dr6", "pa5_f150"] = 150
nu_effs["dr6", "pa6_f090"] = 90
nu_effs["dr6", "pa6_f150"] = 150

##########################################################################################################
# let's generate the filter, the spactial window function and compute the associated mode coupling matrices
###########################################################################################################

window_tuple, mcm_inv, Bbl, filter_std, binary, bl = {}, {}, {}, {}, {}, {}
spec_name_list = []

for ar in arrays:

    binary[ar] = so_map.read_map(f"data/binary_dr6_{ar}_downgraded.fits")
    lmax = int(binary[ar].get_lmax_limit())

    template = binary[ar].copy()
    ny, nx = binary[ar].data.shape
    template.ncomp = 3
    template.data = enmap.empty((3, ny, nx), wcs=binary[ar].data.wcs)

    window_binary = binary[ar].copy()
 
    dist = so_window.get_distance(window_binary, rmax= np.pi / 180)
    window_binary.data[dist.data < 0.5] = 0

    window = so_window.create_apodization(window_binary, apo_type="C1", apo_radius_degree=1)
    mask = so_map.simulate_source_mask(window_binary, n_holes=5000, hole_radius_arcmin=10)
    mask = so_window.create_apodization(mask, apo_type="C1", apo_radius_degree=0.3)
    window.data *= mask.data

    window.plot(file_name=f"{test_dir}/window_dr6_{ar}")

    binary[ar].plot(file_name=f"{test_dir}/binary_{ar}")
    window_tuple[ar] = (window, window)
    
    l, bl[ar] = pspy_utils.read_beam_file(f"data/coadd_{ar}_night_beam_tform_jitter_cmb.txt")

    mcm_inv[ar], Bbl[ar] = so_mcm.mcm_and_bbl_spin0and2(window_tuple[ar],
                                                    bl1 = (bl[ar], bl[ar]),
                                                    binning_file = binning_file,
                                                    lmax=lmax,
                                                    type=type,
                                                    niter=niter,
                                                    binned_mcm=binned_mcm)

    filter_std[ar] = so_map_preprocessing.build_std_filter(template.data.shape,
                                                           template.data.wcs,
                                                           vk_mask,
                                                           hk_mask,
                                                           dtype=np.float64)
                                                           
    spec_name = f"{survey}&{ar}x{survey}&{ar}"
    spec_name_list += [spec_name]
    
############################################################################
# let's prepare the theory and foreground matrix used to generate simulation
############################################################################

l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax + 100)
f_name_cmb = test_dir + "/cmb.dat"

so_spectra.write_ps(f_name_cmb, l_th, ps_dict, type, spectra=spectra)
ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)

freq_list = []
for ar in arrays:
    freq_list += [nu_effs[survey, ar]]
freq_list = list(dict.fromkeys(freq_list))

fg_dict = best_fits.get_foreground_dict(l_th, freq_list, fg_components, fg_params, fg_norm=None)
fg= {}
for freq1 in freq_list:
    for freq2 in freq_list:
        fg[freq1, freq2] = {}
        for spec in spectra:
            fg[freq1,freq2][spec] = fg_dict[spec.lower(), "all", freq1, freq2]
        so_spectra.write_ps(f"{test_dir}/fg_{freq1}x{freq2}.dat", l_th, fg[freq1,freq2], type, spectra=spectra)

f_name_fg = test_dir + "/fg_{}x{}.dat"
l, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, freq_list, lmax, spectra)

#################################
# let's generate the simulations
#################################

ps_list = {}
for ar in arrays:
    ps_list[survey, ar] = {}

scenarios = ["standard", "noE", "noB"]

for iii in range(n_sims):
    t = time.time()
    
    alms_cmb = curvedsky.rand_alm(ps_mat, lmax=lmax, dtype="complex64")
    fglms = simulation.generate_fg_alms(fg_mat, freq_list, lmax)

    for scenario in scenarios:

        for ar in arrays:
        
            alms_beamed = alms_cmb.copy()
            alms_beamed += fglms[nu_effs[survey, ar]]
        
            alms_beamed = curvedsky.almxfl(alms_beamed, bl[ar])
            
            if scenario == "noE": alms_beamed[1] *= 0
            if scenario == "noB": alms_beamed[2] *= 0

            cmb = sph_tools.alm2map(alms_beamed, template)
        
            alms = sph_tools.get_alms(cmb, window_tuple[ar], niter, lmax)

            l, ps = so_spectra.get_spectra(alms,
                                           alms,
                                           spectra=spectra)
                                       
            lb, ps = so_spectra.bin_spectra(l,
                                            ps,
                                            binning_file,
                                            lmax,
                                            type=type,
                                            mbb_inv=mcm_inv[ar],
                                            binned_mcm=binned_mcm,
                                            spectra=spectra)

            cmb_cut_filter = cmb.copy()
            my_binary = binary[ar].copy()
            #my_binary = so_window.create_apodization(my_binary, apo_type="C1", apo_radius_degree=0.5)

            cmb_cut_filter = so_map.fourier_convolution(cmb_cut_filter, filter_std[ar], my_binary)

            alms_cut_filter = sph_tools.get_alms(cmb_cut_filter, window_tuple[ar], niter, lmax)
        
            l, ps_cut_filter = so_spectra.get_spectra(alms_cut_filter,
                                                      alms_cut_filter,
                                                      spectra=spectra)
                                                  
            lb, ps_cut_filter = so_spectra.bin_spectra(l,
                                                       ps_cut_filter,
                                                       binning_file,
                                                       lmax,
                                                       type=type,
                                                       mbb_inv=mcm_inv[ar],
                                                       binned_mcm=binned_mcm,
                                                       spectra=spectra)
                                                   
            if iii == 0:
                ps_list[survey, ar]["nofilter", scenario] = []
                ps_list[survey, ar]["filter", scenario] = []

            ps_list[survey, ar]["nofilter", scenario] += [ps]
            ps_list[survey, ar]["filter", scenario] += [ps_cut_filter]

    print(f"sim %03d/%03d took %.2f s to compute" % (iii, n_sims, time.time() - t))

#########################
# Let's plot the results
#########################

elements  = ["TT_to_TT", "EE_to_EE", "BB_to_BB", "EE_to_BB", "BB_to_EE"]
kspace_matrix = {}

plt.figure(figsize=(12,8))
for ar in arrays:
    kspace_dict, std, kspace_matrix[ar] = kspace.build_kspace_filter_matrix(lb,
                                                                            ps_list[survey, ar],
                                                                            n_sims,
                                                                            spectra,
                                                                            return_dict=True)
                                                                     
    for count, el in enumerate(elements):
        plt.subplot(3, 2, count+1)
        plt.ylabel(el)
        plt.xlabel(r"$\ell$")
        plt.errorbar(lb, kspace_dict[el], std[el] / np.sqrt(n_sims), label = ar)
plt.legend()
plt.savefig(f"{test_dir}/kspace_mat.png", bbox_inches="tight")
plt.clf()
plt.close()

l_cmb, cmb_dict = best_fits.cmb_dict_from_file(f_name_cmb, lmax + 2, spectra)
l_fg, fg_dict = best_fits.fg_dict_from_files(f_name_fg, freq_list, lmax + 2, spectra)
l_cmb, ps_all_th = best_fits.get_all_best_fit(spec_name_list,
                                              l_cmb,
                                              cmb_dict,
                                              fg_dict,
                                              nu_effs,
                                              spectra)

for ar in arrays:

    list_filter = ps_list[survey, ar]["filter", "standard"]
    list_filter_corrected = []
    for ps in list_filter:
        lb, ps = kspace.deconvolve_kspace_filter_matrix(lb, ps, kspace_matrix[ar], spectra)
        
        list_filter_corrected += [ps]
        
    list_no_filter = ps_list[survey, ar]["nofilter", "standard"]
    
    mean_no_filter, _, mc_cov_no_filter = so_cov.mc_cov_from_spectra_list(list_no_filter,
                                                                          list_no_filter,
                                                                          spectra=spectra)
                                                                          
    mean_filter_corrected, _, mc_cov_filter_corrected = so_cov.mc_cov_from_spectra_list(list_filter_corrected,
                                                                                        list_filter_corrected,
                                                                                        spectra=spectra)

    ps_th = {}
    for spec in spectra:
        ps_th[spec] = ps_all_th[f"{survey}&{ar}", f"{survey}&{ar}", spec]
    ps_theory_b = so_mcm.apply_Bbl(Bbl[ar], ps_th, spectra=spectra)

    for spec in spectra:
    
        cov_no_filter = so_cov.selectblock(mc_cov_no_filter,
                                           spectra,
                                           len(lb),
                                           block=spec+spec)
                                           
        cov_filter_corrected = so_cov.selectblock(mc_cov_filter_corrected,
                                                  spectra,
                                                  len(lb),
                                                  block=spec+spec)
                                                  
        std_no_filter = np.sqrt(cov_no_filter.diagonal())
        std_filter_corrected = np.sqrt(cov_filter_corrected.diagonal())
                            
        plt.figure(figsize=(12,8))
        plt.errorbar(l_cmb, ps_all_th[f"{survey}&{ar}", f"{survey}&{ar}", spec], label="CMB + fg", color="gray")
        plt.errorbar(lb, ps_theory_b[spec], color="black")
        plt.errorbar(lb - 5, mean_no_filter[spec], std_no_filter, fmt=".", label="no filter")
        plt.errorbar(lb + 5, mean_filter_corrected[spec], std_filter_corrected, fmt=".", label="filter corrected")
        plt.xlabel(r"$\ell$", fontsize=22)
        plt.ylabel(r"$D^{%s}_\ell$" % spec, fontsize=22)
        plt.legend()
        plt.savefig(f"{test_dir}/spectra_{ar}_{spec}.png", bbox_inches="tight")
        plt.clf()
        plt.close()
    
        diff_no_filter = mean_no_filter[spec] - ps_theory_b[spec]
        diff_filter_corrected = mean_filter_corrected[spec] - ps_theory_b[spec]
    
        plt.figure(figsize=(12,8))
        plt.plot(lb, lb * 0, color="black")
        plt.errorbar(lb - 5, diff_no_filter, std_no_filter/ np.sqrt(n_sims), fmt=".", label="no filter")
        plt.errorbar(lb + 5, diff_filter_corrected, std_filter_corrected / np.sqrt(n_sims), fmt=".", label="filter corrected")
        plt.xlabel(r"$\ell$", fontsize=22)
        plt.ylabel(r"$\Delta D^{%s}_\ell$" % spec, fontsize=22)
        plt.legend()
        plt.savefig(f"{test_dir}/diff_spectra_{ar}_{spec}.png", bbox_inches="tight")
        plt.clf()
        plt.close()

