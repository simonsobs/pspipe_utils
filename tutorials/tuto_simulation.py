"""
"""
import numpy as np
import pylab as plt
from pspipe_utils import simulation, best_fits
from pspy import pspy_utils, so_spectra, sph_tools, so_map, so_cov, so_window, so_mcm
from itertools import combinations
from pixell import curvedsky
import time

##########################################################################
# Let's start by specifying the general parameters of the analysis
##########################################################################

test_dir = "result_simulation"
pspy_utils.create_directory(test_dir)
ncomp = 3
niter = 0
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
type = "Dl"

surveys = ["dr6"]
arrays = {}
arrays["dr6"] = ["ar1", "ar2"]
n_splits = {}
n_splits["dr6"] = 2

ra0, ra1, dec0, dec1, res = -30, 30, -10, 10, 2
apo_type_survey = "Rectangle"
template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
binary = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
binary.data[:] = 0
binary.data[1:-1, 1:-1] = 1

lmax = 2500
lmax_sim = lmax + 500
bin_size = 40
n_sims = 10

binning_file = "%s/binning.dat" % test_dir
pspy_utils.create_binning_file(bin_size=bin_size, n_bins=300, file_name=binning_file)

cosmo_params = {"cosmomc_theta":0.0104085, "logA": 3.044, "ombh2": 0.02237,
                "omch2": 0.1200,  "ns": 0.9649, "tau": 0.0544}

fg_components = {"tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
                 "te": ["radio", "dust"],
                 "ee": ["radio", "dust"],
                 "bb": ["radio", "dust"],
                 "tb": ["radio", "dust"],
                 "eb": []}

fg_params = {"a_tSZ": 3.30, "a_kSZ": 1.60,"a_p": 6.90, "beta_p": 2.08, "a_c": 4.90,
             "beta_c": 2.20, "a_s": 3.10, "xi": 0.1,  "T_d": 9.60,  "a_gtt": 2.79,
             "a_gte": 0.36, "a_pste": 0.05,
             "a_gee": 0.13, "a_psee": 0.05,
             "a_gbb": 0.13, "a_psbb": 0.05,
             "a_gtb": 0.36, "a_pstb": 0.05}

freq_list = []
nu_eff, rms_uKarcmin_T, bl = {}, {}, {}
for sv in surveys:
    for ar in arrays[sv]:
        nu_eff[sv, ar] = np.random.randint(40, high=280)
        rms_uKarcmin_T[sv, f"{ar}x{ar}"] = np.random.uniform(1, high=3)
        freq_list += [nu_eff[sv, ar]]
        fwhm = np.random.uniform(0.5, high=2)
        l_beam, bl[sv, ar] = pspy_utils.beam_from_fwhm(fwhm, lmax_sim)

freq_list = list(dict.fromkeys(freq_list))

for sv in surveys:
    for ar1, ar2 in combinations(arrays[sv], 2):
        r = 0.7 
        rms_uKarcmin_T[sv, f"{ar1}x{ar2}"] = r * np.sqrt(rms_uKarcmin_T[sv, f"{ar1}x{ar1}"] * rms_uKarcmin_T[sv, f"{ar2}x{ar2}"])
        rms_uKarcmin_T[sv, f"{ar2}x{ar1}"] = rms_uKarcmin_T[sv, f"{ar1}x{ar2}"]


#########################################
# Generate the data file for the simulation
#########################################

# First the cmb power spectra

l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax_sim)
f_name = f"{test_dir}/cmb.dat"
so_spectra.write_ps(f_name, l_th, ps_dict, type, spectra=spectra)

# then the beams

for sv in surveys:
    for ar in arrays[sv]:
        f_name = f"{test_dir}/beam_{sv}_{ar}.dat"
        np.savetxt(f_name, np.transpose([l_beam, bl[sv, ar]]))

# then the noise power spectra

for sv in surveys:
    for ar1 in arrays[sv]:
        for ar2 in arrays[sv]:
            l_th, nl_th = pspy_utils.get_nlth_dict(rms_uKarcmin_T[sv, f"{ar1}x{ar2}"],
                                                   type,
                                                   lmax_sim,
                                                   spectra=spectra)
            #if ar1 == ar2:
                # include correlation for the cross component noise power spectrum
            for spec in ["TE", "TB", "EB"]:
                r = 0.7
                s1, s2 = spec
                nl_th[s1 + s2] = r * np.sqrt(nl_th[s1 + s1] * nl_th[s2 + s2] )
                nl_th[s2 + s1] = nl_th[s1 + s2]

            
            f_name = f"{test_dir}/mean_{ar1}x{ar2}_{sv}_noise.dat"
            so_spectra.write_ps(f_name, l_th, nl_th, type, spectra=spectra)
            
# finally the fg power spectra

fg_dict = best_fits.get_foreground_dict(l_th, freq_list, fg_components, fg_params, fg_norm=None)
fg= {}
for freq1 in freq_list:
    for freq2 in freq_list:
        fg[freq1, freq2] = {}
        for spec in spectra:
            fg[freq1,freq2][spec] = fg_dict[spec.lower(), "all", freq1, freq2]
        so_spectra.write_ps(f"{test_dir}/fg_{freq1}x{freq2}.dat", l_th, fg[freq1,freq2], type, spectra=spectra)

##########################################################################
# Let's generate the different window functions
##########################################################################


window = {}
for sv in surveys:
    for ar in arrays[sv]:
        window[sv, ar] = so_window.create_apodization(binary,
                                              apo_type=apo_type_survey,
                                              apo_radius_degree=1)
        window[sv, ar].plot(file_name=f"{test_dir}/window_{sv}_{ar}")

##########################################################################
# Let's generate all the necessary mode coupling matrices
##########################################################################

spec_name_list = []
mbb_inv_dict = {}
for id_sv_a, sv_a in enumerate(surveys):
    for id_ar_a, ar_a in enumerate(arrays[sv_a]):
        # we need both the window for T and pol, here we assume they are the same
        window_tuple_a = (window[sv_a, ar_a], window[sv_a, ar_a])
        bl_a = (bl[sv_a, ar_a], bl[sv_a, ar_a])

        for id_sv_b, sv_b in enumerate(surveys):
            for id_ar_b, ar_b in enumerate(arrays[sv_b]):
        
                if  (id_sv_a == id_sv_b) & (id_ar_a > id_ar_b) : continue
                if  (id_sv_a > id_sv_b) : continue
                # the if here help avoiding redondant computation
                # the mode coupling matrices for sv1 x sv2 are the same as the one for sv2 x sv1
                # identically, within a survey, the mode coupling matrix for ar1 x ar2 =  ar2 x ar1
                window_tuple_b = (window[sv_b, ar_b], window[sv_b, ar_b])
                bl_b = (bl[sv_b, ar_b], bl[sv_b, ar_b])

                mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window_tuple_a,
                                                            win2 = window_tuple_b,
                                                            bl1 = bl_a,
                                                            bl2 = bl_b,
                                                            binning_file = binning_file,
                                                            lmax=lmax,
                                                            type="Dl",
                                                            niter=niter)

                mbb_inv_dict[sv_a, ar_a, sv_b, ar_b] = mbb_inv
                spec_name = f"{sv_a}&{ar_a}x{sv_b}&{ar_b}"
                spec_name_list += [spec_name]
                
##########################################################################
# Let's generate the simulations
##########################################################################

f_name_cmb = test_dir + "/cmb.dat"
f_name_noise = test_dir + "/mean_{}x{}_{}_noise.dat"
f_name_fg = test_dir + "/fg_{}x{}.dat"

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

    
    for id_sv1, sv1 in enumerate(surveys):
        for id_ar1, ar1 in enumerate(arrays[sv1]):
            for id_sv2, sv2 in enumerate(surveys):
                for id_ar2, ar2 in enumerate(arrays[sv2]):
            
                    if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                    if  (id_sv1 > id_sv2) : continue
                    
                    ps_dict = {}
                    for spec in spectra:
                        ps_dict[spec, "auto"] = []
                        ps_dict[spec, "cross"] = []
                    
                    for s1 in range(n_splits[sv1]):
                        for s2 in range(n_splits[sv2]):
                            if (sv1 == sv2) & (ar1 == ar2) & (s1>s2) : continue

                            l, ps_master = so_spectra.get_spectra_pixell(sim_alm[sv1, ar1, s1],
                                                                         sim_alm[sv2, ar2, s2],
                                                                         spectra=spectra)

                            lb, ps = so_spectra.bin_spectra(l,
                                                            ps_master,
                                                            binning_file,
                                                            lmax,
                                                            type=type,
                                                            mbb_inv=mbb_inv_dict[sv1, ar1, sv2, ar2],
                                                            spectra=spectra)
                                                            
                            for count, spec in enumerate(spectra):
                                if (s1 == s2) & (sv1 == sv2): ps_dict[spec, "auto"] += [ps[spec]]
                                else: ps_dict[spec, "cross"] += [ps[spec]]

                    ps_dict_auto_mean, ps_dict_cross_mean, ps_dict_noise_mean = {}, {}, {}
                    for spec in spectra:
                        ps_dict_cross_mean[spec] = np.mean(ps_dict[spec, "cross"], axis=0)
                        if ar1 == ar2 and sv1 == sv2:
                            # Average TE / ET so that for same array same season TE = ET
                            ps_dict_cross_mean[spec] = (np.mean(ps_dict[spec, "cross"], axis=0) + np.mean(ps_dict[spec[::-1], "cross"], axis=0)) / 2.
                            
                        if sv1 == sv2:
                            ps_dict_auto_mean[spec] = np.mean(ps_dict[spec, "auto"], axis=0)
                            ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / n_splits[sv1]

                    spec_name = f"{sv1}&{ar1}x{sv2}&{ar2}"
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

f_name_beam = test_dir + "/beam_{}_{}.dat"
l_beam, bl_dict = best_fits.beam_dict_from_files(f_name_beam, surveys, arrays, lmax)


l_cmb, ps_all_th, nl_all_th = best_fits.get_all_best_fit(spec_name_list,
                                                         l_cmb,
                                                         cmb_dict,
                                                         fg_dict,
                                                         nu_eff,
                                                         spectra,
                                                         nl_dict=nl_dict)


n_bins = len(lb)
for my_spec in spec_name_list:
    na, nb = my_spec.split("x")
    sv_a, ar_a = na.split("&")
    sv_b, ar_b = nb.split("&")

    for spec_type in ["cross", "noise"]:
        mean_a, _, mc_cov = so_cov.mc_cov_from_spectra_list(ps_all[my_spec, spec_type], ps_all[my_spec, spec_type], spectra=spectra)

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
                plt.title(my_spec)
                plt.legend(fontsize=14)
            else:
                plt.plot(l_cmb, nl_all_th[f"{sv_a}&{ar_a}", f"{sv_b}&{ar_b}", spec] / (bl_dict[sv_a, ar_a] * bl_dict[sv_b, ar_b]))
                plt.xlabel("$\ell$", fontsize=14)
                plt.ylabel("$DN^{%s}_\ell$" % spec, fontsize=16)
                
        plt.suptitle(my_spec)
        plt.savefig(f"{test_dir}/spectra_{my_spec}_{spec_type}.png", bbox_inches="tight")
        plt.clf()
        plt.close()

