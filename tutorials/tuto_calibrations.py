"""
This script generates simulations for 2 surveys ("sv1", "sv2"),
introducing a miscalibration between the two surveys. Then, we compute
the calibration amplitudes using pspipe_utils.consistency for each sim
using different power spectra combinations.
"""
from pspy import pspy_utils, so_map, so_spectra, so_window, so_mcm, so_cov, sph_tools
from pspipe_utils import simulation, consistency, best_fits
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib as mpl
from pixell import curvedsky
import numpy as np
import pickle
import time
import sys

# This class is used to prevent cobaya
# from printing details about MCMC steps
# for each of the sims
from io import StringIO
class NullIO(StringIO):
    def write(self, txt):
        pass

# Output dir
output_dir = "result_calibs"
pspy_utils.create_directory(output_dir)


###############################
# Simulation input parameters #
###############################
ncomp = 3
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]
type = "Dl"
niter = 0

surveys = ["sv1", "sv2"]
arrays = {"sv1": ["ar1"],
          "sv2": ["ar1"]}
n_splits = {"sv1": 2,
            "sv2": 2}

ra0, ra1, dec0, dec1, res = -30, 30, -10, 10, 3
apo_type_survey = "Rectangle"
template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
binary = so_map.car_template(1, ra0, ra1, dec0, dec1, res)

binary.data[:] = 0
binary.data[1:-1, 1:-1] = 1

lmax = 1400
lmax_sim = lmax + 1000
bin_size = 40

binning_file = f"{output_dir}/binning.dat"
pspy_utils.create_binning_file(bin_size = bin_size,
                               n_bins = 300, file_name = binning_file)

cosmo_params = {"cosmomc_theta":0.0104085, "logA": 3.044, "ombh2": 0.02237,
                "omch2": 0.1200,  "ns": 0.9649, "tau": 0.0544}

fg_components = {"tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
                 "te": ["radio", "dust"],
                 "ee": ["radio", "dust"],
                 "bb": ["radio", "dust"],
                 "tb": ["radio", "dust"],
                 "eb": []}

fg_params = {"a_tSZ": 3.30, "a_kSZ": 1.60,"a_p": 6.90, "beta_p": 2.08, "a_c": 4.90,
             "beta_c": 2.20, "a_s": 3.10, "xi": 0.1,  "T_d": 9.60,  "a_gtt": 14.0,
             "a_gte": 0.7, "a_pste": 0,
             "a_gee": 0.27, "a_psee": 0.05,
             "a_gbb": 0.13, "a_psbb": 0.05,
             "a_gtb": 0.36, "a_pstb": 0}

rms_range = {"sv1": (1, 3),
             "sv2": (25, 35)}
fwhm_range = {"sv1": (0.5, 2),
              "sv2": (5, 10)}
cal_range = {"sv1": (0.95, 1.05),
             "sv2": (0.95, 1.05)}

freq_list = []
nu_eff, rms_uKarcmin_T, bl = {}, {}, {}
cal_dict = {}

for sv in surveys:
    rms_min, rms_max = rms_range[sv]
    fwhm_min, fwhm_max = fwhm_range[sv]
    cal_min, cal_max = cal_range[sv]
    for ar in arrays[sv]:
        if sv == "sv1":
            nu_eff[sv, ar] = np.random.randint(80, high=240)
            cal_dict[sv, ar] = np.random.uniform(cal_min, high = cal_max)
        else:
            nu_eff[sv, ar] = nu_eff["sv1", ar]
            cal_dict[sv, ar] = 1.0

        rms_uKarcmin_T[sv, f"{ar}x{ar}"] = np.random.uniform(rms_min, high=rms_max)
        freq_list += [nu_eff[sv, ar]]
        fwhm = np.random.uniform(fwhm_min, high=fwhm_max)
        l_beam, bl[sv, ar] = pspy_utils.beam_from_fwhm(fwhm, lmax_sim)

freq_list = list(dict.fromkeys(freq_list))

for sv in surveys:
    for ar1, ar2 in combinations(arrays[sv], 2):
        r = np.random.uniform(0, high = 0.7)
        rms_uKarcmin_T[sv, f"{ar1}x{ar2}"] = r * np.sqrt(rms_uKarcmin_T[sv, f"{ar1}x{ar1}"] * rms_uKarcmin_T[sv, f"{ar2}x{ar2}"])
        rms_uKarcmin_T[sv, f"{ar2}x{ar1}"] = rms_uKarcmin_T[sv, f"{ar1}x{ar2}"]
###############################
###############################


###########################################
# Prepare the data used in the simulation #
###########################################

# CMB power spectra
l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax_sim)
ps_file_name = f"{output_dir}/cmb.dat"
so_spectra.write_ps(ps_file_name, l_th, ps_dict, type, spectra = spectra)

# Beams
for sv in surveys:
    for ar in arrays[sv]:
        beam_file_name = f"{output_dir}/beam_{sv}_{ar}.dat"
        np.savetxt(beam_file_name, np.transpose([l_beam, bl[sv, ar]]))

# Noise power spectra
for sv in surveys:
    for ar1 in arrays[sv]:
        for ar2 in arrays[sv]:
            l_th, nl_th = pspy_utils.get_nlth_dict(rms_uKarcmin_T[sv, f"{ar1}x{ar2}"],
                                                   type,
                                                   lmax_sim,
                                                   spectra=spectra)
            noise_ps_file_name = f"{output_dir}/mean_{ar1}x{ar2}_{sv}_noise.dat"
            so_spectra.write_ps(noise_ps_file_name, l_th, nl_th, type, spectra = spectra)

# Foreground power spectra
fg_dict = best_fits.get_foreground_dict(l_th, freq_list, fg_components, fg_params, fg_norm = None)
fg = {}
for f1 in freq_list:
    for f2 in freq_list:
        fg[f1, f2] = {}
        for spec in spectra:
            fg[f1, f2][spec] = fg_dict[spec.lower(), "all", f1, f2]
        so_spectra.write_ps(f"{output_dir}/fg_{f1}x{f2}.dat", l_th,
                            fg[f1, f2], type, spectra = spectra)

# Window functions
window = {}
for sv in surveys:
    for ar in arrays[sv]:
        window[sv, ar] = so_window.create_apodization(binary, apo_type=apo_type_survey,
                                                      apo_radius_degree = 1)
        window[sv, ar].plot(file_name = f"{output_dir}/window_{sv}_{ar}")

# Mode coupling matrices
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
###########################################
###########################################

############################
# Generate the simulations #
############################
f_name_cmb = output_dir + "/cmb.dat"
f_name_noise = output_dir + "/mean_{}x{}_{}_noise.dat"
f_name_fg = output_dir + "/fg_{}x{}.dat"

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

print("==============")
print("= SIMULATION =")
print("==============\n")

n_sims = 100
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

                # Miscalibrate the maps
                sim_alm[sv, ar, k] /= cal_dict[sv, ar]

    for id_sv1, sv1 in enumerate(surveys):
        for id_ar1, ar1 in enumerate(arrays[sv1]):
            for id_sv2, sv2 in enumerate(surveys):
                for id_ar2, ar2 in enumerate(arrays[sv2]):

                    if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                    if  (id_sv1 > id_sv2) : continue

                    ps_dict = {}
                    for spec in spectra:
                        ps_dict[spec] = []

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
                                if (s1 == s2) & (sv1 == sv2): continue #discard the auto
                                else: ps_dict[spec] += [ps[spec]]

                    ps_dict_cross_mean = {}
                    for spec in spectra:
                        ps_dict_cross_mean[spec] = np.mean(ps_dict[spec], axis=0)
                        if ar1 == ar2 and sv1 == sv2:
                            # Average TE / ET so that for same array same season TE = ET
                            ps_dict_cross_mean[spec] = (np.mean(ps_dict[spec], axis=0) + np.mean(ps_dict[spec[::-1]], axis=0)) / 2.
                    spec_name = f"{sv1}&{ar1}x{sv2}&{ar2}"
                    if iii == 0:
                        ps_all[spec_name] = []

                    ps_all[spec_name] += [ps_dict_cross_mean]

    print("sim %05d/%05d took %.02f s to compute" % (iii, n_sims, time.time() - t))
############################
############################

##############################
# Calibration of the spectra #
##############################
ps_order = [("sv1&ar1", "sv1&ar1"),
            ("sv1&ar1", "sv2&ar1"),
            ("sv2&ar1", "sv2&ar1")]

lmin_cal = 800
lmax_cal = 1300

print("\n===============")
print("= CALIBRATION =")
print("===============\n")

# Get MC covmat
cov_dict = {}
n_bins = len(lb)
for i, ps1 in enumerate(spec_name_list):
    for j, ps2 in enumerate(spec_name_list):
        if j < i: continue

        na, nb = ps1.split("x")
        nc, nd = ps2.split("x")

        mean_a, _, mc_cov = so_cov.mc_cov_from_spectra_list(ps_all[ps1],
                                                            ps_all[ps2],
                                                            spectra=spectra)
        mc_cov_select = so_cov.selectblock(mc_cov, spectra, n_bins, block="TTTT")

        cov_dict[(na, nb), (nc, nd)] = mc_cov_select

# Get analytical covmat
l_cmb, cmb_dict = best_fits.cmb_dict_from_file(f_name_cmb, lmax, spectra)
l_fg, fg_dict = best_fits.fg_dict_from_files(f_name_fg, freq_list, lmax, spectra)
l_noise, nl_dict = best_fits.noise_dict_from_files(f_name_noise,  surveys, arrays, lmax, spectra, n_splits = n_splits)
f_name_beam = output_dir + "/beam_{}_{}.dat"
l_beam, bl_dict = best_fits.beam_dict_from_files(f_name_beam, surveys, arrays, lmax)


l_cmb, ps_all_th, nl_all_th = best_fits.get_all_best_fit(spec_name_list,
                                                         l_cmb,
                                                         cmb_dict,
                                                         fg_dict,
                                                         nu_eff,
                                                         spectra,
                                                         nl_dict=nl_dict,
                                                         bl_dict=bl_dict)

an_cov_dict = {}
for i, ps1 in enumerate(spec_name_list):
    for j, ps2 in enumerate(spec_name_list):
        if j < i: continue

        na, nb = ps1.split("x")
        nc, nd = ps2.split("x")

        sv_a, ar_a = na.split("&")
        sv_b, ar_b = nb.split("&")
        sv_c, ar_c = nc.split("&")
        sv_d, ar_d = nd.split("&")

        win = {}
        win["Ta"] = window[sv_a, ar_a]
        win["Pa"] = window[sv_a, ar_a]

        win["Tb"] = window[sv_b, ar_b]
        win["Pb"] = window[sv_b, ar_b]

        win["Tc"] = window[sv_c, ar_c]
        win["Pc"] = window[sv_c, ar_c]

        win["Td"] = window[sv_d, ar_d]
        win["Pd"] = window[sv_d, ar_d]

        coupling = so_cov.cov_coupling_spin0and2_simple(win, lmax, niter = niter)

        mbb_inv_ab = mbb_inv_dict[sv_a, ar_a, sv_b, ar_b]
        mbb_inv_cd = mbb_inv_dict[sv_c, ar_c, sv_d, ar_d]

        analytic_cov = so_cov.generalized_cov_spin0and2(coupling,
                                                        [na, nb, nc, nd],
                                                        n_splits,
                                                        ps_all_th,
                                                        nl_all_th,
                                                        lmax,
                                                        binning_file,
                                                        mbb_inv_ab,
                                                        mbb_inv_cd,
                                                        binned_mcm=True)
        analytic_cov_select = so_cov.selectblock(analytic_cov, modes,
                                                 n_bins, block="TTTT")
        an_cov_dict[(na, nb), (nc, nd)] = analytic_cov_select

# Get calibration amplitudes
proj_dict = {
             "C1": np.array([1, -1, 0]),
             "C2": np.array([0, -1, 1]),
             "C3": np.array([1, 0, -1])
            }

cal_dict_out = {f"C{i}": [] for i in range(1, 4)}

for iii in range(n_sims):

    ps_dict = {}
    for arA, arB in ps_order:
        ps_name = f"{arA}x{arB}"
        ps_dict[arA, arB] = ps_all[ps_name][iii]["TT"]

    # Concatenate power spectra
    ps_vec, full_cov = consistency.append_spectra_and_cov(ps_dict, an_cov_dict, ps_order)

    # Calibrate the spectra using sv1xsv1 - sv1xsv2
    id = np.where((lb >= lmin_cal) & (lb <= lmax_cal))[0]

    print(f"Calibrating sim {iii} ...")
    for comb, proj_pattern in proj_dict.items():
        # Prevent cobaya from printing chains details for
        # each simulation
        sys.stdout = NullIO()
        cal_mean, cal_std = consistency.get_calibration_amplitudes(ps_vec, full_cov,
                                                                   proj_pattern, "TT",
                                                                   id, f"{output_dir}/chains/mcmc")
        sys.stdout = sys.__stdout__
        cal_dict_out[comb].append(cal_mean)


pickle.dump(cal_dict_out, open(f"{output_dir}/calibs_dict.pkl", "wb"))

nbins = 15
colors = {"C1": "tab:red", "C2": "tab:blue", "C3": "tab:green"}
plt.figure(figsize = (8, 6))
for comb in cal_dict_out:
    plt.hist(cal_dict_out[comb], density = True, bins = nbins,
             label = comb, histtype = "stepfilled",
             edgecolor = mpl.colors.to_rgba(colors[comb], alpha = 1),
             facecolor = mpl.colors.to_rgba(colors[comb], alpha = 0.3),
             linewidth = 1.65)
plt.xlabel("Calibration amplitude")
plt.yticks([])
plt.legend(frameon = False)
plt.axvline(cal_dict["sv1", "ar1"], color = "k")
plt.tight_layout()
plt.savefig(f"{output_dir}/calibration_hist.png", dpi = 300)
