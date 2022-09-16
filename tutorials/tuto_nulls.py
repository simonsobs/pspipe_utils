"""
In this tutorial, we compute a set of simulations for two
different arrays ("ar1", "ar2") in order to perform null
tests between them at the power spectrum level. The goal is
to compare Monte Carlo errors on the power spectrum residuals,
with analytical errors. In this tutorial, we are using different
window functions for "ar1" & "ar2".
"""
from pspy import pspy_utils, so_map, so_spectra, so_window, so_mcm, so_cov, sph_tools
from pspipe_utils import simulation, consistency, best_fits
from itertools import combinations
import matplotlib.pyplot as plt
from pixell import curvedsky
import matplotlib as mpl
import numpy as np
import time

# Output dir
output_dir = "result_nulls"
pspy_utils.create_directory(output_dir)


###############################
# Simulation input parameters #
###############################
ncomp = 3
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]
type = "Dl"
niter = 0

surveys = ["sv1"]
arrays = {"sv1": ["ar1", "ar2"]}

n_splits = {"sv1": 2}

ra0, ra1, dec0, dec1, res = -30, 30, -10, 10, 3
apo_type_survey = "C2"
template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
binary = so_map.car_template(1, ra0, ra1, dec0, dec1, res)

binaries = {("sv1", "ar1"): binary,
            ("sv1", "ar2"): binary.copy()}

nx, ny = binary.data.shape

binaries["sv1", "ar1"].data[:] = 0
binaries["sv1", "ar1"].data[1:-1, 1:-1] = 1

binaries["sv1", "ar2"].data[:] = 0
binaries["sv1", "ar2"].data[nx//4:-nx//4, ny//4:-ny//4] = 1

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

rms_range = (1, 3)
fwhm_range = (0.5, 2)

freq_list = []
nu_eff, rms_uKarcmin_T, bl = {}, {}, {}
cal_dict = {}

for sv in surveys:
    rms_min, rms_max = rms_range
    fwhm_min, fwhm_max = fwhm_range

    for ar in arrays[sv]:
        if ar == "ar1":
            nu_eff[sv, ar] = np.random.randint(80, high=240)
        else:
            nu_eff[sv, ar] = nu_eff[sv, "ar1"]

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
        window[sv, ar] = so_window.create_apodization(binaries[sv, ar],
                                                      apo_type=apo_type_survey,
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

n_sims = 250
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


print("\n===============")
print("= COVARIANCES =")
print("===============")

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

        cov_dict[(na, nb), (nc, nd)] = mc_cov

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

        an_cov_dict[(na, nb), (nc, nd)] = analytic_cov

ps_order = [("sv1&ar1", "sv1&ar1"),
            ("sv1&ar1", "sv1&ar2"),
            ("sv1&ar2", "sv1&ar2")]

proj_dict = {
             "C1": np.array([1, -1, 0]),
             "C2": np.array([0, -1, 1]),
             "C3": np.array([1, 0, -1]),
             "C4": np.array([1, -2, 1])
            }

res_dict = {(c, spec): [] for c in proj_dict for spec in spectra}
an_cov_res_dict = {}
mc_cov_res_dict = {(c, spec): np.zeros((n_bins, n_bins)) for c in proj_dict for spec in spectra}

for iii in range(n_sims):

    ps_dict = {}
    cov_dict = {}
    for mode in modes:
        for i, spec_name1 in enumerate(spec_name_list):
            na, nb = spec_name1.split("x")
            ps_dict[na, nb] = ps_all[spec_name1][iii][mode]

            for j, spec_name2 in enumerate(spec_name_list):
                if j < i: continue
                nc, nd = spec_name2.split("x")
                an_cov = an_cov_dict[(na, nb), (nc, nd)]

                cov_dict[(na, nb), (nc, nd)] = so_cov.selectblock(an_cov, modes,
                                                                  n_bins, block = mode+mode)


        ps_vec, full_cov = consistency.append_spectra_and_cov(ps_dict, cov_dict, ps_order)

        for comb, proj_pattern in proj_dict.items():
            if comb == "C4" and not(mode in ["TT", "EE"]): continue
            ps_res, cov_res = consistency.project_spectra_vec_and_cov(ps_vec, full_cov, proj_pattern)

            res_dict[comb, mode] += [ps_res]
            if iii == 0:
                an_cov_res_dict[comb, mode] = cov_res

            mc_cov_res_dict[comb, mode] += np.outer(ps_res, ps_res)

colors = {"C1": "tab:red", "C2": "tab:blue", "C3": "tab:green", "C4": "gray"}
linestyles = {"mc": "--", "an": "solid"}

for mode in modes:
    plt.figure(figsize = (8, 6))
    for comb in proj_dict:
        if comb == "C4" and not(mode in ["TT", "EE"]): continue
        mc_cov_res_dict[comb, mode] /= n_sims
        mc_cov_res_dict[comb, mode] -= np.outer(np.mean(res_dict[comb, mode], axis = 0),
                                                np.mean(res_dict[comb, mode], axis = 0))

        mc_err = np.sqrt(mc_cov_res_dict[comb, mode].diagonal())
        an_err = np.sqrt(an_cov_res_dict[comb, mode].diagonal())

        plt.plot(lb, mc_err, label = comb, ls = linestyles["mc"], color = colors[comb])
        plt.plot(lb, an_err, label = comb, ls = linestyles["an"], color = colors[comb])

    tmp_lines = []
    for ls in linestyles.values():
        tmp_lines.append(plt.gca().plot([], [], c = "k", ls = ls)[0])
    lines, labels = plt.gca().get_legend_handles_labels()
    legend_color = plt.legend([lines[i] for i in range(len(lines)) if i % 2 == 1],
                              [labels[i] for i in range(len(lines)) if i % 2 == 1],
                              loc = "upper left")
    legend_ls = plt.legend([tmp_lines[i] for i in range(len(tmp_lines))],
                           ["MC", "AN"], loc = "upper right")

    plt.gca().add_artist(legend_color)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\sigma(\Delta D_\ell^{%s})$" % mode)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residual_error_{mode}.png", dpi = 300)
