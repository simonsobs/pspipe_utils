"""

"""
from pspy import pspy_utils, so_map, so_spectra, so_window, so_mcm, so_cov, sph_tools
from pspipe_utils import simulation, consistency, best_fits, covariance
from itertools import combinations
import matplotlib.pyplot as plt
from pixell import curvedsky
import matplotlib as mpl
import numpy as np
import time

# Output dir
output_dir = "result_compare"
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

ra0, ra1, dec0, dec1, res = -30, 30, -10, 10, 2
apo_type_survey = "Rectangle"
template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
binary = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
binary.data[:] = 0
binary.data[1:-1, 1:-1] = 1

lmax = 2500
lmax_sim = lmax + 500
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
        #r = np.random.uniform(0, high = 0.7)
        r = 0
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
        window[sv, ar] = so_window.create_apodization(binary,
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
n_sims = 60

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

# Save spectra and cov
spec_dir = f"{output_dir}/spectra"
cov_dir = f"{output_dir}/covariances"

pspy_utils.create_directory(spec_dir)
pspy_utils.create_directory(cov_dir)


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

for iii in range(n_sims):
    print(f"{iii:05d}")

    for key in ps_all:
        ar1, ar2 = key.split("x")
        file_name = f"{spec_dir}/Dl_{iii:05d}_{ar1}x{ar2}.dat"
        so_spectra.write_ps(file_name, lb, ps_all[key][iii], "Dl", spectra=spectra)


for key in an_cov_dict:
    xps1, xps2 = key
    ar1, ar2 = xps1
    ar3, ar4 = xps2

    np.save(f"{cov_dir}/analytic_cov_{ar1}x{ar2}_{ar3}x{ar4}.npy", an_cov_dict[key])

# Monte Carlo covariances
for key in an_cov_dict:
    xps1, xps2 = key
    ar1, ar2 = xps1
    ar3, ar4 = xps2

    ps_list_12 = []
    ps_list_34 = []
    for iii in range(n_sims):
        ps_12 = ps_all[f"{ar1}x{ar2}"][iii]
        ps_34 = ps_all[f"{ar3}x{ar4}"][iii]

        vec_12 = []
        vec_34 = []
        for spec in ["TT", "TE", "ET", "EE"]:
            vec_12 = np.append(vec_12, ps_12[spec])
            vec_34 = np.append(vec_34, ps_34[spec])
        ps_list_12 += [vec_12]
        ps_list_34 += [vec_34]

    cov_mc = 0
    for iii in range(n_sims):
        cov_mc += np.outer(ps_list_12[iii], ps_list_34[iii])
    cov_mc = cov_mc / n_sims - np.outer(np.mean(ps_list_12, axis = 0), np.mean(ps_list_34, axis = 0))

    np.save(f"{cov_dir}/mc_cov_{ar1}x{ar2}_{ar3}x{ar4}.npy", cov_mc)

op_dict = {"ratio": "aa/bb",
           "diff": "aa-bb",
           "map_diff": "aa+bb-2ab"}

ar_list = ["sv1&ar1", "sv1&ar2"]

res_ps_dict = {op: [] for op in op_dict}
res_cov_dict = {op: [] for op in op_dict}

for iii in range(n_sims):
    ps_template = f"{spec_dir}/Dl_{iii:05d}" + "_{}x{}.dat"
    cov_template = f"{cov_dir}/" + "analytic_cov_{}x{}_{}x{}.npy"

    ps_dict, cov_dict = consistency.get_ps_and_cov_dict(ar_list, ps_template,
                                                        cov_template,
                                                        mc_error_corrections = False)

    for key, op in op_dict.items():
        lb, res_ps, res_cov, chi2, pte = consistency.compare_spectra(ar_list, op, ps_dict, cov_dict)
        res_ps_dict[key].append(res_ps)
        res_cov_dict[key].append(res_cov)



mean_res_ps = {op: np.mean(res_ps_dict[op], axis = 0) for op in res_ps_dict}
mean_res_cov = {op: 1/n_sims * np.mean(res_cov_dict[op], axis = 0) for op in res_ps_dict}
mean_res_std = {op: np.sqrt(mean_res_cov[op].diagonal()) for op in res_ps_dict}

fig, axes = plt.subplots(3, 1, figsize = (10, 15))
for i, op in enumerate(op_dict):
    ax = axes[i]
    if i == 2:
        ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\Delta D_\ell^{TT}$")

    x_plot = lb
    y_plot = mean_res_ps[op]
    yerr_plot = mean_res_std[op]

    if op == "ratio":
        res = y_plot - 1
        ax.axhline(1, color = "k", ls = "--")
    else:
        res = y_plot
        ax.axhline(0, color = "k", ls = "--")
    chi2 = res @ np.linalg.inv(mean_res_cov[op]) @ res
    print(f"X2 = {chi2}/{len(lb)}")
    ax.errorbar(x_plot, y_plot, yerr_plot, ls = "None", marker = "o", label = op)
    ax.legend()
plt.tight_layout()
plt.show()
