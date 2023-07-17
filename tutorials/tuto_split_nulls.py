"""
Tutorial to get split covariances.
To run this script, you must have set to
`True` the `write_split_spectra` flag in the paramfile.
"""
from pspy import so_dict, pspy_utils, so_spectra, so_cov, so_mcm
from pspipe_utils import pspipe_list, best_fits
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement as cwr
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

tuto_data_dir = "tuto_data"
result_dir = "result_simulation"

split_dir = "result_split_nulls"
pspy_utils.create_directory(split_dir)

surveys = d["surveys"]
arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

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
dlth_and_noise_dict = {}
for na, nb, m in ps_all_th:
    sv = na.split("&")[0]
    for s1, s2 in cwr(np.arange(n_splits[sv], dtype=np.int32), 2):

        if s1 == s2:
            dlth_and_noise_dict[f"{na}&{s1}", f"{nb}&{s2}", m] = ps_all_th[na, nb, m] + 2*nl_all_th[na, nb, m]
        else:
            dlth_and_noise_dict[f"{na}&{s1}", f"{nb}&{s2}", m] = ps_all_th[na, nb, m]
            dlth_and_noise_dict[f"{na}&{s2}", f"{nb}&{s1}", m] = ps_all_th[na, nb, m]

spec_name_for_cov = []
for sv in surveys:
    for ar in arrays[sv]:
        cross_split_list = list(combinations(np.arange(n_splits[sv], dtype=np.int32), 2))
        for s1, s2 in cross_split_list:
            spec_name_for_cov += [f"{sv}&{ar}&{s1}x{sv}&{ar}&{s2}"]

cov_names = []
for id1, spec1 in enumerate(spec_name_for_cov):
    for id2, spec2 in enumerate(spec_name_for_cov):
        if id1 > id2: continue
        na, nb = spec1.split("x")
        nc, nd = spec2.split("x")

        cov_names.append((na,nb,nc,nd))

for na,nb,nc,nd in cov_names:

    print(f"Computing cov element {na}x{nb}_{nc}x{nd} ...")
    sv_a, ar_a, sa = na.split("&")
    sv_b, ar_b, sb = nb.split("&")
    sv_c, ar_c, sc = nc.split("&")
    sv_d, ar_d, sd = nd.split("&")

    na = f"{sv_a}&{ar_a}"
    nb = f"{sv_b}&{ar_b}"
    nc = f"{sv_c}&{ar_c}"
    nd = f"{sv_d}&{ar_d}"

    na_r, nb_r, nc_r, nd_r = na.replace("&", "_"), nb.replace("&", "_"), nc.replace("&", "_"), nd.replace("&", "_")

#    coupling = so_cov.fast_cov_coupling_spin0and2(tuto_data_dir,
#                                                  [na, nb, nc, nd],
#                                                  lmax)
#
#
#
#    # These objects are symmetric in (sv_a, ar_a) and (sv_b, ar_b)
#    try: mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(prefix=f"{tuto_data_dir}/{na}x{nb}", spin_pairs=spin_pairs)
#    except: mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(prefix=f"{tuto_data_dir}/{nb}x{na}", spin_pairs=spin_pairs)
#
#    try: mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(prefix=f"{tuto_data_dir}/{nc}x{nd}", spin_pairs=spin_pairs)
#    except: mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(prefix=f"{tuto_data_dir}/{nd}x{nc}", spin_pairs=spin_pairs)
#
#
#    cross_dict = {
#        "a": f"{na}&{sa}",
#        "b": f"{nb}&{sb}",
#        "c": f"{nc}&{sc}",
#        "d": f"{nd}&{sd}"
#    }
#
#    dlth_dict = {}
#    for field1 in ["T", "E", "B"]:
#        for id_1, cross_name_1 in cross_dict.items():
#            for field2 in ["T", "E", "B"]:
#                for id_2, cross_name_2 in cross_dict.items():
#                    dlth_dict[f"{field1}{id_1}{field2}{id_2}"] = dlth_and_noise_dict[cross_name_1, cross_name_2, field1+field2]
#
#    analytic_cov = so_cov.cov_spin0and2(dlth_dict,
#                                        coupling,
#                                        binning_file,
#                                        lmax,
#                                        mbb_inv_ab,
#                                        mbb_inv_cd,
#                                       binned_mcm=binned_mcm,
#                                        cov_T_E_only=False,
#                                        dtype=np.float32)
#
#    np.save(f"{split_dir}/analytic_cov_{na_r}x{nb_r}_{nc_r}x{nd_r}_{sa}{sb}x{sc}{sd}.npy", analytic_cov)
    analytic_cov = np.load(f"{split_dir}/analytic_cov_{na_r}x{nb_r}_{nc_r}x{nd_r}_{sa}{sb}x{sc}{sd}.npy")
    spec_name_ab = f"{na}x{nb}_{sa}{sb}"
    spec_name_cd = f"{nc}x{nd}_{sc}{sd}"

    spectra_list_ab = [so_spectra.read_ps(f"{result_dir}/Dl_{spec_name_ab}_{iii:05d}.dat", spectra=spectra)[1] for iii in range(d["n_sims"])]
    spectra_list_cd = [so_spectra.read_ps(f"{result_dir}/Dl_{spec_name_cd}_{iii:05d}.dat", spectra=spectra)[1] for iii in range(d["n_sims"])]

    ps_mean_ab = np.zeros(analytic_cov.shape[0])
    ps_mean_cd = np.zeros_like(ps_mean_ab)
    mc_cov = np.zeros_like(analytic_cov)
    for iii in range(d["n_sims"]):
        ps_dict_ab = spectra_list_ab[iii]
        ps_dict_cd = spectra_list_cd[iii]

        vec_ab = np.array([ps_dict_ab[spec] for spec in spectra]).flatten()
        vec_cd = np.array([ps_dict_cd[spec] for spec in spectra]).flatten()
        mc_cov += np.outer(vec_ab, vec_cd)
        ps_mean_ab += vec_ab
        ps_mean_cd += vec_cd

    ps_mean_ab /= d["n_sims"]
    ps_mean_cd /= d["n_sims"]
    mc_cov = mc_cov / d["n_sims"] - np.outer(ps_mean_ab, ps_mean_cd)

    np.save(f"{split_dir}/mc_cov_{na_r}x{nb_r}_{nc_r}x{nd_r}_{sa}{sb}x{sc}{sd}.npy", mc_cov)

    fig, axes = plt.subplots(3, 3, figsize = (12, 10))
    for i, spec in enumerate(spectra):

        nbins = len(mc_cov)//len(spectra)
        id_row = i // 3
        id_col = i % 3

        ax = axes[id_row, id_col]

        ax.plot(lb, analytic_cov.diagonal()[i*nbins:(i+1)*nbins], color = "navy", label = "Analytic")
        ax.plot(lb, mc_cov.diagonal()[i*nbins:(i+1)*nbins], color = "darkorange", label = "Montecarlo")

        if i == 0:
            ax.legend()

        if spec in ["TT", "EE", "BB"]:
            ax.set_yscale("log")

        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\sigma^2(D_\ell^{%s})$" % spec)

    fig.suptitle(f"{spec_name_ab}_{spec_name_cd}")
    plt.tight_layout()
    plt.savefig(f"{split_dir}/split_cov_{na}x{nb}_{nc}x{nd}_{sa}{sb}{sc}{sd}.png", bbox_inches="tight")
