"""
"""
import numpy as np
from pspipe_utils import best_fits, pspipe_list
from pspy import pspy_utils, so_spectra, so_map, so_window, so_dict, so_mcm, sph_tools
from itertools import combinations
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

tuto_data_dir = "tuto_data"
pspy_utils.create_directory(tuto_data_dir)
ncomp, niter = 3, 0
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
type = "Dl"

lmax = d["lmax"]
lmax_sim = lmax + 500

surveys = d["surveys"]
cosmo_params = d["cosmo_params"]
fg_components = d["fg_components"]
fg_params = d["fg_params"]
binned_mcm = d["binned_mcm"]

bin_size = d["bin_size"]
binning_file = f"{tuto_data_dir}/binning.dat"
pspy_utils.create_binning_file(bin_size=bin_size, n_bins=300, file_name=binning_file)

bl, arrays, rms_uKarcmin_T = {}, {}, {}

for sv in surveys:
    arrays[sv] = d[f"arrays_{sv}"]

    for ar in arrays[sv]:
        rms_uKarcmin_T[sv, f"{ar}x{ar}"] = d[f"rms_uKarcmin_T_{sv}_{ar}"]

        fwhm =  d[f"beam_fwhm_{sv}_{ar}"]
        l_beam, bl[sv, ar, "T"] = pspy_utils.beam_from_fwhm(fwhm, lmax_sim)
        l_beam, bl[sv, ar, "E"] = pspy_utils.beam_from_fwhm(fwhm * 1.3, lmax_sim)

        f_name = f"{tuto_data_dir}/beam_{sv}_{ar}_T.dat"
        np.savetxt(f_name, np.transpose([l_beam, bl[sv, ar, "T"]]))
        
        f_name = f"{tuto_data_dir}/beam_{sv}_{ar}_pol.dat"
        np.savetxt(f_name, np.transpose([l_beam, bl[sv, ar, "E"]]))


for sv in surveys:
    for ar1, ar2 in combinations(arrays[sv], 2):
        r = d["r_xar"]
        rms_uKarcmin_T[sv, f"{ar1}x{ar2}"] = r * np.sqrt(rms_uKarcmin_T[sv, f"{ar1}x{ar1}"] * rms_uKarcmin_T[sv, f"{ar2}x{ar2}"])
        rms_uKarcmin_T[sv, f"{ar2}x{ar1}"] = rms_uKarcmin_T[sv, f"{ar1}x{ar2}"]

l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax_sim)
f_name = f"{tuto_data_dir}/cmb.dat"
so_spectra.write_ps(f_name, l_th, ps_dict, type, spectra=spectra)

pspy_utils.create_directory(d["passband_dir"])
passbands = {}

narrays, sv_list, ar_list = pspipe_list.get_arrays_list(d)
for sv, ar in zip(sv_list, ar_list):
    freq_info = d[f"freq_info_{sv}_{ar}"]
    if d["do_bandpass_integration"]:
        central_nu, delta_nu = freq_info["freq_tag"], d[f"bandwidth_{sv}_{ar}"]
        nu_min, nu_max = central_nu - 2 * delta_nu, central_nu + 2 * delta_nu
        nu_ghz = np.linspace(nu_min, nu_max, 200)
        bp = np.where(nu_ghz > central_nu + delta_nu, 0., np.where(nu_ghz < central_nu - delta_nu, 0., 1.))
        np.savetxt(freq_info["passband"], np.array([nu_ghz, bp]).T)
    else:
        nu_ghz, bp = np.array([freq_info["freq_tag"]]), np.array([1])

    passbands[f"{sv}_{ar}"] = [nu_ghz, bp]


fg_dict = best_fits.get_foreground_dict(l_th, passbands, fg_components, fg_params, fg_norm=None)

fg= {}
for sv1, ar1 in zip(sv_list, ar_list):
    for sv2, ar2 in zip(sv_list, ar_list):
        name1 = f"{sv1}_{ar1}"
        name2 = f"{sv2}_{ar2}"
        fg[name1, name2] = {}
        for spec in spectra:
            fg[name1, name2][spec] = fg_dict[spec.lower(), "all", name1, name2]

        so_spectra.write_ps(f"{tuto_data_dir}/fg_{name1}x{name2}.dat", l_th, fg[name1, name2], type, spectra=spectra)

for sv in surveys:
    for ar1 in arrays[sv]:
        for ar2 in arrays[sv]:
            l_th, nl_th = pspy_utils.get_nlth_dict(rms_uKarcmin_T[sv, f"{ar1}x{ar2}"],
                                                                  type,
                                                                  lmax_sim,
                                                                  spectra=spectra)
            for spec in ["TE", "TB", "EB"]:
                r = d["r_xcomp"]
                s1, s2 = spec
                nl_th[s1 + s2] = r * np.sqrt(nl_th[s1 + s1] * nl_th[s2 + s2] )
                nl_th[s2 + s1] = nl_th[s1 + s2]

            f_name = f"{tuto_data_dir}/mean_{ar1}x{ar2}_{sv}_noise.dat"
            so_spectra.write_ps(f_name, l_th, nl_th, type, spectra=spectra)

ra0, ra1, dec0, dec1, res = d["ra0"], d["ra1"], d["dec0"], d["dec1"], d["res"]
apo_type_survey = d["apo_type_survey"]

template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
template.write_map(f"{tuto_data_dir}/template.fits")

binary = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
binary.data[:] = 0
binary.data[1:-1, 1:-1] = 1

window = {}
for sv in surveys:
    for ar in arrays[sv]:
        window[sv, ar] = so_window.create_apodization(binary,
                                              apo_type=apo_type_survey,
                                              apo_radius_degree=d["apo_radius_degree"])
        window[sv, ar].plot(file_name=f"{tuto_data_dir}/window_{sv}_{ar}")
        window[sv, ar].write_map(f"{tuto_data_dir}/window_{sv}_{ar}.fits")


for id_sv_a, sv_a in enumerate(surveys):
    for id_ar_a, ar_a in enumerate(arrays[sv_a]):
        # we need both the window for T and pol, here we assume they are the same

        window_tuple_a = (window[sv_a, ar_a], window[sv_a, ar_a])
        bl_a = (bl[sv_a, ar_a, "T"], bl[sv_a, ar_a, "E"])

        for id_sv_b, sv_b in enumerate(surveys):
            for id_ar_b, ar_b in enumerate(arrays[sv_b]):

                if  (id_sv_a == id_sv_b) & (id_ar_a > id_ar_b) : continue
                if  (id_sv_a > id_sv_b) : continue

                spec_name = f"{sv_a}&{ar_a}x{sv_b}&{ar_b}"

                # the if here help avoiding redondant computation
                # the mode coupling matrices for sv1 x sv2 are the same as the one for sv2 x sv1
                # identically, within a survey, the mode coupling matrix for ar1 x ar2 =  ar2 x ar1
                window_tuple_b = (window[sv_b, ar_b], window[sv_b, ar_b])
                bl_b = (bl[sv_b, ar_b, "T"], bl[sv_b, ar_b, "E"])

                mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window_tuple_a,
                                                            win2 = window_tuple_b,
                                                            bl1 = bl_a,
                                                            bl2 = bl_b,
                                                            binning_file = binning_file,
                                                            lmax=lmax,
                                                            type="Dl",
                                                            niter=niter,
                                                            binned_mcm=binned_mcm,
                                                            save_file=f"{tuto_data_dir}/{spec_name}")

                sq_win = window[sv_a, ar_a].copy()
                sq_win.data *= window[sv_b, ar_b].data
                sqwin_alm = sph_tools.map2alm(sq_win, niter=niter, lmax=lmax)

                np.save(f"{tuto_data_dir}/alms_{spec_name}.npy", sqwin_alm)
