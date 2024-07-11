
from pspy import so_map, so_window, sph_tools, so_spectra, pspy_utils, so_map_preprocessing, so_mcm, so_cov
import pylab as plt
import numpy as np
import pspipe_utils
from pspipe_utils import simulation, best_fits, kspace, misc
from pixell import curvedsky, enmap
import os, time
import matplotlib


matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "20"


test_dir = "result_leakage_ell_by_ell"
pspy_utils.create_directory(test_dir)
data_path = os.path.join(os.path.dirname(os.path.abspath(pspipe_utils.__file__)), "data/")

########################################################
# let's first specify the parameters used in the analysis
########################################################

ncomp = 3
niter = 0
n_sims = 100
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
binning_file = f"{data_path}/binning_files/BIN_ACTPOL_50_4_SC_large_bin_at_low_ell"
vk_mask = [-90, 90]
hk_mask = [-50, 50]
binned_mcm = False
type = "Cl"
cosmo_params = {"cosmomc_theta":0.0104085,
                "logA": 3.044,
                "ombh2": 0.02237,
                "omch2": 0.1200,
                "ns": 0.9649,
                "tau": 0.0544}

array = "pa6_f150"
map_set = f"dr6_{array}"
binary = so_map.read_map(f"{data_path}/binaries/binary_{map_set}_downgraded.fits")
lmax = int(binary.get_lmax_limit())

template = binary.copy()
ny, nx = binary.data.shape
template.ncomp = 3
template.data = enmap.empty((3, ny, nx), wcs=binary.data.wcs)

window_binary = binary.copy()
dist = so_window.get_distance(window_binary, rmax= np.pi / 180)
window_binary.data[dist.data < 0.5] = 0

window = so_window.create_apodization(window_binary, apo_type="C1", apo_radius_degree=1)
mask = so_map.simulate_source_mask(window_binary, n_holes=1000, hole_radius_arcmin=10)
mask = so_window.create_apodization(mask, apo_type="C1", apo_radius_degree=0.3)
window.data *= mask.data

window.plot(file_name=f"{test_dir}/window_{map_set}")

binary.plot(file_name=f"{test_dir}/binary_{map_set}")
window_tuple = (window, window)
    

mcm_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window_tuple,
                                            binning_file = binning_file,
                                            lmax=lmax,
                                            type=type,
                                            niter=niter,
                                            binned_mcm=binned_mcm)

filter_std = so_map_preprocessing.build_std_filter(template.data.shape,
                                                   template.data.wcs,
                                                   vk_mask,
                                                   hk_mask,
                                                   dtype=np.float64)
                                          
    



l_impuls = 500
ps_dict = {}
for spec in spectra:
    ps_dict[spec] = np.zeros(lmax + 100)
ps_dict["TT"][l_impuls-2] = (l_impuls * (l_impuls + 1)) / (2 * np.pi)
ps_dict["EE"][l_impuls-2] = (l_impuls * (l_impuls + 1)) / (2 * np.pi)

l_th = np.arange(2, len(ps_dict["TT"]) + 2)

f_name_cmb = test_dir + "/cmb.dat"
so_spectra.write_ps(f_name_cmb, l_th, ps_dict, "Dl", spectra=spectra)
ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)



ps_list = {}
for spec in spectra:
    ps_list["nofilter", spec] = []
    ps_list["filter", spec] = []
for iii in range(n_sims):

    t = time.time()
    
    alms_cmb = curvedsky.rand_alm(ps_mat, lmax=lmax, dtype="complex64")
    cmb = sph_tools.alm2map(alms_cmb, template)

    alms = sph_tools.get_alms(cmb, window_tuple, niter, lmax)

    _, ps = so_spectra.get_spectra(alms,
                                   alms,
                                   spectra=spectra)
                                   
        
    l = np.arange(2, lmax)
    cl = {f: ps[f][l] for f in spectra}
    l, cl = so_spectra.deconvolve_mode_coupling_matrix(l, cl, mcm_inv, spectra)


    cmb_cut_filter = cmb.copy()
    my_binary = binary.copy()

    cmb_cut_filter = kspace.filter_map(cmb_cut_filter, filter_std, my_binary)

    alms_cut_filter = sph_tools.get_alms(cmb_cut_filter, window_tuple, niter, lmax)
        
    _, ps_cut_filter = so_spectra.get_spectra(alms_cut_filter,
                                              alms_cut_filter,
                                              spectra=spectra)
                                              
    l = np.arange(2, lmax)
    cl_cut_filter = {f: ps_cut_filter[f][l] for f in spectra}
    l, cl_cut_filter = so_spectra.deconvolve_mode_coupling_matrix(l, cl_cut_filter, mcm_inv, spectra)

    for spec in spectra:
        ps_list["nofilter", spec] += [cl[spec]]
        ps_list["filter", spec] += [cl_cut_filter[spec]]

    print(f"sim %03d/%03d took %.2f s to compute" % (iii, n_sims, time.time() - t))


mean_filter, mean_nofilter = {}, {}
for spec in spectra:
    mean_filter[spec] = np.mean(ps_list["filter", spec], axis=0)
    mean_nofilter[spec] = np.mean(ps_list["nofilter", spec], axis=0)


plt.figure(figsize=(12,12))
plt.suptitle("Response to $C^{TT}_\ell = \delta_{\ell,500}$ and $C^{EE}_\ell = \delta_{\ell,500}$ ")
plt.plot(l, mean_nofilter["TT"], color="darkcyan", label=r"$C^{TT}_{\ell}$", alpha=0.8, linewidth=2)
plt.plot(l, mean_nofilter["EE"], color="darkviolet", linestyle=(0, (5, 10)), label=r"$C^{EE}_{\ell}$", linewidth=2)
plt.plot(l, mean_filter["TT"], color="grey", label=r"$C^{TT, F}_{\ell}$", linewidth=2)
plt.plot(l, mean_filter["EE"], color="red", linestyle=(0, (5, 10)), label=r"$C^{EE, F}_{\ell}$", alpha=0.8, linewidth=2)
#plt.plot(l, np.abs(mean_filter["BB"]), color="green", label=r"$C^{BB, F}_{\ell}$")
#plt.plot(l, np.abs(mean_filter["EB"]), color="purple", label=r"$C^{EB, F}_{\ell}$", linestyle="--")
plt.legend()
plt.xlabel(r"$\ell$", fontsize=30)
plt.xlim(l_impuls-10, l_impuls+10)
plt.xticks([l_impuls-10, l_impuls-5, l_impuls, l_impuls+5, l_impuls+10])
#plt.plot(l, mean_nofilter/np.max(mean_nofilter))
plt.savefig(f"{test_dir}/kspace_leakage_{l_impuls}.png", bbox_inches='tight', dpi=300)
plt.clf()
plt.close()
