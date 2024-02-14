"""
this is a small simulation script to check that the  radio power and trispectrum of simualtion match the theoretical expectation
"""

import numpy as np
import pylab as plt
import scipy
from pixell import enmap, curvedsky
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils, so_cov
from pspipe_utils import radio_sources

test_dir = "test_poisson"
pspy_utils.create_directory(test_dir)

S, dNdSdOmega = radio_sources.read_tucci_source_distrib(plot_fname=f"{test_dir}/source_distrib.png")

poisson_power = radio_sources.get_poisson_power(S, dNdSdOmega, plot_fname=f"{test_dir}/as.png")
trispectrum = radio_sources.get_trispectrum(S, dNdSdOmega)

poisson_power_15mJy, trispectrum_15mJy = radio_sources.get_power_and_trispectrum_at_Smax(S, poisson_power, trispectrum, Smax=0.015)
print(f"poisson_power_15mJy: {poisson_power_15mJy}", f"trispectrum_15mJy: {trispectrum_15mJy}")

dunkley_radio_data = {"S_cut_Jy" : 15e-3, "D3000" : 3.1, "sigmaD3000" : .4} # extracted from Dunkley et al

l0 = 3000 # pivot scale for the fg amplitude
fac0 = (l0 * (l0 + 1)) / (2 * np.pi)

plt.figure(figsize=(12, 10))
plt.loglog(S * 1e3, poisson_power * fac0) # a_s in Dunkley is computed at l=3000, in Dl unit
plt.ylabel("$D^{148GHz}_{\ell = 3000}$  [$\mu K^{2}$]", fontsize=22)
plt.xlabel("$S_{max}$ (mJy)", fontsize=22)
plt.xlim([1e0, 1e3])
plt.ylim(0.1, 1000)
plt.errorbar(dunkley_radio_data["S_cut_Jy"] * 1e3, dunkley_radio_data["D3000"], dunkley_radio_data["sigmaD3000"], fmt=".", label = "ACT 2013 radio(Dunkley+)")
plt.errorbar(dunkley_radio_data["S_cut_Jy"] * 1e3, poisson_power_15mJy * fac0, fmt=".", label = "code prediction at 15 mJY")
plt.legend(fontsize=22)
plt.savefig(f"{test_dir}/check_dunkley.png")
plt.clf()
plt.close()

#### Now do a montecarlo to check if it work, we will generate the sim at the ref frequency 148

ra0, ra1, dec0, dec1, res = -180, 180, -5, 5, 1
ncomp = 1
ps_type = "Cl"
lmax = 3000
l = np.arange(2, lmax + 2)
rms_uKarcmin_T = 1
niter = 0
pspy_utils.create_binning_file(bin_size=100, n_bins=300, file_name=f"{test_dir}/binning.dat")
binning_file = f"{test_dir}/binning.dat"
n_splits = 2
ref_freq = 148
Jy_per_str_to_muK = radio_sources.convert_Jy_per_str_to_muK_cmb(ref_freq)

if ps_type=="Dl":
    tex_name = "D_{\ell}"
    fac = l * (l + 1) / (2 * np.pi)
if ps_type=="Cl":
    fac = l * 0 + 1
    tex_name = "C_{\ell}"

S_min_Jy = .001
S_max_Jy = 1.58e-02 #15.8 mJY flux cut

poisson_power_15_8mJy  = poisson_power[S == S_max_Jy]
trispectrum_15_8mJy = trispectrum[S == S_max_Jy]

print(f"poisson_power_15.8mJy: {poisson_power_15_8mJy}", f"trispectrum_15.8mJy: {trispectrum_15_8mJy}")

ps_theory = poisson_power_15_8mJy * fac
l_th, nl_th = pspy_utils.get_nlth_dict(rms_uKarcmin_T, ps_type, lmax)

survey_id = ["Ta", "Tb", "Tc", "Td"]
survey_name = ["split_0", "split_1", "split_0", "split_1"]

Clth_dict = {}
for name1, id1 in zip(survey_name, survey_id):
    for name2, id2 in zip(survey_name, survey_id):
        spec = id1[0] + id2[0]
        Clth_dict[id1 + id2] = ps_theory + nl_th["TT"] * so_cov.delta2(name1, name2)
        Clth_dict[id1 + id2] = Clth_dict[id1 + id2][:-2]
        
template_car = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
shape, wcs = template_car.data.shape, template_car.data.wcs

source_mean_numbers = radio_sources.get_mean_number_of_source(template_car, S, dNdSdOmega, plot_fname=f"{test_dir}/N_source.png")

binary = template_car.copy()
binary.data[:] = 0
binary.data[1:-1, 1:-1] = 1
window = so_window.create_apodization(binary, apo_type="C1", apo_radius_degree=1)

mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0(window, binning_file, lmax=lmax, type=ps_type, niter=niter)

coupling_dict = so_cov.cov_coupling_spin0(window, lmax, niter=niter)
analytic_cov = so_cov.cov_spin0(Clth_dict, coupling_dict, binning_file, lmax, mbb_inv, mbb_inv)

n_sims = 400

sim_types = ["sim_poissonian"]
mean, std, cov = {}, {}, {}

for sim_type in sim_types:
    Db_list = []
    cov_mc = 0
    for iii in range(n_sims):
        print(sim_type, iii)
    
        source_map = template_car.copy()
        if sim_type == "sim_poissonian":
            my_numbers = np.random.poisson(source_mean_numbers)
    
            for si, fluxval in enumerate(S[S <= S_max_Jy]):
                xlocs = np.random.randint(shape[0], size = my_numbers[si])
                ylocs = np.random.randint(shape[1], size = my_numbers[si])
                
                source_area_str = (res * np.pi / (180. * 60.)) ** 2
                source_map.data[xlocs, ylocs] += Jy_per_str_to_muK * fluxval / source_area_str
                
            source_map.data -= np.mean(source_map.data)
            
        if sim_type == "sim_gaussian":
            source_map.data = curvedsky.rand_map(source_map.data.shape, source_map.data.wcs, ps_theory/fac)
    
        almList = []
        for i in range(n_splits):
            split = source_map.copy()
            noise = so_map.white_noise(split, rms_uKarcmin_T=rms_uKarcmin_T)
            split.data += noise.data
            almList += [sph_tools.get_alms(split, window, niter, lmax)]

        l_, ps = so_spectra.get_spectra(almList[0], almList[1])
        lb, Db = so_spectra.bin_spectra(l_,
                                        ps,
                                        binning_file,
                                        lmax,
                                        type=ps_type,
                                        mbb_inv=mbb_inv)

        Db_list += [Db]
        cov_mc += np.outer(Db, Db)

    mean[sim_type], std[sim_type] = np.mean(Db_list, axis=0), np.std(Db_list, axis=0)
    cov[sim_type] = cov_mc / n_sims - np.outer(mean[sim_type], mean[sim_type])


plt.figure(figsize=(12, 10))
plt.plot(l, ps_theory, label="theory")
for sim_type in sim_types:
    plt.errorbar(lb, mean[sim_type], std[sim_type], fmt="o", label=sim_type)
plt.ylabel("$%s$" % tex_name, fontsize=22)
plt.xlabel("$\ell$", fontsize=22)
plt.legend(fontsize=22)
plt.savefig(f"{test_dir}/power_spectrum.png", bbox_inches="tight")
plt.clf()
plt.close()

if ps_type=="Dl":
    fac_b = lb * (lb + 1) / (2 * np.pi)
if ps_type=="Cl":
    fac_b = lb * 0 + 1
    
area_sr = enmap.area(shape, wcs)
win_corr = np.sum(window.data)/np.size(binary.data) # correction to the area due to apodisation
area_sr *= win_corr
print(f"area_sr = {area_sr}", f"win correction = {win_corr}")

plt.figure(figsize=(12, 10))
plt.plot(lb, analytic_cov.diagonal() + trispectrum_15_8mJy / area_sr * fac_b ** 2, label=r"$\Sigma^{\rm gauss}_{\ell, \ell} + T_{P}/\Omega$")
plt.plot(lb, analytic_cov.diagonal(), label=r"$\Sigma^{\rm gauss}_{\ell, \ell}$", color="gray")
for sim_type in sim_types:
    plt.plot(lb, cov[sim_type].diagonal(), "o", label=f"MC cov {sim_type}")
plt.ylabel("$\Sigma_{\ell, \ell}$" , fontsize=22)
plt.xlabel("$\ell$", fontsize=22)
plt.legend(fontsize=22)
plt.savefig(f"{test_dir}/covariance.png", bbox_inches="tight")
plt.clf()
plt.close()


