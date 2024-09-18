"""
this is a small simulation script to check that the  radio power and trispectrum of simualtion match the theoretical expectation
"""

import numpy as np
import pylab as plt
import scipy
from pixell import enmap, curvedsky
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils, so_cov
from pspipe_utils import poisson_sources, get_data_path

test_dir = "test_poisson"
pspy_utils.create_directory(test_dir)
ref_freq_dusty_GHz = 217
ref_lam_dusty_um = 1380.0
#check that this 
if not np.isclose((ref_freq_dusty_GHz  * 1e9 )* (ref_lam_dusty_um * 1e-6), 3e8, rtol = .01):
    raise "Check values of ref freq and lambda"


S, dNdSdOmega = poisson_sources.read_bethermin_source_distrib(lam = ref_lam_dusty_um, plot_fname=f"{test_dir}/source_distrib_dusty.png")

#Scale the 15 mJy cut at 148 GHz to 217 GHz.  Asssume we are in the RJ regime and beta = 1.6
Smax_148GHz = 0.015
Smax_dusty_217GHz = (0.015) * (217 / 148)**(2 + 1.6)

poisson_power = poisson_sources.get_poisson_power(S, dNdSdOmega, plot_fname=f"{test_dir}/as_dusty.png", ref_freq_GHz=ref_freq_dusty_GHz)
trispectrum = poisson_sources.get_trispectrum(S, dNdSdOmega, ref_freq_GHz=ref_freq_dusty_GHz)

poisson_power_cut, trispectrum_cut = poisson_sources.get_power_and_trispectrum_at_Smax(S, poisson_power, trispectrum, Smax=Smax_dusty_217GHz)
print(f"poisson_power_cut: {poisson_power_cut}", f"trispectrum_cut: {trispectrum_cut}")

dunkley_dusty_data = {"S_cut_Jy" : 15e-3, "D3000" : 90, "sigmaD3000" : 10} # extracted from Dunkley et al. 

l0 = 3000 # pivot scale for the fg amplitude
fac0 = (l0 * (l0 + 1)) / (2 * np.pi)

plt.figure(figsize=(12, 10))
plt.loglog(S * 1e3, poisson_power * fac0) # a_s in Dunkley is computed at l=3000, in Dl unit
plt.ylabel("$D^{217GHz}_{\ell = 3000}$  [$\mu K^{2}$]", fontsize=22)
plt.xlabel("$S_{max}$ (mJy)", fontsize=22)
plt.xlim([1e0, 1e3])
plt.ylim(0.1, 1000)
plt.errorbar(dunkley_dusty_data["S_cut_Jy"] * 1e3, dunkley_dusty_data["D3000"], dunkley_dusty_data["sigmaD3000"], fmt=".", label = "ACT 2013 dusty (Dunkley+)")
plt.errorbar(dunkley_dusty_data["S_cut_Jy"] * 1e3, poisson_power_cut * fac0, fmt=".", label = "code prediction at 15 mJY")
plt.legend(fontsize=22)
plt.savefig(f"{test_dir}/check_dunkley_dusty.png")
plt.clf()
plt.close()

#Note, the rest of the tuto_radio file contains code to make Poisson sims and check these predictions.  
#These are not included here - there are far more dusty sources per unit area than radio, 
#so this will take too long and would ultimately show the same thing.
