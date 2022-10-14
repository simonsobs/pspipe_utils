import matplotlib.pyplot as plt
import numpy as np
from pspipe_utils import best_fits
from pspipe_utils import external_data as ext
from pspy import pspy_utils

output_dir = "results_passbands"
pspy_utils.create_directory(output_dir)

# Load passbands
dr6_wafers = ["pa4_f150", "pa4_f220", "pa5_f090", "pa5_f150", "pa6_f090", "pa6_f150"]
npipe_wafers = [f"npipe_f{freq}" for freq in [100, 143, 217, 353, 545, 857]]
npipe_freq_range = [(50, 1100) for wafer in npipe_wafers]

npipe_passbands = ext.get_passband_dict_npipe(npipe_wafers, freq_range_list=npipe_freq_range)
dr6_passbands = ext.get_passband_dict_dr6(dr6_wafers)

passbands = {**dr6_passbands, **npipe_passbands}

# Plot passbands
plt.figure(figsize = (8, 6))
plt.xlabel(r"$\nu$ [GHz]")
for wafer in dr6_wafers:
    nu_ghz, trans = dr6_passbands[wafer]
    plt.plot(nu_ghz, trans / np.trapz(trans, nu_ghz), label = wafer)
plt.legend()
plt.ylim(bottom = 0)
plt.tight_layout()
plt.savefig(f"{output_dir}/dr6_passbands.png", dpi = 300)

plt.figure(figsize = (8, 6))
plt.xlabel(r"$\nu$ [GHz]")
for wafer in npipe_wafers:
    nu_ghz, trans = npipe_passbands[wafer]
    plt.plot(nu_ghz, trans / np.trapz(trans, nu_ghz), label = wafer)
plt.legend()
plt.ylim(bottom = 0)
plt.tight_layout()
plt.savefig(f"{output_dir}/npipe_passbands.png", dpi = 300)


# Compute foreground models
fg_components = {
    "tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
    "te": ["radio", "dust"],
    "ee": ["radio", "dust"],
    "bb": ["radio", "dust"],
    "tb": ["radio", "dust"],
    "eb": []
}

fg_params = {
    "a_tSZ": 3.30,
    "a_kSZ": 1.60,
    "a_p": 5.71,
    "beta_p": 2.20,
    "a_c": 6.53,
    "beta_c": 2.20,
    "a_s": 3.10,
    "xi": 0.1,
    "T_d": 9.60,
    "a_gtt": 13.5,
    "a_gte": 0.36,
    "a_gtb": 0.36,
    "a_gee": 0.13,
    "a_gbb": 0.13,
    "a_psee": 0.05,
    "a_psbb": 0.05,
    "a_pste": 0,
    "a_pstb": 0
}

ell = np.arange(30, 5000)
frequencies = [] # No need for frequencies when using external passbands
fg_dict = best_fits.get_foreground_dict(ell, frequencies, fg_components,
                                        fg_params, external_bandpass = passbands)


# Impact of bandpass integration on the calibration
cosmo_params = {"cosmomc_theta":0.0104085, "logA": 3.044, "ombh2": 0.02237,
                "omch2": 0.1200,  "ns": 0.9649, "tau": 0.0544}
l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, "Dl", lmax = ell.max()+1)

plt.figure(figsize = (8, 6))
for wafer in ["pa4_f150", "pa5_f150", "pa6_f150"]:
    dr6xdr6 = ps_dict["TT"][l_th >= ell.min()] + fg_dict["tt", "all", wafer, wafer]
    dr6xnpipe = ps_dict["TT"][l_th >= ell.min()] + fg_dict["tt", "all", wafer, "npipe_f143"]
    res = dr6xdr6 / dr6xnpipe
    mask = (ell >= 1200) & (ell <= 1800) # calibration range
    l0, = plt.plot(ell[mask], res[mask] - 1, label = wafer)
    plt.axhline(np.mean(res[mask]) - 1, color = l0.get_color(), ls = "--", lw = 0.6)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$D_\ell^{TT, \mathrm{ACT}\times\mathrm{ACT}} / D_\ell^{TT, \mathrm{ACT}\times\mathrm{NPIPE}} - 1$")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/deltaDl_150ghz.png", dpi = 300)

plt.figure(figsize = (8, 6))
for wafer in ["pa5_f090", "pa6_f090"]:
    dr6xdr6 = ps_dict["TT"][l_th >= ell.min()] + fg_dict["tt", "all", wafer, wafer]
    dr6xnpipe = ps_dict["TT"][l_th >= ell.min()] + fg_dict["tt", "all", wafer, "npipe_f100"]
    res = dr6xdr6 / dr6xnpipe
    mask = (ell >= 800) & (ell <= 1300) # calibration range
    l0, = plt.plot(ell[mask], res[mask] - 1, label = wafer)
    plt.axhline(np.mean(res[mask]) - 1, color = l0.get_color(), ls = "--", lw = 0.6)

plt.xlabel(r"$\ell$")
plt.ylabel(r"$D_\ell^{TT, \mathrm{ACT}\times\mathrm{ACT}} / D_\ell^{TT, \mathrm{ACT}\times\mathrm{NPIPE}} - 1$")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/deltaDl_090ghz.png", dpi = 300)
