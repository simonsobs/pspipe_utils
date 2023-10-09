"""
"""
import numpy as np
import pylab as plt
from pspipe_utils import pol_angle, simulation
from pspy import pspy_utils, so_spectra
from pixell import curvedsky

result_dir = "result_pol_rot"
pspy_utils.create_directory(result_dir)

type = "Dl"
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

lmax= 3000
cosmo_params = {"cosmomc_theta":0.0104085,
                "logA": 3.044,
                "ombh2": 0.02237,
                "omch2": 0.1200,
                "ns": 0.9649,
                "tau": 0.0544}

l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax)
f_name_cmb = f"{result_dir}/cmb.dat"
so_spectra.write_ps(f_name_cmb, l_th, ps_dict, type, spectra=spectra)

ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)
alms_cmb = curvedsky.rand_alm(ps_mat, lmax=lmax, dtype="complex64")


phi_alpha = 10
phi_beta = -15

alms_rot_alpha = pol_angle.rot_alms(alms_cmb, phi_alpha)
alms_rot_beta = pol_angle.rot_alms(alms_cmb, phi_beta)


l, ps_rot_estimated = so_spectra.get_spectra_pixell(alms_rot_alpha,
                                                    alms_rot_beta,
                                                    spectra=spectra)

l_th, psth_rot = pol_angle.rot_theory_spectrum(l_th, ps_dict, phi_alpha, phi_beta)

for spec in spectra:
    plt.plot(l, ps_rot_estimated[spec] * l * (l + 1) / (2 * np.pi),"--", label="rot alms")
    plt.plot(l_th, psth_rot[spec], label="rot Cls")
    plt.plot(l_th, ps_dict[spec], label="LCDM")
    plt.legend()
    plt.xlabel(r"$\ell$", fontsize = 15)
    plt.ylabel(r"$D_\ell^{%s}$" % spec, fontsize = 15)
    plt.show()

