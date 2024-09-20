"""
This part read the monofrequency beam and rescale it according to frequency
"""
import pylab as plt
import numpy as np
from pspipe_utils import get_data_path, beam_chromaticity, external_data
from pspy import pspy_utils
import matplotlib as mpl


output_dir = "beam_chorma"
pspy_utils.create_directory(output_dir)


data_path = get_data_path()
arrays = ["pa4_f220", "pa5_f090", "pa5_f150", "pa6_f090", "pa6_f150"]
lmax = 10000
alpha_dict, nu_ref_dict = beam_chromaticity.act_dr6_beam_scaling()
passband_dict = external_data.get_passband_dict_dr6(arrays)

for ar in arrays:

    beam_path_cmb = f"{data_path}/beams/coadd_{ar}_night_beam_tform_jitter_cmb.txt"
    l, bl_cmb = pspy_utils.read_beam_file(beam_path_cmb, lmax=lmax)

    beam_path_mono = f"{data_path}/beams/coadd_{ar}_night_beam_tform_jitter_mono.txt"
    l, bl_mono = pspy_utils.read_beam_file(beam_path_mono, lmax=lmax)
    
    l, nu_array, bl_nu = beam_chromaticity.get_multifreq_beam(l,
                                                              bl_mono,
                                                              passband_dict[ar],
                                                              nu_ref_dict[ar],
                                                              alpha_dict[ar])
                                        


    cmap = mpl.pyplot.get_cmap('cool', len(nu_array)+1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(nu_array), vmax=np.max(nu_array)))
    #sm = plt.cm.ScalarMappable(cmap='coolwarm')


    fig, ax = plt.subplots(dpi=100, figsize=(12, 8))

    for i_nu, nu in enumerate(nu_array):
        ax.plot(l, bl_nu[:, i_nu], c=cmap(i_nu), alpha=0.2)
    
    ax.plot(l, bl_cmb, color="black", label="CMB beam")
    plt.legend(fontsize=18)
    plt.title(f"beam {ar}", fontsize=22)
    plt.xlabel(r"$\ell$", fontsize=22)
    plt.ylabel(r"$b_{\ell}(\nu)$", fontsize=22)

    plt.colorbar(sm, ax=plt.gca()).set_label(label=r"$\nu$ GHz",size=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/beam_{ar}.png", dpi = 300)
    plt.clf()
    plt.close()
