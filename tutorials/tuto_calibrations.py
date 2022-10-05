"""
This script generates simulations for 2 surveys ("sv1", "sv2"),
introducing a miscalibration between the two surveys. Then, we compute
the calibration amplitudes using pspipe_utils.consistency for each sim
using different power spectra combinations.
"""
from pspy import pspy_utils, so_spectra, so_cov, so_dict
from pspipe_utils import consistency, pspipe_list
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import sys

# This class is used to prevent cobaya
# from printing details about MCMC steps
# for each of the sims
from io import StringIO
class NullIO(StringIO):
    def write(self, txt):
        pass

# Load global.dict
d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

# Directories
ps_dir = "result_simulation"
cov_dir = "result_covariances"
tuto_data_dir = "tuto_data"

# Output dir
output_dir = "result_calibs"
pspy_utils.create_directory(output_dir)

survey = d["surveys"][0] # Select the first survey
arrays = d[f"arrays_{survey}"][:2] # Select only the first two arrays
array_to_calibrate, ref_array = arrays
n_sims = d["n_sims"]

binning_file = f"{tuto_data_dir}/binning.dat"
lmax = d["lmax"]
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(lb)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]

##############################
# Calibration of the spectra #
##############################

print("\n===============")
print("= CALIBRATION =")
print("===============\n")

cal_amplitude = np.random.uniform(0.95, 1.05)

ps_order = [(f"{survey}&{array_to_calibrate}", f"{survey}&{array_to_calibrate}"),
            (f"{survey}&{array_to_calibrate}", f"{survey}&{ref_array}"),
            (f"{survey}&{ref_array}", f"{survey}&{ref_array}")]

# Calibration range
lmin_cal = 800
lmax_cal = 1300

# Get calibration amplitudes
proj_dict = {
             "C1": np.array([1, -1, 0]),
             "C2": np.array([0, -1, 1]),
             "C3": np.array([1, 0, -1])
            }

cal_dict_out = {f"C{i}": [] for i in range(1, 4)}

# Load analytic cov dict
an_cov_dict = {}
spec_name_list = pspipe_list.get_spec_name_list(d)

for i, spec1 in enumerate(spec_name_list):
    for j, spec2 in enumerate(spec_name_list):
        if i > j: continue
        an_cov = np.load(f"{cov_dir}/analytic_cov_{spec1}_{spec2}.npy")
        an_cov = so_cov.selectblock(an_cov, modes, n_bins, block = "TTTT")

        na, nb = spec1.split("x")
        nc, nd = spec2.split("x")

        an_cov_dict[(na, nb), (nc, nd)] = an_cov

for iii in range(n_sims):

    # Load ps dict & miscalibrate the array we want to calibrate
    ps_dict = {}
    for spec in spec_name_list:
        lb, ps = so_spectra.read_ps(f"{ps_dir}/Dl_{spec}_cross_{iii:05d}.dat", spectra = spectra)
        ps = ps["TT"]
        na, nb = spec.split("x")
        if array_to_calibrate in na:
            ps /= cal_amplitude
        if array_to_calibrate in nb:
            ps /= cal_amplitude

        ps_dict[na, nb] = ps

    # Concatenate power spectra
    ps_vec, full_cov = consistency.append_spectra_and_cov(ps_dict, an_cov_dict, ps_order)

    # Get the calibration range
    id = np.where((lb >= lmin_cal) & (lb <= lmax_cal))[0]

    print(f"Calibrating sim {iii} ...")
    for comb, proj_pattern in proj_dict.items():
        # Prevent cobaya from printing chains details for
        # each simulation
        sys.stdout = NullIO()
        cal_mean, cal_std = consistency.get_calibration_amplitudes(ps_vec, full_cov,
                                                                   proj_pattern, "TT",
                                                                   id, f"{output_dir}/chains/mcmc")
        sys.stdout = sys.__stdout__
        cal_dict_out[comb].append(cal_mean)


pickle.dump(cal_dict_out, open(f"{output_dir}/calibs_dict.pkl", "wb"))

nbins = 15
colors = {"C1": "tab:red", "C2": "tab:blue", "C3": "tab:green"}
plt.figure(figsize = (8, 6))
for comb in cal_dict_out:
    plt.hist(cal_dict_out[comb], density = True, bins = nbins,
             label = comb, histtype = "stepfilled",
             edgecolor = mpl.colors.to_rgba(colors[comb], alpha = 1),
             facecolor = mpl.colors.to_rgba(colors[comb], alpha = 0.3),
             linewidth = 1.65)
plt.xlabel("Calibration amplitude")
plt.yticks([])
plt.legend(frameon = False)
plt.axvline(cal_amplitude, color = "k")
plt.tight_layout()
plt.savefig(f"{output_dir}/calibration_hist.png", dpi = 300)
