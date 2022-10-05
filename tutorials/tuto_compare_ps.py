"""

"""
from pspy import pspy_utils, so_dict
from pspipe_utils import consistency
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys

# Load global.dict
d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
n_sims = d["n_sims"]

# Directories
ps_dir = "result_simulation"
cov_dir = "result_covariances"
tuto_data_dir = "tuto_data"

# Output dir
output_dir = "result_compare"
pspy_utils.create_directory(output_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]

# Define the different operations we want to perform
op_dict = {"ratio": "aa/bb",
           "diff": "aa-bb",
           "map_diff": "aa+bb-2ab"}

# Define the arrays we want to compare
ar_list = ["dr6&ar1", "dr6&ar2"]

res_ps_dict = {op: [] for op in op_dict}
res_cov_dict = {op: [] for op in op_dict}

for iii in range(n_sims):
    ps_template = f"{ps_dir}/Dl_" + "{}x{}" + f"_cross_{iii:05d}.dat"
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
    print(f"X² = {chi2}/{len(lb)}")
    ax.errorbar(x_plot, y_plot, yerr_plot, ls = "None", marker = "o", label = op)
    ax.legend()
plt.tight_layout()
plt.show()
