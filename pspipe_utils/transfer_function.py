"""
Some utility functions for additional transfer function.
"""
from pspy import so_spectra, pspy_utils
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from getdist.mcsamples import loadMCSamples
from cobaya.run import run



def deconvolve_xtra_tf(lb, ps, spectra, xtra_pw1=None, xtra_pw2=None, mm_tf1=None, mm_tf2=None):
    """
    this function deconvolve an xtra transfer function (beyond the kspace filter one)
    it included possibly pixwin and map maker transfer function
    Parameters
    ----------
    lb: 1d array
        the multipoles
    ps:  dict of 1d array
        the power spectra
    spectra: list of string
        needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    xtra_pw1: 1d array
        an extra pixel window due to healpix or planck projected on car
    xtra_pw2: 1d array
        an extra pixel window due to healpix or planck projected on car
    mm_tf1: dict of 1d array
        a map maker transfer function
    mm_tf2: dict of 1d array
        a map maker transfer function

    """

    for spec in spectra:
        if xtra_pw1 is not None:
            ps[spec] /= xtra_pw1
        if xtra_pw2 is not None:
            ps[spec] /= xtra_pw2
        if mm_tf1 is not None:
            ps[spec] /= mm_tf1[spec]
        if mm_tf2 is not None:
            ps[spec] /= mm_tf2[spec]
    return lb, ps

def plot_tf(lb_list, tf_list, tf_err_list, titles, plot_file, ell_list = None, tf_model_list = None):

    n = len(tf_list)
    if n > 3:
        fig, axes = plt.subplots(2, 3, sharey = True, figsize = (16, 9))
    else:
        fig, axes = plt.subplots(1, n, sharey=True, figsize = (16, 5))


    for i in range(n):
        if n > 3:
            ax = axes[i//3, i%3]
            if i % 3 == 0:
                ax.set_ylabel(r"$F_\ell^T$")
            if i // 3 == 1:
                ax.set_xlabel(r"$\ell$")
        else:
            ax = axes[i]
            if i == 0:
                ax.set_ylabel(r"$F_\ell^T$")
            ax.set_xlabel(r"$\ell$")

        ax.axhline(1, color = "k", ls = "--", lw = 0.8)
        if tf_model_list is not None:
            ax.plot(ell_list[i], tf_model_list[i], color = "gray")
        ax.errorbar(lb_list[i], tf_list[i], yerr = tf_err_list[i], marker = ".",
                    capsize = 1, elinewidth = 1.1, ls = "None", color = "tab:red")
        ax.set_title(titles[i])
        ax.set_ylim(0, 1.3)
        ax.set_xlabel(r"$\ell$")

    plt.tight_layout()
    plt.savefig(plot_file, dpi = 300)

def tf_model(ell, aa, bb, cc, method = "logistic"):

    if method == "sigurd":
        x = 1 / (1 + (ell / bb) ** aa)
        tf = x / (x + cc)

    if method == "thib":
        tf = aa + (1 - aa) * np.sin(np.pi / 2 * (ell - bb) / (cc - bb)) ** 2
        tf[ell > cc] = 1

    if method == "beta":
        tf = np.zeros(len(lb))
        id = np.where(ell < bb)
        tf[id] = aa + (1-aa) / (1 + (ell[id] / (bb - ell[id])) ** (-cc))
        tf[ell >= bb] = 1

    if method == "logistic":
        tf = aa / (1 + bb * np.exp(-cc * ell))

    return tf

def fit_tf(lb, tf_est, tf_cov, prior_dict, chain_name, method = "logistic"):

    def loglike(aa, bb, cc):
        res = tf_est - tf_model(lb, aa, bb, cc, method = method)
        chi2 = res @ np.linalg.inv(tf_cov) @ res
        return -0.5 * chi2

    info = {"likelihood": {
                "my_like": loglike},
            "params": {
                "aa": {
                    "prior": prior_dict["aa", method],
                    "latex": "aa"},
                "bb": {
                    "prior": prior_dict["bb", method],
                    "latex": "bb"},
                "cc": {
                    "prior": prior_dict["cc", method],
                    "latex": "cc"},},
            "sampler": {
                "mcmc": {
                    "max_tries": 1e8,
                    "Rminus1_stop": 0.005,
                    #"Rminus1_cl_stop": 0.03
                     }},
            "output": chain_name,
            "force": True
            }
    updated_info, sampler = run(info)

def get_parameter_mean_and_std(chain_name, pars):

    s = loadMCSamples(chain_name, settings = {"ignore_rows": 0.5})

    mean = s.mean(pars)
    cov = s.cov(pars)

    return mean, np.sqrt(cov.diagonal())

def get_tf_bestfit(ell, chain_name, method = "logistic"):

    mu, _ = get_parameter_mean_and_std(chain_name, ["aa", "bb", "cc"])
    aa, bb, cc = mu

    tf = tf_model(ell, aa, bb, cc, method = method)

    return tf
