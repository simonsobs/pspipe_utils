"""
Some utility functions for handling systematics.
"""
import numpy as np
from pixell import curvedsky
from pspy import pspy_utils, so_cov

def leakage_correction(lth,
                       ps_dict_th,
                       gamma_alpha,
                       var_gamma_alpha,
                       lmax,
                       gamma_beta=None,
                       return_residual=False,
                       binning_file=None):
                       
    """
    We are applying the leakage model to theoretical cross spectra of the form
    X_alpha Y_beta, where X,Y in {T,E,B} and alpha and beta are two different detectors array
    gamma is defined such as
    {alm^E} = alm^E+ gamma_TE * alm^T
    where the curly bracket means including leakage
    see the available documentation in pspipe_utils
    
    Parameters
    ----------
    lth: array
        array of mutlipole corresponding to ps_dict_th
    ps_dict_th: dict
      dict containing the different theoretical power spectra
    gamma_alpha: dict with two key "TE", and "TB" each corresponding to a 1d array
        the expected leakage of the alpha array
    var_gamma_alpha: dict with three key "TETE", "TBTB", "TETB" each corresponding to a 1d array
        the variance of the leakage beam, if not corrected for, could include a bias
        not that the term is non zero only if alpha == beta, this is because we assume zero
        correlation between alpha and beta leakage measurement
    gamma_beta: dict with two key "TE", and "TB" each corresponding to a 1d array
        the expected leakage of the beta array
    lmax: integer
        max multipole to consider
    return_residual: boolean
        if True, only return the leakage model correction, otherwise return modified theory
    binning_file: str (optionnal)
        the name of the binning file you want to use to bin the spectra
    """
                         
    if gamma_beta is None: gamma_beta = gamma_alpha

    gE_a = gamma_alpha["TE"]
    gE_b = gamma_beta["TE"]
    gB_a = gamma_alpha["TB"]
    gB_b = gamma_beta["TB"]
    
    ps_dict_th_leak = {}
    
    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    for spec in spectra:
        X, Y = spec
        ps_dict_th[spec] = ps_dict_th[spec][:lmax]
        
        ps_dict_th_leak[spec] = ps_dict_th[spec].copy()
        ps_dict_th_leak[spec] +=  (so_cov.delta2(X,"E") * gE_a + so_cov.delta2(X,"B") * gB_a) *  ps_dict_th[f"T{Y}"]
        ps_dict_th_leak[spec] +=  (so_cov.delta2(Y,"E") * gE_b + so_cov.delta2(Y,"B") * gB_b) *  ps_dict_th[f"{X}T"]
        ps_dict_th_leak[spec] +=  (so_cov.delta2(X,"E") * gE_a + so_cov.delta2(X,"B") * gB_a) * (so_cov.delta2(Y,"E") * gE_b + so_cov.delta2(Y,"B") * gB_b) * ps_dict_th["TT"]
        if np.all(gE_a == gE_b) :
            ps_dict_th_leak[spec] += so_cov.delta2(X,"E") * so_cov.delta2(Y,"E") * var_gamma_alpha["TETE"] * ps_dict_th["TT"]
            ps_dict_th_leak[spec] += so_cov.delta2(X,"B") * so_cov.delta2(Y,"B") * var_gamma_alpha["TBTB"] * ps_dict_th["TT"]
            ps_dict_th_leak[spec] += so_cov.delta2(X,"B") * so_cov.delta2(Y,"E") * var_gamma_alpha["TETB"] * ps_dict_th["TT"]
            ps_dict_th_leak[spec] += so_cov.delta2(X,"E") * so_cov.delta2(Y,"B") * var_gamma_alpha["TETB"] * ps_dict_th["TT"]

        if return_residual:
            ps_dict_th_leak[spec] -= ps_dict_th[spec]
        if binning_file is not None:
            l, ps_dict_th_leak[spec] = pspy_utils.naive_binning(lth, ps_dict_th_leak[spec], binning_file, lmax)
    
    if binning_file is None: l = lth

    return l, ps_dict_th_leak

def apply_leakage_model_to_alm(alms, gamma_TE, gamma_TB):
    """
    take in alms and apply the leakage model
    the resulting alms are
    {alm_E} = alm_E + gamma_TE alm_T
    {alm_B} = alm_B + gamma_TB alm_T
    
    Parameters
    ----------
    alms : array of alm shape (3, alm.shape)
        the alms you want to apply the leakage mode lto
    gamma_TE : 1d array
        the T-> E leakage
    gamma_TB : 1d array
        the T-> B leakage
    """
    alms[1] = alms[1] + curvedsky.almxfl(alms[0], gamma_TE)
    alms[2] = alms[2] + curvedsky.almxfl(alms[0], gamma_TB)
    return alms

def read_leakage_model_old(leakage_file_dir, file_name, lmax, lmin=0, include_error_modes=True):
    """
    This routine serves to read the leakage model in the ACT format, both for
    the mean value of the leakage and its covariance.
    not that the error modes file is expected to be of the form
    l, err_modegE1, err_modegE2, err_modegE3, err_modegB1, err_modegE2, err_modegE3

    Parameters
    ----------
    leakage_file_dir : str
        location of the files describing the leakage
    file_name : str
        name of the specific file you want to load
        (e.g gamma_mp_uranus_pa4_f150.txt)
    lmin : integer
        minimum multipole to consider
    lmax : integer
        maximum multipole to consider
    """
    
    l, gamma_TE, gamma_TB, _, _ = np.loadtxt(f"{leakage_file_dir}/{file_name}", unpack=True)
    l = l[lmin: lmax]
    gamma_TE = gamma_TE[lmin: lmax]
    gamma_TB = gamma_TB[lmin: lmax]
    
    if include_error_modes == True:
        error_modes = np.loadtxt(f"{leakage_file_dir}/error_modes_{file_name}")
        error_modes_gTE = error_modes[lmin: lmax, 1:4]
        error_modes_gTB = error_modes[lmin: lmax, 4:7]
    else:
        error_modes_gTE = np.zeros((gamma_TE.shape, 3))
        error_modes_gTB = np.zeros((gamma_TE.shape, 3))
    
        
    return l, gamma_TE, error_modes_gTE,  gamma_TB, error_modes_gTB


def read_leakage_model(leakage_file_dir, file_name_TE, file_name_TB, lmax, lmin=0, include_error_modes=True):
    """
    This routine serves to read the leakage model in the ACT format, both for
    the mean value of the leakage and its covariance.
    not that the error modes file is expected to be of the form
    l, err_modegE1, err_modegE2, err_modegE3, err_modegB1, err_modegE2, err_modegE3

    Parameters
    ----------
    leakage_file_dir : str
        location of the files describing the leakage
    file_name_TE : str
        name of the  file  that contain gamma_TE and error_modes (e.g pa4_f150_gamma_t2e.txt)
    file_name_TB : str
        name of the  file  that contain gamma_TB and error_modes (e.g pa4_f150_gamma_t2b.txt)
    lmin : integer
        minimum multipole to consider
    lmax : integer
        maximum multipole to consider
    """
    
    def extract_beam_leakage_and_error_modes(file_name):
        data = np.loadtxt(f"{leakage_file_dir}/{file_name}")
        l, gamma = data[lmin: lmax, 0], data[lmin: lmax, 1]
        error_modes = data[lmin: lmax, 2:]
        return l, gamma, error_modes

    l, gamma_TE, error_modes_gTE = extract_beam_leakage_and_error_modes(file_name_TE)
    l, gamma_TB, error_modes_gTB = extract_beam_leakage_and_error_modes(file_name_TB)

    return l, gamma_TE, error_modes_gTE,  gamma_TB, error_modes_gTB




def error_modes_to_cov(error_modes):
    """
    Use the beam leakage error modes to reconstruct the beam leakage covarance matrix
    the format of the error modes is a matrix let's call it E (lmax, nmodes)
    the cov mat is reconstructed as M_{ij} = \sum^{n_modes}_{k=1} E_(ik}E_{jk}
    
    Parameters
    ----------
    error_modes: 2d array
        the error modes corresponding to the leakage measurement (lmax, nmodes)
    """
    
    return error_modes @ error_modes.T

def leakage_beam_sim(mean, error_modes):
    """
    Generate a realisation of beam leakage from a mean value and the error modes
    
    Parameters:
    ----------
    mean: 1d array
        the mean value of the leakage
    error_modes: 2d array
        the error modes corresponding to the leakage measurement (lmax, nmodes)

    """
    
    n_modes = error_modes.shape[1]
    sim = mean + error_modes @ np.random.randn(n_modes)
    return sim
