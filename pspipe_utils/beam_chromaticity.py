"""
Some utility functions for evaluating the scaling of the beams with frequency
Work from Adri Duivenvoorden and Serena Giardiello
"""
from pspy import pspy_utils
import numpy as np
from scipy.interpolate import interp1d

def act_dr6_beam_scaling():
    """
    the scaling of the dr6 beams
    """
    alpha_dict, nu_ref_dict = {}, {}
    alpha_dict["pa4_f150"], nu_ref_dict["pa4_f150"] = 1.66, 148.47
    alpha_dict["pa4_f220"], nu_ref_dict["pa4_f220"] = 1.13, 226.73
    alpha_dict["pa5_f090"], nu_ref_dict["pa5_f090"] = 1.81, 96.54
    alpha_dict["pa5_f150"], nu_ref_dict["pa5_f150"] = 1.60, 149.31
    alpha_dict["pa6_f090"], nu_ref_dict["pa6_f090"] = 1.59, 95.33
    alpha_dict["pa6_f150"], nu_ref_dict["pa6_f150"] = 1.59, 147.90
    return alpha_dict, nu_ref_dict

def get_mono_b_ell(ells, b_ell_template, nu_array, nu_ref, alpha):
    """
    Model the frequency dependent beam b_ell: (B)_{ell x (freq / freq_ref)^{-alpha / 2}}

    Parameters
    ----------
    ells: (nell) array
        The multipoles
    b_ell_template : (nell) array
        Template for B_ell, should be 1 at ell=0.
    nu_array : (nfreq) array
        Frequencies.
    nu_ref : float
        Reference frequency.
    alpha : float
        Power law index.
    Returns
    -------
    b_ell : (nell, nfreq) array
        B_ell array at each input frequency.
    """
    fi = interp1d(ells, b_ell_template, kind="linear", fill_value="extrapolate")
    out = fi(ells[:,np.newaxis] * (nu_array / nu_ref) ** (- alpha / 2))
    # if unphysical, we set these to zero.
    out[out < 0] = 0
    return out
    
def get_multifreq_beam(ells, b_ell_template, passband, nu_ref, alpha):
    """
    take a monochromatic beam template and scale it for the frequency of the considered bandpass
    Parameters
    ----------
    ells: (nell) array
        The multipoles
    b_ell_template : (nell) array
        Template for B_ell, should be 1 at ell=0.
    passband : dict
        a dictionnary with bandpass info
    nu_ref : float
        Reference frequency.
    alpha : float
        Power law index.
    """
    nu_array = passband[0]
    bl_nu = get_mono_b_ell(ells, b_ell_template, nu_array, nu_ref, alpha)
    return ells, nu_array, bl_nu
    
