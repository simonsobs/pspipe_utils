"""
Some utility functions for evaluating the scaling of the beams with frequency
"""
from pspy import pspy_utils
import numpy as np
from scipy.interpolate import interp1d

def act_dr6_beam_scaling():
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
    Model for monochromatic b_ell: (B x f)_{ell x (freq / freq_ref)^{-alpha / 2}}

    Parameters
    ----------
    b_ell_template : (nell) array
        Template for B_ell, should be 1 at ell=0.
    f_ell : (nell) array
        Multiplicate correction to the B_ell template. Should be 1 at ell=0.
    freqs : (nfreq) array
        Frequencies.
    freq_ref : float
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
    
def get_multifreq_beam(l, bl, passband, alpha, nu_ref):
    nu_array = passband[0]
    bl_nu = get_mono_b_ell(l, bl, nu_array, nu_ref, alpha)
    return l, nu_array, bl_nu
    
