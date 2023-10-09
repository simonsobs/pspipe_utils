"""
Some utility functions for handling rotation of the polarisation angle.
"""
import numpy as np
from pixell import curvedsky
from pspy import pspy_utils, so_cov
from copy import deepcopy

def rot_theory_spectrum(lth, psth, phi_alpha, phi_beta):
    """
    propagate the effect of a polarisation angle rotation into the expected power spectra
    this consider the cross power spectrum between two different arrays alpha and beta
    (see doc)
    a^{E}_{alpha} = a^{E}_{alpha} cos(2 phi_alpha) - a^{B} sin(2 phi_alpha)
    a^{B}_{alpha} = a^{E}_{alpha} sin(2 phi_alpha) + a^{B} cos(2 phi_alpha)
    
    Parameters
    ----------
    lth: array
        array of mutlipole corresponding to ps_dict_th
    psth: dict
      dict containing the different theoretical power spectra
    phi_alpha: float
      the rotation of the polarisation for the alpha array (in degree)
    phi_beta: float
      the rotation of the polarisation for the beta array (in degree)
      
    """
    psth_rot = deepcopy(psth)
    
    deg_to_rad = np.pi / 180
    phi_alpha *= deg_to_rad
    phi_beta *= deg_to_rad
    
    psth_rot["TE"] = psth["TE"] * np.cos(2 * phi_beta) - psth["TB"] * np.sin(2 * phi_beta)
    psth_rot["ET"] = psth["TE"] * np.cos(2 * phi_alpha) - psth["TB"] * np.sin(2 * phi_alpha)

    psth_rot["TB"] = psth["TE"] * np.sin(2 * phi_beta) + psth["TB"] * np.cos(2 * phi_beta)
    psth_rot["BT"] = psth["TE"] * np.sin(2 * phi_alpha) + psth["TB"] * np.cos(2 * phi_alpha)

    psth_rot["EE"] = psth["EE"] * np.cos(2 * phi_alpha) * np.cos(2 * phi_beta)
    psth_rot["EE"] += psth["BB"] * np.sin(2 * phi_alpha) * np.sin(2 * phi_beta)
    psth_rot["EE"] -= psth["EB"] * np.sin(2 * (phi_alpha + phi_beta))

    psth_rot["BB"] = psth["EE"] * np.sin(2 * phi_alpha) * np.sin(2 * phi_beta)
    psth_rot["BB"] += psth["BB"] * np.cos(2 * phi_alpha) * np.cos(2 * phi_beta)
    psth_rot["BB"] += psth["EB"] * np.sin(2 * (phi_alpha + phi_beta))

    psth_rot["EB"] = psth["EE"] * np.cos(2 * phi_alpha) * np.sin(2 * phi_beta)
    psth_rot["EB"] -= psth["BB"] * np.sin(2 * phi_alpha) * np.cos(2 * phi_beta)
    psth_rot["EB"] += psth["EB"] * np.cos(2 * (phi_alpha + phi_beta))

    psth_rot["BE"] = psth["EE"] * np.cos(2 * phi_beta) * np.sin(2 * phi_alpha)
    psth_rot["BE"] -= psth["BB"] * np.sin(2 * phi_beta) * np.cos(2 * phi_alpha)
    psth_rot["BE"] += psth["EB"] * np.cos(2 * (phi_alpha + phi_beta))


    return lth, psth_rot
    
def rot_alms(alms, phi):
    """
    Effect on the alms due to a rotation of the polarisation angle by an angle phi
    
    Parameters
    ----------
    alms : array of alm shape (3, alm.shape)
        the alms you want to apply the leakage mode lto
    phi : float
        the rotation of polarisation  (in degree)

    """
    
    deg_to_rad = np.pi / 180
    phi *= deg_to_rad

    alms_rot = alms.copy()
    alms_rot[1] = alms[1] * np.cos(2 * phi) - alms[2] * np.sin(2 * phi)
    alms_rot[2] = alms[1] * np.sin(2 * phi) + alms[2] * np.cos(2 * phi)
    
    return alms_rot
