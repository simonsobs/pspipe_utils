"""
Some utility functions for additional transfer function.
"""
from pspy import so_spectra, pspy_utils
import numpy as np
import healpy as hp




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
