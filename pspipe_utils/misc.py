"""
Some general utility functions that do not belong to other files.
"""
import numpy as np
from pspy import pspy_utils
from pixell import curvedsky

def str_replace(my_str, old, new):
    """
    just like replace but check that the replacement actually happened

    Parameters
    __________
    my_str: string
        the string in which the replacment will happen
    old: string
        old part of the string to be replaced
    new: string
        what will replace old
    """

    my_new_str = my_str.replace(old, new)
    if my_new_str == my_str:
        error = f" the name '{my_str}' does not contain '{old}' so I can't replace '{old}' by '{new}'"
        raise NameError(error)
    return my_new_str

def read_beams(f_name_beam_T, f_name_beam_pol, lmax=None):
    """
    read T and pol beams and return a beam dictionnary with entry T, E, B

    Parameters
    __________
    f_name_beam_T: string
        the filename of the temperature beam file
    f_name_beam_pol: string
        the filename of the polarisation beam file
    lmax : integer
        the maximum multipole to consider (note that usually beam file start at l=0)
    """

    bl = {}
    l, bl["T"] = pspy_utils.read_beam_file(f_name_beam_T, lmax=lmax)
    l, bl["E"] = pspy_utils.read_beam_file(f_name_beam_pol, lmax=lmax)
    bl["B"] = bl["E"]
    return l, bl

def prep_beams(fn, norm=None):
    """Read and normalize data from a beam file.

    Parameters
    ----------
    fn : path-like
        Beam file on disk.
    norm : str or scalar, optional
        Information to help scale the beam data, by default None. If 'mono',
        then normalize the beam and beam error modes by the beam monopole.
        Can also be an explicit scalar value.

    Returns
    -------
    (nl) np.ndarray, (nmode, nl) np.ndarray
        The beam and beam error modes.
    """
    beam = np.loadtxt(fn).T
    l, bl, bl_err = beam[0], beam[1], beam[2:]
    assert l[0] == 0, "the file is expected to start at l=0"
    
    if norm == 'mono':
        norm = bl[0]
    if norm is not None:
        bl /= norm
        bl_err /= norm
    
    return bl, bl_err

def apply_beams(alms, bl):
    """
    apply T and pol beams to alms

    Parameters
    __________
    alms: 2d array
        array of alms, alms[0]=alm_T, alms[1]=alm_E, alms[2]=alm_B
    bl: dict
        dictionnary containing T and E,B beams
    """
    for i, f in enumerate(["T", "E", "B"]):
        alms[i] = curvedsky.almxfl(alms[i], bl[f])
    return alms
