"""
Some utility functions for the generation of simulations.
"""
import numpy as np
from pixell import curvedsky
from pspy import so_spectra


def cmb_matrix_from_file(f_name, lmax, spectra, input_type="Dl"):
    """This function read the cmb power spectra from disk and return a
     [3, 3, lmax] matrix with the cmb power spectra.

    Parameters
    ----------
    f_name : string
        the file_name of the power spectra
    lmax: integer
        the maximum multipole for the cmb power spectra

    """
    l, ps_theory = so_spectra.read_ps(f_name, spectra=spectra)
    assert l[0] == 2, "the file is expected to start at l=2"

    ps_mat = np.zeros((3, 3, lmax))
    for i, f1 in enumerate("TEB"):
        for j, f2 in enumerate("TEB"):
            if input_type == "Dl":
                ps_theory[f1+f2] *= 2 * np.pi / (l * (l + 1))
            ps_mat[i, j][2:lmax] = ps_theory[f1+f2][:lmax-2]

    return ps_mat


def noise_matrix_from_files(f_name_tmp, survey, arrays, lmax, nsplits, spectra, input_type="Dl"):

    """This function uses the measured noise power spectra from disk
    and generate a three dimensional array of noise power spectra [3 * n_arrays, 3 * n_arrays, lmax]
    The measured noise spectra is supposed to be the "mean noise" so to get the split noise we have to multiply by nsplits
    Note the the function return noise matrix in "Cl", so apply an extra correction if the input is "Dl"

    Parameters
    ----------
    f_name_tmp : string
        a template name of the noise power spectra
    survey : string
        the survey to consider
    arrays: list of string
        the arrays we consider
    lmax: integer
        the maximum multipole for the noise power spectra
    n_splits: integer
        the number of data splits we want to simulate
        nl_per_split= nl * n_{splits}
    input_type: str
        "Cl" or "Dl"

    """

    n_arrays = len(arrays)
    nl_array = np.zeros((3 * n_arrays, 3 * n_arrays, lmax))

    for c1, ar1 in enumerate(arrays):
        for c2, ar2 in enumerate(arrays):
            l, nl_dict = so_spectra.read_ps(f_name_tmp.format(ar1, ar2, survey), spectra=spectra)
            assert l[0] == 2, "the file is expected to start at l=2"

            for s1, field1 in enumerate("TEB"):
                for s2, field2 in enumerate("TEB"):
                    if input_type == "Dl":
                        nl_dict[field1 + field2] *=  2 * np.pi / (l * (l + 1))

                    nl_array[c1 + n_arrays * s1, c2 + n_arrays * s2, 2:lmax] = nl_dict[field1 + field2][:lmax-2] * nsplits

    return l, nl_array


def foreground_matrix_from_files(f_name_tmp, arrays_list, lmax, spectra, input_type="Dl"):

    """This function read the best fit foreground power spectra from disk
    and generate a three dimensional array of foregroung power spectra [3 * nfreqs, 3 * nfreqs, lmax].
    The file on disk are expected to start at l=2 while the matrix will start at l=0

    Parameters
    ----------
    f_name_tmp : string
      a template name of the fg power spectra
    arrays_list: 1d array of string
      the arrays we consider
    lmax: integer
      the maximum multipole for the foreground power spectra
    spectra: list of strings
      the arrangement of the spectra for example:
      ['TT','TE','TB','ET','BT','EE','EB','BE','BB']
    input_type: str
        "Cl" or "Dl"

    """

    narrays = len(arrays_list)
    fl_array = np.zeros((3 * narrays, 3 * narrays, lmax))

    for c1, array1 in enumerate(arrays_list):
        for c2, array2 in enumerate(arrays_list):
            l, fl_dict = so_spectra.read_ps(f_name_tmp.format(array1, array2), spectra=spectra)
            assert l[0] == 2, "the file is expected to start at l=2"

            for s1, field1 in enumerate("TEB"):
                for s2, field2 in enumerate("TEB"):
                    if input_type == "Dl":
                        fl_dict[field1 + field2] *=  2 * np.pi / (l * (l + 1))

                    fl_array[c1 + narrays * s1, c2 + narrays * s2, 2:lmax] = fl_dict[field1 + field2][:lmax-2]

    return l, fl_array


def generate_fg_alms(fg_mat, arrays_list, lmax, dtype="complex64"):
    """
    This function generate the alms corresponding to a fg matrix
    the alms are returned in the form of a dict with key "freq"
    the alms are in the format [T,E,B]

    Parameters
    ----------
    fg_mat : 2d array
      the fg matrix of size [3 * nfreqs, 3 * nfreqs, lmax]
    arrays_list: 1d array of string
      the arrays we consider
    lmax: integer
      the maximum multipole for the noise power spectra
    dtype: str
      the datatype of the alms (e.g complex64)
    """

    narrays = len(arrays_list)

    fglms_all = curvedsky.rand_alm(fg_mat, lmax=lmax, dtype=dtype)
    fglm_dict = {}
    for i, array in enumerate(arrays_list):
        fglm_dict[array] = [fglms_all[i + k * narrays] for k in range(3)]

    return fglm_dict

def generate_noise_alms(noise_mat, array_list, lmax, dtype="complex64"):
    """
    This function generate the alms corresponding to a noise matrix
    the alms are returned in the form of a dict with key "array"
    the alms are in the format [T,E,B]

    Parameters
    ----------
    noise_mat : 2d array
        the noise matrix of size [3 * narrays, 3 * narrays, lmax]
    array_list: 1d array of string
        the frequencies we consider
    lmax: integer
        the maximum multipole for the noise power spectra
    dtype: str
        the datatype of the alms (e.g complex64)
    """

    narrays = len(array_list)
    nlms_all = curvedsky.rand_alm(noise_mat, lmax=lmax, dtype=dtype)
    nlm_dict = {}
    for i, array in enumerate(array_list):
        nlm_dict[array] = [nlms_all[i + k * narrays] for k in range(3)]

    return nlm_dict
