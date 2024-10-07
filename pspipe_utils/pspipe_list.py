"""
Some utility functions for building list for mpi.
"""
from itertools import combinations_with_replacement as cwr
from itertools import product
import numpy as np


def get_arrays_list(dict):
    """This function creates the lists over which mpi is done
    when we parallelized over each arrays

    Parameters
    ----------
    dict : dict
        the global dictionnary file used in pspipe
    """

    surveys = dict["surveys"]
    sv_list, ar_list = [], []
    n_arrays = 0
    for sv in surveys:
        arrays = dict[f"arrays_{sv}"]
        for ar in arrays:
            sv_list += [sv]
            ar_list += [ar]
            n_arrays += 1
    return n_arrays, sv_list, ar_list

def get_spectra_list(dict):
    """This function creates the lists over which mpi is done
    when we parallelized over each spectra

    Parameters
    ----------
    dict : dict
        the global dictionnary file used in pspipe

    """
    surveys = dict["surveys"]

    sv1_list, ar1_list, sv2_list, ar2_list = [], [], [], []
    n_spec = 0
    for id_sv1, sv1 in enumerate(surveys):
        arrays_1 = dict[f"arrays_{sv1}"]
        for id_ar1, ar1 in enumerate(arrays_1):
            for id_sv2, sv2 in enumerate(surveys):
                arrays_2 = dict[f"arrays_{sv2}"]
                for id_ar2, ar2 in enumerate(arrays_2):
                    # This ensures that we do not repeat redundant computations
                    if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                    if  (id_sv1 > id_sv2) : continue
                    sv1_list += [sv1]
                    ar1_list += [ar1]
                    sv2_list += [sv2]
                    ar2_list += [ar2]
                    n_spec += 1

    return n_spec, sv1_list, ar1_list, sv2_list, ar2_list

def get_covariances_list(dict, delimiter="&"):
    """This function creates the lists over which mpi is done
    when we parallelized over each covariance element

    Parameters
    ----------
    dict : dict
        the global dictionnary file used in pspipe

    """

    spec_name = get_spec_name_list(dict, delimiter)
    na_list, nb_list, nc_list, nd_list = [], [], [], []
    ncovs = 0

    for sid1, spec1 in enumerate(spec_name):
        for sid2, spec2 in enumerate(spec_name):
            if sid1 > sid2: continue
            na, nb = spec1.split("x")
            nc, nd = spec2.split("x")
            na_list += [na]
            nb_list += [nb]
            nc_list += [nc]
            nd_list += [nd]
            ncovs += 1

    return ncovs, na_list, nb_list, nc_list, nd_list

def get_spec_name_list(dict, delimiter="&", kind=None, freq_pair=None, remove_same_ar_and_sv=False, return_nu_tag=False):
    """This function creates a list with the name of all spectra we consider

    Parameters
    ----------
    dict : dict
        the global dictionnary file used in pspipe
    delimiter: str
        a character that separate the suvey and array name
    kind : str
        if "noise" or "auto" won't return
        a spectra with different survey1 and survey2
    freq_pair: list of two elements
        select only spectra with effective frequencies corresponding
        to the specified freq_pair
    same_ar_and_sv: boolean
        select only spectra from a same array and season
    return_nu_tag: boolean
        also return a list of frequency tags in the same order
    """

    surveys = dict["surveys"]
    spec_name_list = []
    nu_tag_list = []
    for id_sv1, sv1 in enumerate(surveys):
        arrays_1 = dict[f"arrays_{sv1}"]
        for id_ar1, ar1 in enumerate(arrays_1):
            for id_sv2, sv2 in enumerate(surveys):
                arrays_2 = dict[f"arrays_{sv2}"]
                for id_ar2, ar2 in enumerate(arrays_2):
                    # This ensures that we do not repeat redundant computations
                    if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                    if  (id_sv1 > id_sv2) : continue

                    if (kind == "noise") or (kind == "auto"):
                        if (sv1 != sv2): continue

                    nu_tag1 = dict[f"freq_info_{sv1}_{ar1}"]["freq_tag"]
                    nu_tag2 = dict[f"freq_info_{sv2}_{ar2}"]["freq_tag"]
                    c = 0

                    if freq_pair is not None:
                        f1, f2 = freq_pair
                        if (f1 != nu_tag1) or (f2 != nu_tag2): c +=1
                        if (f2 != nu_tag1) or (f1 != nu_tag2): c +=1
                    if c == 2: continue

                    if remove_same_ar_and_sv == True:
                        if (sv1 == sv2) & (ar1 == ar2): continue

                    spec_name_list += [f"{sv1}{delimiter}{ar1}x{sv2}{delimiter}{ar2}"]
                    nu_tag_list += [(nu_tag1, nu_tag2)]

    if return_nu_tag == False:
        return spec_name_list
    else:
        return spec_name_list, nu_tag_list

def get_freq_list(dict):
    """This function creates the list of all frequencies to consider

    Parameters
    ----------
    dict : dict
        the global dictionnary file used in pspipe
    """
    surveys = dict["surveys"]

    freq_list = []
    for sv in surveys:
        arrays = dict[f"arrays_{sv}"]
        for ar in arrays:
            freq_list += [dict[f"freq_info_{sv}_{ar}"]["freq_tag"]]

    # remove doublons
    freq_list = np.sort(list(dict.fromkeys(freq_list)))

    return freq_list


def x_ar_cov_order(spec_name_list,
                   nu_tag_list,
                   spectra_order = ["TT", "TE", "ET", "EE"]):

    """This function creates the list of spectra that enters
    the cross array covariance matrix.
    Note that ET, BT, and BE are removed for spectra of the type "dr6_pa4_f150xdr6_pa4_f150"
    where the are kept in the case "dr6_pa4_f150xdr6_pa5_f150", its because TE=ET in the former
    case

    Parameters
    ----------
    spec_name_list: list of str
        list of the cross spectra
    spectra_order: list of str
        the order of the spectra e.g  ["TT", "TE", "ET", "EE"]
    """
    x_ar_list = []
    for spec in spectra_order:
        for spec_name, nu_tag in zip(spec_name_list, nu_tag_list):
            na, nb = spec_name.split("x")
            if (spec == "ET" or spec == "BT" or spec == "BE") & (na == nb): continue
            x_ar_list += [[spec, spec_name, nu_tag]]

    return x_ar_list


def x_freq_cov_order(freq_list,
                     spectra_order = ["TT", "TE", "EE"]):


    """This function creates the list of spectra that enters
    the cross frequency covariance matrix.

    Parameters
    ----------
    freq_list: list of str
        the frequency we consider
    spectra_order: list of str
        the order of the spectra e.g  ["TT", "TE", "EE"]
    """
    x_freq_list = []

    for spec in spectra_order:
        if spec in ["ET", "BT", "BE"]:
            raise ValueError("spectra_order can not contain [ET, BT, BE] the cross freq cov matrix convention is to assign all ET, BT, BE into TE,TB,EB")

        if spec[0] == spec[1]:
            x_freq_list += [[spec, (f0, f1)] for f0, f1 in cwr(freq_list, 2)]
        else:
            x_freq_list +=  [[spec, (f0, f1)] for f0, f1 in product(freq_list, freq_list)]

    return x_freq_list

def final_cov_order(freq_list, spectra_order = ["TT", "TE", "EE"]):
    
    """This function creates the list of spectra that enters
    the final covariance matrix.

    Parameters
    ----------
    freq_list: list of str
        the frequency we consider
    spectra_order: list of str
        the order of the spectra e.g  ["TT", "TE", "EE"]
    """

    final_list = []
    for spec in spectra_order:
        if spec in ["ET", "BT", "BE"]:
            raise ValueError("spectra_order can not contain [ET, BT, BE] the final cov matrix convention is to assign all ET, BT, BE into TE, TB, EB")

        if spec == "TT":
            final_list += [[spec, (f0, f1)] for f0, f1 in cwr(freq_list, 2)]
        else:
            final_list += [[spec, None]]
            
    return  final_list


def get_map_set_list(d):
    """
    construct a list of all map data sets specified in the dictionnary
    a map set is for example: dr6_pa4_f150, planck_f143, etc
    
    Parameters
    ----------
    dict : dict
        the global dictionnary file used in pspipe
    """
    
    map_set_list = []
    for sv in d["surveys"]:
        for ar in d[f"arrays_{sv}"]:
            map_set_list.append(f"{sv}_{ar}")
    return map_set_list

def get_null_list(d, spectra, remove_TT_diff_freq=True):

    """
    construct a list of all valid null test between the different map data set specified in the dictionnary
    note that we exclude null test if they contains T at different frequency
        
    Parameters
    ----------
    dict : dict
        the global dictionnary file used in pspipe
    """
    
    map_set_list = get_map_set_list(d)
    null_list = []
    for i, (ms1, ms2) in enumerate(cwr(map_set_list, 2)):
        for j, (ms3, ms4) in enumerate(cwr(map_set_list, 2)):

            if j <= i: continue
            f1, f2 = d[f"freq_info_{ms1}"]["freq_tag"], d[f"freq_info_{ms2}"]["freq_tag"]
            f3, f4 = d[f"freq_info_{ms3}"]["freq_tag"], d[f"freq_info_{ms4}"]["freq_tag"]

            for m in spectra:
                m0, m1 = m[0], m[1]
                if remove_TT_diff_freq:
                    if (f1 != f3) and (m0 == "T"): continue
                    if (f2 != f4) and (m1 == "T"): continue
                null_list += [[m, ms1, ms2, ms3, ms4]]
                    
    return null_list
