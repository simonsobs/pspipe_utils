"""
Some utility functions for building list for mpi.
"""
from itertools import combinations_with_replacement as cwr
from itertools import permutations, product
import os
import re
import numpy as np
import yaml

def get_windownames_list(dict):
    """Get the unique window names from the paramdict."""
    windownames = []
    windowkey_pattern = 'window_(T|pol|kspace)'
    windowname_pattern = 'window_(.*)_(kspace|baseline)'
    for k, v in dict.items():
        if re.search(windowkey_pattern, k):
            windowname_text = os.path.splitext(os.path.basename(v))[0]
            match_val = re.search(windowname_pattern, windowname_text)
            if match_val is None:
                raise ValueError(f"paramfile key {k} matches 'window_(T|pol|kspace)' but value {v} does not match 'window_(.*)_(kspace|baseline)'")
            windowname = match_val.group(1)
            if windowname not in windownames:
                windownames.append(windowname)
    return len(windownames), windownames

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

# to get around delimiters...
def get_sv_and_m_from_sv_m(sv_m, sv_m_list, sv_list, m_list):
    idx = sv_m_list.index(sv_m)
    return sv_list[idx], m_list[idx]

# given a dictionary of rules, test if a pair of two dicts of tags passes
def eval_mpair_rules(t1, t2, rules):
    if not rules:
        return True

    # 1. Require specific tags (strict list pools expected)
    for tag_key, (pool1, pool2) in rules.get('require_tags', {}).items():
        v1 = t1.get(tag_key)
        v2 = t2.get(tag_key)
        
        # Check standard and flipped orientations using the pools
        match_standard = (v1 in pool1) and (v2 in pool2)
        match_flipped  = (v1 in pool2) and (v2 in pool1)
        
        if not (match_standard or match_flipped):
            return False

    # 2. Dynamic matching (must have the same value for these tags)
    if 'match_any_tag' in rules:
        if not any(t1.get(key) == t2.get(key) for key in rules['match_any_tag']):
            return False
            
    if 'match_all_tags' in rules:
        if not all(t1.get(key) == t2.get(key) for key in rules['match_all_tags']):
            return False

    return True

def get_spec2nullgroup2nullflag_mpairs(d, return_spectra_list=False):
    with open(d['spec2nullgroup2nullflag_mpairs_yaml'], 'r') as file:
        inp = yaml.safe_load(file)

    _, _sv_list, _m_list = get_arrays_list(d)
    _sv_m_list = ['_'.join(_sv_m) for _sv_m in zip(_sv_list, _m_list)] # assume delimiter = '_' to match against tags dict
    _full_mpairs_list = list(cwr(_sv_m_list, r=2))
    _full_spec_name_list = [f'{m1}x{m2}' for m1, m2 in _full_mpairs_list]
    _full_mpairs_list_with_reversed = list(product(_sv_m_list, repeat=2))

    spec2nullgroup2nullflag_mpairs = {}
    _spec_name_list = []
    for spec, nullgroup2nullflag_mpairrules in inp.items():
        if spec[0] == spec[1]:
            mpairs_iter = _full_mpairs_list
        else:
            mpairs_iter = _full_mpairs_list_with_reversed

        spec2nullgroup2nullflag_mpairs[spec] = {}
        for nullgroup, nullflag_mpairrules in nullgroup2nullflag_mpairrules.items():
            spec2nullgroup2nullflag_mpairs[spec][nullgroup] = [nullflag_mpairrules['nullflag']]
        
            mpairrules = nullflag_mpairrules.get('mpair_rules', {})
            mpairs = [f'{m1}x{m2}' for m1, m2 in mpairs_iter if eval_mpair_rules(d[f'tags_{m1}'], d[f'tags_{m2}'], mpairrules)]
            assert len(mpairs) == len(np.unique(mpairs)), \
                f'mpairs from spec2nullgroup2nullflag_mpairs_yaml, {spec} not unique'
            spec2nullgroup2nullflag_mpairs[spec][nullgroup].append(mpairs)
            
            for mpair in mpairs:
                for _mpair in (mpair, 'x'.join(mpair.split('x')[::-1])):
                    if _mpair in _full_spec_name_list and _mpair not in _spec_name_list:
                        _spec_name_list.append(_mpair)

    # from _spec_name_list to spectra list. 
    # do this because "spectra_list" is more fundamental even though spec_name_list is used as
    # an intermediary above because it's easier to test membership in
    n_spec = 0
    sv1_list, ar1_list, sv2_list, ar2_list = [], [], [], []
    for _spec_name in _spec_name_list:
        n1, n2 = _spec_name.split('x')
        sv1, ar1 = get_sv_and_m_from_sv_m(n1, _sv_m_list, _sv_list, _m_list)
        sv2, ar2 = get_sv_and_m_from_sv_m(n2, _sv_m_list, _sv_list, _m_list)
        sv1_list += [sv1]
        ar1_list += [ar1]
        sv2_list += [sv2]
        ar2_list += [ar2]
        n_spec += 1

    if return_spectra_list:
        return spec2nullgroup2nullflag_mpairs, (n_spec, sv1_list, ar1_list, sv2_list, ar2_list)
    else:
        return spec2nullgroup2nullflag_mpairs

def get_spectra_list(dict, from_spec_nullgroups=False):
    """This function creates the lists over which mpi is done
    when we parallelized over each spectra.
    
    Parameters
    ----------
    dict : dict
        the global dictionnary file used in pspipe

    """
    if from_spec_nullgroups:
        _, (n_spec, sv1_list, ar1_list, sv2_list, ar2_list) = get_spec2nullgroup2nullflag_mpairs(dict, return_spectra_list=True)

    else:
        surveys = dict["surveys"]

        sv1_list, ar1_list, sv2_list, ar2_list = [], [], [], []
        n_spec = 0
        for id_sv1, sv1 in enumerate(surveys):
            arrays_1 = dict[f"arrays_{sv1}"]
            for id_ar1, ar1 in enumerate(arrays_1):
                for id_sv2, sv2 in enumerate(surveys):
                    arrays_2 = dict[f"arrays_{sv2}"]
                    for id_ar2, ar2 in enumerate(arrays_2):
                        if  (id_sv1 > id_sv2) : continue
                        if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                        sv1_list += [sv1]
                        ar1_list += [ar1]
                        sv2_list += [sv2]
                        ar2_list += [ar2]
                        n_spec += 1

    return n_spec, sv1_list, ar1_list, sv2_list, ar2_list

def get_covariances_list(dict, delimiter="&", from_spec_nullgroups=False):
    """This function creates the lists over which mpi is done
    when we parallelized over each covariance element

    Parameters
    ----------
    dict : dict
        the global dictionnary file used in pspipe

    """

    spec_name = get_spec_name_list(dict, delimiter=delimiter, from_spec_nullgroups=from_spec_nullgroups)
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

def get_spec_name_list(dict, delimiter="&", from_spec_nullgroups=False):
    """This function creates a list with the name of all spectra we consider

    Parameters
    ----------
    dict : dict
        the global dictionnary file used in pspipe
    delimiter: str
        a character that separate the suvey and array name
    """

    spec_name_list = []
    n_spec, sv1_list, ar1_list, sv2_list, ar2_list = get_spectra_list(dict, from_spec_nullgroups=from_spec_nullgroups)
    for sv1, ar1, sv2, ar2 in zip(sv1_list, ar1_list, sv2_list, ar2_list):
        spec_name_list += [f"{sv1}{delimiter}{ar1}x{sv2}{delimiter}{ar2}"]

    return spec_name_list

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

def get_splits_auto_iterator(svi, nspliti, svj, nsplitj):
    """List of the split pairs that enter the auto-spectrum of a given
    spectrum measurement. If the surveys are the same, then the pairs are like
    (0, 0), (1, 1) ... (n, n). There is no auto-spectrum if the surveys are not
    the same.

    Parameters
    ----------
    svi : str
        Identifier of first survey.
    nspliti : int
        Number of splits in first survey.
    svj : str
        Identifier of second survey.
    nsplitj : int
        Number of splits in second survey.

    Returns
    -------
    list
        Pairs of splits that enter the auto-spectrum.
    """
    if svi == svj:
        assert nspliti == nsplitj, \
            f'{svi=} and {svj=} are equal but {nspliti=} and {nsplitj=} are not'
        split_ij_iterator = list(zip(range(nspliti), range(nspliti)))
    else:
        split_ij_iterator = []

    return split_ij_iterator

def get_splits_cross_iterator(svi, nspliti, svj, nsplitj):
    """List of the split pairs that enter the cross-spectrum of a given
    spectrum measurement. If surveys are the same, this skips pairs that would
    enter the autos-spectrum. If surveys are different, this is all pairs.

    Parameters
    ----------
    svi : str
        Identifier of first survey.
    nspliti : int
        Number of splits in first survey.
    svj : str
        Identifier of second survey.
    nsplitj : int
        Number of splits in second survey.

    Returns
    -------
    list
        Pairs of splits that enter the cross-spectrum.
    """
    if svi == svj:
        assert nspliti == nsplitj, \
            f'{svi=} and {svj=} are equal but {nspliti=} and {nsplitj=} are not'
        split_ij_iterator = list(permutations(range(nspliti), r=2))
    else:
        split_ij_iterator = list(product(range(nspliti), range(nsplitj)))

    return split_ij_iterator

def canonize_connected_2pt(leg1, leg2):
    """Re-order leg1 and leg2 to be in a canonical order with respect to a 
    connected two-point function of the legs: sorted((leg1, leg2)).

    Parameters
    ----------
    leg1 : any
        A label for the first leg. Must support comparison operations.
    leg2 : any
        A label for the second leg. Must support comparison operations.

    Returns
    -------
    any, any
        canonical_leg_1, canonical_leg_2
    """
    # we dont need to see all legs because ordering is global
    leg1, leg2 = sorted((leg1, leg2)) 
    
    return leg1, leg2

def canonize_disconnected_4pt(leg1, leg2, leg3, leg4):
    """Re-order the legs to be in a canonical order with respect to a 
    disconnected four-point function of the legs:
    sorted((sorted((leg1, leg2)), sorted((leg3, leg4)))).

    Parameters
    ----------
    leg1 : any
        A label for the first leg. Must support comparison operations.
    leg2 : any
        A label for the second leg. Must support comparison operations.
    leg3 : any
        A label for the third leg. Must support comparison operations.
    leg4 : any
        A label for the fourth leg. Must support comparison operations.

    Returns
    -------
    any, any, any, any
        canonical_leg_1, canonical_leg_2, canonical_leg_3, canonical_leg_4
    """
    canonical_pair_1 = canonize_connected_2pt(leg1, leg2)
    canonical_pair_2 = canonize_connected_2pt(leg3, leg4)

    # we dont need to see all legs because ordering is global
    (leg1, leg2), (leg3, leg4) = sorted((canonical_pair_1, canonical_pair_2))
    
    return leg1, leg2, leg3, leg4

def canonize_connected_4pt(leg1, leg2, leg3, leg4):
    """Re-order the legs to be in a canonical order with respect to a 
    connected four-point function of the legs:
    sorted((leg1, leg2, leg3, leg4)).

    Parameters
    ----------
    leg1 : any
        A label for the first leg. Must support comparison operations.
    leg2 : any
        A label for the second leg. Must support comparison operations.
    leg3 : any
        A label for the third leg. Must support comparison operations.
    leg4 : any
        A label for the fourth leg. Must support comparison operations.

    Returns
    -------
    any, any, any, any
        canonical_leg_1, canonical_leg_2, canonical_leg_3, canonical_leg_4
    """
    # we dont need to see all legs because ordering is global
    leg1, leg2, leg3, leg4 = sorted((leg1, leg2, leg3, leg4))
    
    return leg1, leg2, leg3, leg4