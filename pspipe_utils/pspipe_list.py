"""
Some utility functions for building list for mpi.
"""

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

def get_covariances_list(dict):
    """This function creates the lists over which mpi is done
    when we parallelized over each covariance element
    
    Parameters
    ----------
    dict : dict
        the global dictionnary file used in pspipe

    """

    spec_name = get_spec_name_list(dict)
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

def get_spec_name_list(dict, char="&", kind=None, freq_pair=None):
    """This function creates a list with the name of all spectra we consider
     
    Parameters
    ----------
    dict : dict
        the global dictionnary file used in pspipe
    char: str
        a character that separate the suvey and array name
    """

    surveys = dict["surveys"]
    spec_name = []
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

                    c = 0

                    if freq_pair is not None:
                        f1, f2 = freq_pair
                        nu_eff1 = dict[f"nu_eff_{sv1}_{ar1}"]
                        nu_eff2 = dict[f"nu_eff_{sv2}_{ar2}"]
                        if (f1 != nu_eff1) or (f2 != nu_eff2): c +=1
                        if (f2 != nu_eff1) or (f1 != nu_eff2): c +=1
                    if c == 2: continue

                    spec_name += [f"{sv1}{char}{ar1}x{sv2}{char}{ar2}" ]

    return spec_name

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
        arrays = dict["arrays_%s" % sv]
        for ar in arrays:
            freq_list += [dict["nu_eff_%s_%s" % (sv, ar)]]

    # remove doublons
    freq_list = list(dict.fromkeys(freq_list))
    
    return freq_list
