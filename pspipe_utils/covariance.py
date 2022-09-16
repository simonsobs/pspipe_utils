import numpy as np
import pylab as plt
from pspy import so_cov
from pspy import pspy_utils


def read_cov_block_and_build_dict(spec_name_list,
                                  cov_dir,
                                  cov_type,
                                  spectra_order = ["TT", "TE", "ET", "EE"]):
                                  
    """
    Read the different block covariances corresponding to spec_name_list
    and build a dictionnary with the different elements
    
    Parameters
    ----------
    spec_name_list: list of str
        list of the cross spectra
    cov_dir: str
        path to the folder with the covariance matrices block
    cov_type: str
        `cov_type` is used to specify which
         kind of covariance will be read ("analytic", "mc", ...)
    spectra_order: list of str
        the order of the spectra e.g  ["TT", "TE", "ET", "EE"]
    """
    
    cov_dict = {}
    for sid1, name1 in enumerate(spec_name_list):
        for sid2, name2 in enumerate(spec_name_list):
            if sid1 > sid2: continue
            cov_block = np.load(f"{cov_dir}/{cov_type}_{name1}_{name2}.npy")
            n_bins = int(cov_block.shape[0] / len(spectra_order))
            for s1, spec1 in enumerate(spectra_order):
                for s2, spec2 in enumerate(spectra_order):
                    sub_cov_block = cov_block[s1 * n_bins:(s1 + 1) * n_bins, s2 * n_bins:(s2 + 1) * n_bins]
                    cov_dict[name1, name2, spec1, spec2] = sub_cov_block
                    
    return cov_dict

def cov_dict_to_full_cov(cov_dict,
                         spec_name_list,
                         spectra_order = ["TT", "TE", "ET", "EE"],
                         remove_doublon = False,
                         check_pos_def = False):
                         
    """
    Build the full covariance matrix corresponding to spec_name_list using the covariance dictionnary
    There is an option to remove doublon e.g, TE and ET are the same for pa4_f150 x pa4_f150
    while they differ for pa4_f150 and pa5_f150
    There is also an option to check that the full matrix is positive definite and symmetric
    
    
    Parameters
    ----------
    cov_dict: dict
        dict of covariance element, keys of the dict are of the form
        cov_dict["dr6_pa4_f150xdr6_pa4_f150", "dr6_pa4_f150xdr6_pa5_f150", "TT", "EE"]
    spec_name_list: list of str
        list of the cross spectra
    spectra_order: list of str
        the order of the spectra e.g  ["TT", "TE", "ET", "EE"]
    remove_doublon: boolean
        wether to remove doublon, since TE == ET for spectrum of the type pa4_f150xpa4_f150, we combine
        them and should not keep them separated in the covariance matrix
    check_pos_def: boolean
        check that the full covariancde matrix is positive definite and symmetric
    """

    n_cross = len(spec_name_list)
    n_spec = len(spectra_order)
    # this looks complicated but just read the first element of the dict, takes its shape and divide by len(
    n_bins = int(cov_dict[list(cov_dict)[0]].shape[0])

    full_cov = np.zeros((n_cross * n_spec * n_bins,  n_cross * n_spec * n_bins))

    for sid1, name1 in enumerate(spec_name_list):
        for sid2, name2 in enumerate(spec_name_list):
            if sid1 > sid2: continue
            for s1, spec1 in enumerate(spectra_order):
                for s2, spec2 in enumerate(spectra_order):
                    id_start_1 = sid1 * n_bins + s1 * n_cross * n_bins
                    id_stop_1 = (sid1 + 1) * n_bins + s1 * n_cross * n_bins
                    id_start_2 = sid2 * n_bins + s2 * n_cross * n_bins
                    id_stop_2 = (sid2 + 1) * n_bins + s2 * n_cross * n_bins
                    full_cov[id_start_1:id_stop_1, id_start_2: id_stop_2] = cov_dict[name1, name2, spec1, spec2]
    transpose = full_cov.copy().T
    transpose[full_cov != 0] = 0
    full_cov += transpose
    
    if remove_doublon == True:
        block_to_delete = []
        for sid, name in enumerate(spec_name_list):
            na, nb = name.split("x")
            for s, spec in enumerate(spectra_order):
                id_start = sid * n_bins + s * n_cross * n_bins
                id_stop = (sid + 1) * n_bins + s * n_cross * n_bins
                if (na == nb) & (spec == "ET" or spec == "BT" or spec == "BE"):
                    block_to_delete = np.append(block_to_delete, np.arange(id_start, id_stop))
        block_to_delete = block_to_delete.astype(int)
        
        full_cov = np.delete(full_cov, block_to_delete, axis=1)
        full_cov = np.delete(full_cov, block_to_delete, axis=0)
        
    if check_pos_def == True:
        pspy_utils.is_pos_def(full_cov)
        pspy_utils.is_symmetric(full_cov)

    return full_cov
    
    
def read_cov_block_and_build_full_cov(spec_name_list,
                                      cov_dir,
                                      cov_type,
                                      spectra_order = ["TT", "TE", "ET", "EE"],
                                      remove_doublon = False,
                                      check_pos_def = False):
                                      
    """
    Build the full covariance matrix corresponding to spec_name_list from files
    There is an option to remove doublon e.g, TE and ET are the same for pa4_f150 x pa4_f150
    while they differ for pa4_f150 and pa5_f150
    There is also an option to check that the full matrix is positive definite and symmetric
    
    
    Parameters
    ----------
    spec_name_list: list of str
        list of the cross spectra
    cov_dir: str
        path to the folder with the covariance matrices block
    cov_type: str
        `cov_type` is used to specify which
         kind of covariance will be read ("analytic", "mc", ...)
    spectra_order: list of str
        the order of the spectra e.g  ["TT", "TE", "ET", "EE"]
    remove_doublon: boolean
        wether to remove doublon, since TE == ET for spectrum of the type pa4_f150xpa4_f150, we combine
        them and should not keep them separated in the covariance matrix
    check_pos_def: boolean
        check that the full covariancde matrix is positive definite and symmetric
    """
                                        
    cov_dict = read_cov_block_and_build_dict(spec_name_list,
                                             cov_dir,
                                             cov_type,
                                             spectra_order)
    
    full_cov = cov_dict_to_full_cov(cov_dict,
                                    spec_name_list,
                                    spectra_order,
                                    remove_doublon,
                                    check_pos_def)

    return full_cov


def full_cov_to_cov_dict(full_cov,
                         spec_name_list,
                         n_bins,
                         spectra_order = ["TT", "TE", "ET", "EE"]):
                         
    """
    Decompose the full covariance into a covariance dict, note that
    the full covariance should NOT have been produced with remove_doublon=True
    
    Parameters
    ----------
    full_cov: 2d array
        the full covariance to decompose
    spec_name_list: list of str
        list of the cross spectra
    n_bins: int
        the number of bins per spectra
    spectra_order: list of str
        the order of the spectra e.g  ["TT", "TE", "ET", "EE"]
    """

    n_cross = len(spec_name_list)
    n_spec = len(spectra_order)
    assert full_cov.shape[0] == n_cross * n_spec * n_bins, "full covariance do not have the correct shape"


    cov_dict = {}
    for sid1, name1 in enumerate(spec_name_list):
        for sid2, name2 in enumerate(spec_name_list):
            if sid1 > sid2: continue
            for s1, spec1 in enumerate(spectra_order):
                for s2, spec2 in enumerate(spectra_order):
                    id_start_1 = sid1 * n_bins + s1 * n_cross * n_bins
                    id_stop_1 = (sid1 + 1) * n_bins + s1 * n_cross * n_bins
                    id_start_2 = sid2 * n_bins + s2 * n_cross * n_bins
                    id_stop_2 = (sid2 + 1) * n_bins + s2 * n_cross * n_bins
                    cov_dict[name1, name2, spec1, spec2] = full_cov[id_start_1:id_stop_1, id_start_2: id_stop_2]

    return cov_dict
    
def cov_dict_to_file(cov_dict,
                     spec_name_list,
                     cov_dir,
                     cov_type,
                     spectra_order = ["TT", "TE", "ET", "EE"]):
                     
    """
    Write a cov dict into a bunch of files corresponding to cov mat block
    
    Parameters
    ----------
    cov_dict: dict
        dict of covariance element, keys of the dict are of the form
        cov_dict["dr6_pa4_f150xdr6_pa4_f150", "dr6_pa4_f150xdr6_pa5_f150", "TT", "EE"]
    spec_name_list: list of str
        list of the cross spectra
    cov_dir: str
        path to the folder with the covariance matrices block
    cov_type: str
        `cov_type` is used to specify which
         kind of covariance will be read ("analytic", "mc", ...)
    spectra_order: list of str
        the order of the spectra e.g  ["TT", "TE", "ET", "EE"]
    """

                     
    n_spec = len(spectra_order)
    n_bins = int(cov_dict[list(cov_dict)[0]].shape[0])

    for sid1, name1 in enumerate(spec_name_list):
        for sid2, name2 in enumerate(spec_name_list):
            if sid1 > sid2: continue
             
            cov_block = np.zeros((n_spec * n_bins, n_spec * n_bins))
 
            for s1, spec1 in enumerate(spectra_order):
                for s2, spec2 in enumerate(spectra_order):
                    cov_block[s1 * n_bins:(s1 + 1) * n_bins, s2 * n_bins:(s2 + 1) * n_bins] = cov_dict[name1, name2, spec1, spec2]
            
            np.save(f"{cov_dir}/{cov_type}_{name1}_{name2}.npy", cov_block)


def correct_analytical_cov(an_full_cov,
                           mc_full_cov):
    """
    Correct the analytical covariance matrix  using Monte Carlo estimated covariances.
    We keep the correlation structure of the analytical covariance matrices, rescaling
    the diagonal using MC covariances.
    Parameters
    ----------
    an_full_cov: 2d array
      Full analytical covariance matrix
    mc_full_cov: 2d array
      Full MC covariance matrix
     """
    an_full_corr = so_cov.cov2corr(an_full_cov)

    an_var = an_full_cov.diagonal()
    mc_var = mc_full_cov.diagonal()

    rescaling_var = np.where(mc_var>=an_var, mc_var, an_var)

    corrected_cov = so_cov.corr2cov(an_full_corr, rescaling_var)

    return corrected_cov
            
            
