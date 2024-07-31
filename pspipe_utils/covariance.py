from pspipe_utils import misc

from pspy import pspy_utils, so_cov, so_spectra

import numpy as np
import numba
from scipy.optimize import curve_fit
import pylab as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from pixell import utils

from itertools import combinations_with_replacement as cwr
from functools import partial


# for use in building couplings filenames
spintypes2fntags = {
    '00': '00',
    '02': '02',
    '++': 'pp',
    '--': 'mm'
}


# for use in processing coupling fields loaded from disk
optags2ops = {
    'identity': lambda x: x,
    'sqrt_inv': lambda x: np.sqrt(np.reciprocal(x, where=x!=0) * (x!=0))
}


def read_cov_block_and_build_dict(spec_name_list,
                                  cov_dir,
                                  cov_type,
                                  spectra_order=["TT", "TE", "ET", "EE"]):

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


def spec_dict_to_full_vec(spec_dict,
                          spec_name_list,
                          spectra_order=["TT", "TE", "ET", "EE"],
                          remove_doublon=False):
    n_cross = len(spec_name_list)
    n_spec = len(spectra_order)
    # this looks complicated but just read the first element of the dict, takes its shape and divide by len(
    n_bins = int(spec_dict[list(spec_dict)[0]].shape[0])

    full_vec = np.zeros(n_cross * n_spec * n_bins)

    for sid1, name1 in enumerate(spec_name_list):
        for s1, spec1 in enumerate(spectra_order):
            id_start_1 = sid1 * n_bins + s1 * n_cross * n_bins
            id_stop_1 = (sid1 + 1) * n_bins + s1 * n_cross * n_bins
            full_vec[id_start_1:id_stop_1] = spec_dict[name1, spec1]

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

        full_vec = np.delete(full_vec, block_to_delete, axis=0)        

    return full_vec


def cov_dict_to_full_cov(cov_dict,
                         spec_name_list,
                         spectra_order=["TT", "TE", "ET", "EE"],
                         remove_doublon=False,
                         check_pos_def=False):

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
                    full_cov[id_start_1:id_stop_1, id_start_2:id_stop_2] = cov_dict[name1, name2, spec1, spec2]
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


def full_cov_to_cov_dict(full_cov,
                         spec_name_list,
                         spectra_order=["TT", "TE", "ET", "EE"],
                         remove_doublon=False):
    
    # we need nbins. to calculate that we need to know what n_cross * n_spec 
    # becomes when removing doublon
    n_cross = len(spec_name_list)
    n_spec = len(spectra_order)

    num_deleted_blocks = 0
    if remove_doublon == True:
        for sid, name in enumerate(spec_name_list):
            na, nb = name.split("x")
            for s, spec in enumerate(spectra_order):
                if (na == nb) & (spec == "ET" or spec == "BT" or spec == "BE"):
                    num_deleted_blocks += 1

    num_cov_blocks = n_cross * n_spec - num_deleted_blocks
    n_bins = int(full_cov.shape[0] // num_cov_blocks)

    # now that we have nbins, we want to figure out which idxs were remapped
    # in the case of removing doublon
    idxs = np.arange(n_cross * n_spec * n_bins)
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

        idxs = np.delete(idxs, block_to_delete, axis=0)

    # now we can get the cov_dict with this mapping
    cov_dict = {}
    for sid1, name1 in enumerate(spec_name_list):
        na, nb = name1.split("x")
        for sid2, name2 in enumerate(spec_name_list):
            nc, nd = name2.split("x")
            if sid1 > sid2: continue
            for s1, spec1 in enumerate(spectra_order):
                for s2, spec2 in enumerate(spectra_order):
                    if remove_doublon == True:
                        # if a doublon, pretend we are looking at the reversed spec
                        if (na == nb) & (spec1 == "ET" or spec1 == "BT" or spec1 == "BE"):
                            s1 = spectra_order.index(spec1[::-1])
                        if (nc == nd) & (spec2 == "ET" or spec2 == "BT" or spec2 == "BE"):
                            s2 = spectra_order.index(spec2[::-1])
                        
                    # get the plain index, whether reversed spec or not
                    id_start_1 = sid1 * n_bins + s1 * n_cross * n_bins
                    id_stop_1 = (sid1 + 1) * n_bins + s1 * n_cross * n_bins
                    id_start_2 = sid2 * n_bins + s2 * n_cross * n_bins
                    id_stop_2 = (sid2 + 1) * n_bins + s2 * n_cross * n_bins
            
                    # get the new index based on the plain index, whether remove_doublon or not
                    id_start_1 = np.where(idxs == id_start_1)[0][0]
                    id_stop_1 = np.where(idxs == id_stop_1 - 1)[0][0] + 1
                    id_start_2 = np.where(idxs == id_start_2)[0][0]
                    id_stop_2 = np.where(idxs == id_stop_2 - 1)[0][0] + 1

                    cov_dict[name1, name2, spec1, spec2] = full_cov[id_start_1:id_stop_1, id_start_2:id_stop_2]

    return cov_dict


def read_cov_block_and_build_full_cov(spec_name_list,
                                      cov_dir,
                                      cov_type,
                                      spectra_order=["TT", "TE", "ET", "EE"],
                                      remove_doublon=False,
                                      check_pos_def=False):

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


# def full_cov_to_cov_dict(full_cov,
#                          spec_name_list,
#                          n_bins,
#                          spectra_order=["TT", "TE", "ET", "EE"]):

#     """
#     Decompose the full covariance into a covariance dict, note that
#     the full covariance should NOT have been produced with remove_doublon=True

#     Parameters
#     ----------
#     full_cov: 2d array
#         the full covariance to decompose
#     spec_name_list: list of str
#         list of the cross spectra
#     n_bins: int
#         the number of bins per spectra
#     spectra_order: list of str
#         the order of the spectra e.g  ["TT", "TE", "ET", "EE"]
#     """

#     n_cross = len(spec_name_list)
#     n_spec = len(spectra_order)
#     assert full_cov.shape[0] == n_cross * n_spec * n_bins, "full covariance do not have the correct shape"


#     cov_dict = {}
#     for sid1, name1 in enumerate(spec_name_list):
#         for sid2, name2 in enumerate(spec_name_list):
#             if sid1 > sid2: continue
#             for s1, spec1 in enumerate(spectra_order):
#                 for s2, spec2 in enumerate(spectra_order):
#                     id_start_1 = sid1 * n_bins + s1 * n_cross * n_bins
#                     id_stop_1 = (sid1 + 1) * n_bins + s1 * n_cross * n_bins
#                     id_start_2 = sid2 * n_bins + s2 * n_cross * n_bins
#                     id_stop_2 = (sid2 + 1) * n_bins + s2 * n_cross * n_bins
#                     cov_dict[name1, name2, spec1, spec2] = full_cov[id_start_1:id_stop_1, id_start_2: id_stop_2]

#     return cov_dict

def cov_dict_to_file(cov_dict,
                     spec_name_list,
                     cov_dir,
                     cov_type,
                     spectra_order=["TT", "TE", "ET", "EE"]):

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
                           mc_full_cov,
                           only_diag_corrections=False,
                           use_max_error=True):
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

    an_var = an_full_cov.diagonal()
    mc_var = mc_full_cov.diagonal()

    if use_max_error:
        rescaling_var = np.where(mc_var>=an_var, mc_var, an_var)
    else:
        rescaling_var = mc_var
    if only_diag_corrections:
        corrected_cov = an_full_cov - np.diag(an_var) + np.diag(rescaling_var)
    else:
        an_full_corr = so_cov.cov2corr(an_full_cov)
        corrected_cov = so_cov.corr2cov(an_full_corr, rescaling_var)

    return corrected_cov


def correct_analytical_cov_skew(an_full_cov, mc_full_cov, nkeep=50, do_final_mc=True, return_S=False):
    """
    Correct the analytical covariance matrix  using Monte Carlo estimated covariances.
    We use the skew method proposed by Sigurd Naess.
    to be merged with correct_analytical_cov  at some point
    Parameters
    ----------
    an_full_cov: 2d array
      Full analytical covariance matrix
    mc_full_cov: 2d array
      Full MC covariance matrix
    nkeep: int
      number of sigular value above the S/N threshold
    do_final_mc: bool
      If True, keep correlation structure of skew-svd corrected covmat, but
      replace total diagonal with monte carlo.
     """

    def skew(cov, dir=1):
        ocov = np.zeros(cov.shape)
        for i in range(len(cov)):
            ocov[i] = np.roll(cov[i], - i * dir)
        return ocov

    mc_var = mc_full_cov.diagonal()
    sqrt_an_full_cov  = utils.eigpow(an_full_cov, 0.5)
    inv_sqrt_an_full_cov = np.linalg.inv(sqrt_an_full_cov)
    res = inv_sqrt_an_full_cov @ mc_full_cov @ inv_sqrt_an_full_cov
    skew_res = skew(res)
    U, S, Vh = np.linalg.svd(skew_res)
    good = np.argsort(S)[::-1] < nkeep
    skew_res_clean = U.dot((S*good)[:,None] * Vh)
    res_clean = skew(skew_res_clean, dir = -1)
    res_clean = 0.5 * (res_clean + res_clean.T)
    res_clean = sqrt_an_full_cov @ res_clean @ sqrt_an_full_cov

    if do_final_mc:
        v  = np.diag(res_clean)
        res_clean = res_clean / (v[:,None] ** 0.5 * v[None,:] ** 0.5)
        corrected_cov = so_cov.corr2cov(res_clean, mc_var)
    else:
        corrected_cov = res_clean

    if return_S:
        return S, corrected_cov
    else:
        return corrected_cov


def correct_analytical_cov_eigenspectrum_ratio(an_full_cov, mc_full_cov, return_all=False):
    """Correct an analytical covariance matrix with a monte carlo covariance
    matrix. Assumes the following rotated monte carlo matrix is diagonal:

    mc_rot = (ana**-.5) @ mc @ (ana**-.5).T

    and then simply replaces the diagonal in that basis with the observed 
    values:

    ana_corrected = (ana**.5) @ np.diag(np.diag(mc_rot)) @ (ana**.5).T

    Parameters
    ----------
    an_full_cov : (nblock*nbin, nblock*nbin) np.ndarray
        Analytic covariance matrix to be corrected.
    mc_full_cov : (nblock*nbin, nblock*nbin) np.ndarray
        Noisy monte carlo matrix to use to correct the analytic matrix.
    return_all : bool, optional
        If True, in addition to returning ana_corrected, also return the
        diagonal of the mc_rot matrix, by default False.

    Returns
    -------
    (nblock*nbin, nblock*nbin) np.ndarray, {(nblock*nbin,) np.ndarray}
        ana_corrected as above, and if return_all, np.diag(mc_rot)
    """
    sqrt_an_full_cov  = utils.eigpow(an_full_cov, 0.5)
    inv_sqrt_an_full_cov = np.linalg.inv(sqrt_an_full_cov)
    res = inv_sqrt_an_full_cov @ mc_full_cov @ inv_sqrt_an_full_cov.T # res should be close to the identity if an_full_cov is good
    res_diag = np.diag(res)
    
    corrected_cov = sqrt_an_full_cov @ np.diag(res_diag) @ sqrt_an_full_cov.T

    if return_all:
        return corrected_cov, res_diag
    else:
        return corrected_cov


def correct_analytical_cov_eigenspectrum_ratio_gp(lb, an_full_cov, mc_full_cov,
                                                  var_eigenspectrum_ratios_by_block=None,
                                                  idx_arrs_by_block=None, return_all=False):
    """Correct an analytical covariance matrix with a monte carlo covariance
    matrix. Assumes the following rotated monte carlo matrix is diagonal:

    mc_rot = (ana**-.5) @ mc @ (ana**-.5).T

    and then fits the diagonal in that basis with a Gaussian process (GP) on
    the observed values:

    ana_corrected = (ana**.5) @ np.diag(GP(np.diag(mc_rot))) @ (ana**.5).T

    The GP is applied to each block diagonal separately.

    Parameters
    ----------
    lb : (nbin,) np.ndarray
        The locations in ell of the bins.
    an_full_cov : (nblock*nbin, nblock*nbin) np.ndarray
        Analytic covariance matrix to be corrected.
    mc_full_cov : (nblock*nbin, nblock*nbin) np.ndarray
        Noisy monte carlo matrix to use to correct the analytic matrix.
    var_eigenspectrum_ratios_by_block : (nblock, nbin) np.ndarray, optional
        The variance of the diagonal elements of mc_rot, by default None. These
        would need to be precomputed in the rotated basis. If None, the
        noise level in the diagonal elements of mc_rot is estimated from the  
        elements themselves by the Gaussian process.
    idx_arrs_by_block : (nblock,) list, optional
        A list of np.ndarrays, each of which may have between 0 and nbin elements,
        that gives for each block the bin indices that should actually be used
        in the Gaussian Process, by default None. If the list is None, or if any
        element of the list is None, all elements for all blocks (or for the 
        specific block) are used in the Gaussian process. If an element of the list
        is an empty array, then no Gaussian process is run on that block; its 
        observed values are used without any smoothing applied to them.
    return_all : bool, optional
        If True, in addition to returning ana_corrected, also return the
        diagonal of the mc_rot matrix, the smoothed diagonal, and a list
        of the Gaussian process regression objects for each block,
        by default False.

    Returns
    -------
    (nblock*nbin, nblock*nbin) np.ndarray, {(nblock*nbin,) np.ndarray, 
    (nblock*nbin,) np.ndarray, (nblock,) list of sklearn.GaussianProcessRegressor
    objects}
        ana_corrected as above, and if return_all, np.diag(mc_rot),
        GP(np.diag(mc_rot)), and a list of the GPs for each block. If for any
        block the idx_arrs_by_block array is empty, the returned GP is None 
        since no GP was actually used for that block.
    """
    sqrt_an_full_cov = utils.eigpow(an_full_cov, 0.5)
    inv_sqrt_an_full_cov = np.linalg.inv(sqrt_an_full_cov)
    res = inv_sqrt_an_full_cov @ mc_full_cov @ inv_sqrt_an_full_cov.T # res should be close to the identity if an_full_cov is good
    res_diag = np.diag(res)

    n_spec = len(res_diag) // len(lb)

    res_diags = np.split(res_diag, n_spec)
    
    if var_eigenspectrum_ratios_by_block is None:
        var_eigenspectrum_ratios_by_block = [None] * n_spec

    if idx_arrs_by_block is None:
        idx_arrs_by_block = [None] * n_spec

    smoothed_res_diags = []
    gprs = []
    for i in range(n_spec):
        smoothed_gp_diag, gpr = smooth_gp_diag(lb, res_diags[i], var_eigenspectrum_ratios_by_block[i],
                                               idx_arr=idx_arrs_by_block[i], return_gpr=True)
        smoothed_res_diags.append(smoothed_gp_diag)
        gprs.append(gpr)

    smoothed_res_diag = np.hstack(smoothed_res_diags)

    corrected_cov = sqrt_an_full_cov @ np.diag(smoothed_res_diag) @ sqrt_an_full_cov.T

    if return_all:
        return corrected_cov, res_diag, smoothed_res_diag, gprs
    else:
        return corrected_cov


def smooth_gp_diag(lb, arr_diag, var_diag=None, idx_arr=None,
                   length_scale=200.0, length_scale_bounds=(2, 2e4),
                   noise_level=0.01, noise_level_bounds=(1e-6, 1e1),
                   n_restarts_optimizer=5, remove_mean=True, 
                   return_gpr=False):
    """Fit a Gaussian process (GP) to a 1-dimensional target from
    a 1-dimensional feature set. The GP uses a radial basis function (RBF)
    kernel to model the signal and a white noise kernel to model the noise.
    The RBF kernel has a constant kernel prefactor.

    Parameters
    ----------
    lb : (n,) np.ndarray
        The featureset.
    arr_diag : (n,) np.ndarray
        The target.
    var_diag : (n,) np.ndarray, optional
        The white-noise variance in the target, by default None. If None, the
        noise level in the target is estimated from the target values
        themselves by the Gaussian process.
    idx_arr : np.ndarray, optional
        An array of size 0..n that indexes which data points are fit by the GP,
        by default None. If None, all datapoints are used. Any unused datapoints
        are not fit, but rather the target values are simply adopted at face 
        value.
    length_scale : float, optional
        The initial RBF scale guess, by default 200.0.
    length_scale_bounds : tuple, optional
        The bounds of the RBF scale fit, by default (2, 2e4).
    noise_level : float, optional
        The initial white noise level guess, by default 0.01. Only used if 
        var_diag is None.
    noise_level_bounds : tuple, optional
        The bounds of the white noise level guess, by default (1e-6, 1e1).
        Only used if var_diag is None.
    n_restarts_optimizer : int, optional
        The number of optimizers to run to try to find a gloval minimum, 
        by default 5.
    remove_mean : bool, optional
        If True, subtract the mean of the target prior to the fit and add it
        back in the prediction, by default True.
    return_gpr : bool, optional
        If True, also return the sklearn.GaussianProcessRegressor object, by
        default False.

    Returns
    -------
    (n,) np.ndarray, {sklearn.GaussianProcessRegressor object}
        The smoothed target, and optionally the object used to smooth.
    """
    if idx_arr is None:
        idx_arr = np.arange(lb.size)

    out = arr_diag.copy()

    if len(idx_arr) > 0:
        # fit only on the indexed data bins
        X_train = lb[idx_arr, None]
        y_train = arr_diag[idx_arr]

        # remove mean
        if remove_mean:
            prior_mean = y_train.mean()
            y_train = y_train - prior_mean
        
        # get the fit
        if var_diag is None:
            kernel = 1.0 * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds) + \
                WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=0,
                                           n_restarts_optimizer=n_restarts_optimizer)
        else:
            kernel = 1.0 * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=var_diag[idx_arr],
                                           n_restarts_optimizer=n_restarts_optimizer)
        gpr.fit(X_train, y_train)
        y_fit = gpr.predict(X_train, return_std=False)

        if remove_mean:
            y_fit += prior_mean
        
        out[idx_arr] = y_fit
    else:
        gpr = None
    
    if return_gpr:
        return out, gpr
    else:
        return out


def canonize_connected_2pt(leg1, leg2, all_legs):
    """A connected 2-point term has two legs but is invariant to their
    order. Thus, if we enforce a strict global order (a canonical order)
    on all the possible legs, we can skip calculating redundant terms.

    Parameters
    ----------
    leg1 : any
        The first leg.
    leg2 : any
        The second leg.
    all_legs : list of any
        A list containing the global order of the possible legs.

    Returns
    -------
    2-tuple of any
        The supplied legs in the canonical order.
    """
    leg1_idx = all_legs.index(leg1)
    leg2_idx = all_legs.index(leg2)
    if leg2_idx < leg1_idx:
        return all_legs[leg2_idx], all_legs[leg1_idx]
    else:
        return all_legs[leg1_idx], all_legs[leg2_idx]
    

def canonize_disconnected_4pt(leg1, leg2, leg3, leg4, all_legs):
    """A disconnected 4-point term has two pairs of two legs but is invariant
    to the order of the pairs and the order of the legs within the pairs. Thus,
    if we enforce a strict global order (a canonical order) on all the possible
    pairs of legs and legs themselves, we can skip calculating redundant terms.
    Legs are paired like (leg1, leg2) and (leg3, leg4).

    Parameters
    ----------
    leg1 : any
        The first leg.
    leg2 : any
        The second leg.
    leg3 : any
        The third leg.
    leg4 : any
        The fourth leg.
    all_legs : list of any
        A list containing the global (canonical) order of the possible legs.
        Note, the canonical order of all leg pairs is constructed from
        all_legs.

    Returns
    -------
    4-tuple of any
        The supplied legs in the canonical order.
    """
    canonical_pair_1 = canonize_connected_2pt(leg1, leg2, all_legs)
    canonical_pair_2 = canonize_connected_2pt(leg3, leg4, all_legs)

    all_leg_pairs = list(cwr(all_legs, 2))
    (leg1, leg2), (leg3, leg4) = canonize_connected_2pt(canonical_pair_1,
                                                        canonical_pair_2,
                                                        all_leg_pairs)

    return leg1, leg2, leg3, leg4


def pol2pol_info(pol):
    """T -> (T, 0); E -> (P, 1); B -> (P, 2)"""
    assert pol in 'TEB', f'expected {pol=} to be one of T, E, B'
    return ('T' if pol == 'T' else 'P', 'TEB'.index(pol))
    

def get_ewin_info_from_field_info(field_info, d, mode, extra=None, return_paths_ops=False):
    """Return information on the effective window corresponding to field.

    Parameters
    ----------
    field_info : tuple
        Field information (survey, array, chan, split, pol).
    d : dict
        PSpipe param dict.
    mode : str
        One of 'w', 's', or 'ws', referring to 'analysis mask', 'sigma map',
        and their product, respectively.
    extra : str, optional
        Any other extra information to join via underscore with the effective
        window name, e.g. pixel area factors, by default None.
    return_paths_ops : bool, optional
        Whether to return the full path on-disk for each window composing the 
        effective window, and the name of a lambda function for each window
        composing the effective window, by default False. See Returns and Notes
        for more information.
    
    Returns
    -------
    str, {tuple, tuple}
        Effective window name, followed by a tuple containing the full path
        on-disk for each window composing the effective window, and another
        tuple containing the name of a lambda function for each window composing 
        the effective window. The lambda function is to be applied to
        the array upon loading it from disk to return the effective window,
        like arr = op(arr). See covariance.optags2ops.

    Notes
    -----
    The reason the lambda functions need to be returned by their string name 
    (see covariance.optags2ops for the actual lambda functions corresponding
    to those names) is because the entire (str, tuple, tuple) object will be
    used by PSpipe scripts as a "leg" passed into canonize_connected_2pt,
    canonize_disconnected_4pt. In that case, we need everything in the leg
    to play nicely with list.index(), which lambda functions do not.
    """
    sv1, ar1, chan1, split1, pol1 = field_info

    if mode not in ['w', 's', 'ws']:
        raise ValueError(f"{mode=} not one of 'w', 's', ws'")
    
    # hard-coding that analysis masks don't depend on split
    if 'w' in mode:
        polstr = 'T' if pol1 == 'T' else 'pol'
        w_full_path = d[f'window_{polstr}_{sv1}_{ar1}_{chan1}']
        w_alias = d[f'window_{polstr}_{sv1}_{ar1}_{chan1}_alias']
        w_op = 'identity'
    
    # hard-coding that sigma maps don't depend on pol
    if 's' in mode:
        s_full_path = d[f'ivars_{sv1}_{ar1}_{chan1}'][split1]
        s_alias = d[f'ivars_{sv1}_{ar1}_{chan1}_aliases'][split1]
        s_op = 'sqrt_inv'

    if extra is None:
        extra = ''
    else:
        extra = f'_{extra}'
    
    if mode == 'w':
        if return_paths_ops:
            return w_alias + extra, (w_full_path,), (w_op,)
        else:
            return w_alias + extra
    elif mode == 's':
        if return_paths_ops:
            return s_alias + extra, (s_full_path,), (s_op,)
        else:
            return s_alias + extra
    elif mode == 'ws':
        if return_paths_ops:
            return f'{w_alias}_{s_alias}' + extra, (w_full_path, s_full_path), (w_op, s_op)
        else:
            return f'{w_alias}_{s_alias}' + extra


def get_mock_noise_ps(lmax, lknee, lcap, pow):
    """Get a mock power spectrum that is white at high-ell, a power-law at low
    ell, but is capped at a given minimum ell.

    Parameters
    ----------
    lmax : int
        lmax of power spectrum.
    lknee : int
        lknee of power law.
    lcap : int
        minimum ell at which the power law is capped.
    pow : scalar
        exponent of power law.

    Returns
    -------
    np.ndarray (lmax+1,)
        mock power spectrum. 
    """
    ells = np.arange(lmax + 1, dtype=np.float64)
    ps = np.zeros_like(ells)
    ps[lcap:] = (ells[lcap:]/lknee)**pow + 1
    ps[:lcap] = ps[lcap]
    return ps


def bin_spec(specs, bin_low, bin_high):
    """Bin spectra along their last axis.

    Parameters
    ----------
    specs : (..., nell) np.ndarray
        Spectra to be binned, with ell along last axis.
    bin_low : (nbin) np.ndarray
        Inclusive low-bounds of bins.
    bin_high : (nbin)
        Inclusive high-bounds of bins.

    Returns
    -------
    (..., nbin) np.ndarray
        Binned spectra.
    """
    out = np.zeros((*specs.shape[:-1], len(bin_low)))
    for i in range(len(bin_low)):
        out[..., i] = specs[..., bin_low[i]:bin_high[i] + 1].mean(axis=-1) 
    return out


def bin_mat(mats, bin_low, bin_high):
    """Bin a matrix along its last two axes.

    Parameters
    ----------
    mats : (..., nell, nell) np.ndarray
        Matrices to be binned, with ells along last two axes.
    bin_low : (nbin) np.ndarray
        Inclusive low-bounds of bins.
    bin_high : (nbin)
        Inclusive high-bounds of bins.

    Returns
    -------
    (..., nbin, nbin) np.ndarray
        Binned matrices.
    """
    out = np.zeros((*mats.shape[:-2], len(bin_low), len(bin_low)))
    for i in range(len(bin_low)):
        for j in range(len(bin_low)):
            out[..., i, j] = mats[..., bin_low[i]:bin_high[i] + 1, bin_low[j]:bin_high[j] + 1].mean(axis=(-2, -1))
    return out


def get_expected_pseudo_func(mcm, tf, ps, bin_low=None, bin_high=None):
    """Build a function that returns the theory pseudospectrum from a theory
    powerspectrum, multiplied by some one-dimensional transfer function raised
    to the alpha power:

    f(alpha): mcm @ (tf**alpha * ps)

    Parameters
    ----------
    mcm : (nell, nell) np.ndarray
        Mode-coupling matrix.
    tf : (nell)
        One-dimensional transfer function.
    ps : (nell)
        Power spectrum.
    bin_low : (nbin) np.ndarray, optional
        One-dimensional array of inclusive bin lowerbounds, by default None.
        Binning occurs if not None.
    bin_high : (nbin) np.ndarray, optional
        One-dimensional array of inclusive bin upperbounds, by default None.

    Returns
    -------
    function
        f(alpha): mcm @ (tf**alpha * ps)
    """
    if bin_low is not None:
        def f(alpha):
            return bin_spec(mcm @ (tf**alpha * ps), bin_low, bin_high)       
    else:
        def f(alpha):
            return mcm @ (tf**alpha * ps)
    return f


def get_expected_cov_diag_func(mcm, w2, tf, ps, coup, bin_low=None, bin_high=None, pre_mcm_inv=None):
    """Build a function that returns the theory covariance diagonal (under the 
    arithmetic INKA approximation) from a theory powerspectrum, multiplied by
    some one-dimensional transfer function raised to the alpha power, and the
    other covariance ingredients (mcm, w2, coup):

    f(alpha): 0.5 * ((mcm @ (tf**(alpha/2) * ps / w2)) + (mcm @ (tf**(alpha/2) * ps / w2))[:, None])**2 * coup

    Parameters
    ----------
    mcm : (nell, nell) np.ndarray
        Mode-coupling matrix.
    w2 : scalar
        w2 factor of the mask generating the mode-coupling matrix.
    tf : (nell)
        One-dimensional transfer function.
    ps : (nell)
        Power spectrum.
    coup : (nell, nell) np.ndarray
        Coupling matrix.
    bin_low : (nbin) np.ndarray, optional
        One-dimensional array of inclusive bin lowerbounds, by default None.
        Binning occurs if not None.
    bin_high : (nbin) np.ndarray, optional
        One-dimensional array of inclusive bin upperbounds, by default None.
    pre_mcm_inv : (nell, nell) np.ndarray, optional
        Linear operator that takes pseudospectra to powerspectra, used in 
        calculating the powerspectrum covariance matrix, by default None.
        Returns the powerspectrum covariance matrix if not None.

    Returns
    -------
    function
        f(alpha): 0.5 * ((mcm @ (tf**(alpha/2) * ps / w2)) + (mcm @ (tf**(alpha/2) * ps / w2))[:, None])**2 * coup
    """
    def pseudo_cov(alpha):
        return 0.5 * ((mcm @ (tf**(alpha/2) * ps / w2)) + (mcm @ (tf**(alpha/2) * ps / w2))[:, None])**2 * coup

    if bin_low is not None:
        if pre_mcm_inv is None:
            def f(alpha):
                return np.diag(bin_mat(pseudo_cov(alpha), bin_low, bin_high))
        else:
            def f(alpha):
                return np.diag(bin_mat(pre_mcm_inv @ pseudo_cov(alpha) @ pre_mcm_inv.T, bin_low, bin_high))
    else:
        if pre_mcm_inv is None:
            def f(alpha):
                return np.diag(pseudo_cov(alpha))
        else:
            def f(alpha):
                return np.diag(pre_mcm_inv @ pseudo_cov(alpha) @ pre_mcm_inv.T)  
    
    return f


def fit_func(x, alpha, func, xmin, den):
    """A wrapper around a one-dimensional function to fit (a function of alpha
    only) that allows its result to be normalized by some denominator and only
    fit at and above some element xmin.

    Parameters
    ----------
    x : any
        x-values (not used)
    alpha : scalar
        See get_expected_pseudo_func and get_expected_cov_diag_func.
    func : function
        get_expected_pseudo_func or get_expected_cov_diag_func.
    xmin : int
        Minimum element used in the fit.
    den : np.ndarray
        Denominator used in the fitting, same size as func.

    Returns
    -------
    np.ndarray
        The normalized function result.

    Notes
    -----
    Passed to scipy.optimize.curvefit by freezing func, xmin, and den with
    functools.partial. Note, scipy.optimize.curvefit requires x-values to be
    passable, but we don't use them.
    """
    return np.divide(func(alpha)[xmin:], den[xmin:], where=den[xmin:]!=0, out=np.zeros_like(den[xmin:]))


def get_alpha_fit(res_dict, func, target, tag, xmin=0):
    """A wrapper that performs a fit of a function taking a single parameter
    to a target dataset. The results are tagged in a dictionary.

    Parameters
    ----------
    res_dict : dict
        The dictionary holding the results.
    func : callable
        A function of a single variable.
    target : (ntar, size) array-like
        A 2d array of data. The mean of this array over the first axis is the
        data vector we attempt to fit with func. The sample standard deviation
        of this array over the first axis is the error vector we use in the fit.
    tag : str
        A description for this fit to tag results with.
    xmin : int, optional
        The minimum element into the last axis of target we actually use in the
        fit, by default 0.

    Notes
    -----
    The passed func is further passed into fit_func; thus, it must be one of 
    get_expected_pseudo_func or get_expected_cov_diag_func. The single
    parameter we fit for is therefore alpha, the power to which we raise a
    transfer function template multiplying the power spectrum.
    """
    den = func(0)
    res_dict[f'{tag}_den'] = den

    res_dict[f'{tag}_mean'] = target.mean(axis=0) 
    res_dict[f'{tag}_var'] = target.var(axis=0, ddof=1) / len(target)

    ydata = np.divide(res_dict[f'{tag}_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict[f'{tag}_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
    res_dict[f'{tag}_ydata'] = ydata
    res_dict[f'{tag}_yerr'] = yerr

    res_dict[f'{tag}_xmin'] = xmin

    popt, pcov = curve_fit(partial(fit_func, func=func, xmin=xmin, den=den), 1, ydata[xmin:], sigma=yerr[xmin:])
    best_fit = fit_func(1, popt[0], func, 0, den)

    res_dict[f'{tag}_alpha'] = popt[0]
    res_dict[f'{tag}_alpha_err'] = pcov[0, 0]**0.5 
    res_dict[f'{tag}_best_fit'] = best_fit
    res_dict[f'{tag}_best_fit_err'] = ydata - best_fit
    res_dict[f'{tag}_best_fit_stderr'] = np.divide(ydata - best_fit, yerr, where=yerr!=0, out=np.zeros_like(yerr))


def cmb_matrix_from_file(f_name, lmax, spectra, input_type='Dl'):
    """Return a 3x3 matrix of CMB power spectra from products
    on-disk. Dl factors are removed.

    Parameters
    ----------
    f_name : path-like
        Path to a cmb.dat file with 9 polarization-cross fields.
    lmax : int
        lmax of output. If cmb.dat file does not support up to the requested
        lmax, the value at the last available lmax is extended.
    spectra : list of str
        The list of polarization crosses, passed to so_spectra.read_ps.
    input_type : str, optional
        'Cl' or 'Dl', by default 'Dl'. If 'Dl', assuemd that the data in
        cmd.dat is in 'Dl' format. The 'Dl' factor is then removed, resulting
        in pure physical power spectra.

    Returns
    -------
    (3, 3, lmax+1) np.ndarray
        The TEB x TEB x nell ordered physical CMB power spectra.
    """
    ps_mat = np.zeros((3, 3, lmax+1))
    
    l, ps_theory = so_spectra.read_ps(f_name, spectra=spectra)
    assert l[0] == 2, 'the file is expected to start at l=2'
    _lmax = min(lmax, int(max(l)))  # make sure lmax doesn't exceed model lmax
    
    for p1, pol1 in enumerate('TEB'):
        for p2, pol2 in enumerate('TEB'):
            if input_type == 'Dl':
                ps_theory[pol1 + pol2] *= 2 * np.pi / (l * (l + 1))
            ps_mat[p1, p2, 2:(_lmax+1)] = ps_theory[pol1 + pol2][:(_lmax+1) - 2]
    
    ps_mat[..., _lmax+1:] = ps_mat[..., _lmax][..., None] # extend with last val
    
    return ps_mat


def foreground_matrix_from_files(f_name_tmp, sv_ar_chans_list, lmax, spectra, input_type='Dl'):
    """Return a (narrx3) x (narrx3) matrix of foreground power spectra from
    products on-disk. Dl factors are removed.

    Parameters
    ----------
    f_name_tmp : str
        Filename template to foreground cross-spectra on disk, to be populated
        with the 6 entries formed by unpacking a pair of 3-element tuples from
        sv_ar_chans_list.
    sv_ar_chans_list : list of str
        A list of tuples. Each tuple in should have 3 elements: the field
        survey string, the array string, and the channel string.
    lmax : int
        lmax of output. If cmb.dat file does not support up to the requested
        lmax, the value at the last available lmax is extended.
    spectra : list of str
        The list of polarization crosses, passed to so_spectra.read_ps.
    input_type : str, optional
        'Cl' or 'Dl', by default 'Dl'. If 'Dl', assuemd that the data in
        cmd.dat is in 'Dl' format. The 'Dl' factor is then removed, resulting
        in pure physical power spectra.

    Returns
    -------
    (nsac, 3, nsac, 3, lmax+1) np.ndarray
        The (nsacxTEB) x (nsacxTEB) x nell ordered physical foreground,
        power spectra. Here, nsac is len(sv_ar_chans_list).
    """
    nsacs = len(sv_ar_chans_list)
    fg_mat = np.zeros((nsacs, 3, nsacs, 3, lmax+1))
    
    for sac1, sv_ar_chan1 in enumerate(sv_ar_chans_list):
        for sac2, sv_ar_chan2 in enumerate(sv_ar_chans_list):
            l, fg_theory = so_spectra.read_ps(f_name_tmp.format(*sv_ar_chan1, *sv_ar_chan2), spectra=spectra)
            assert l[0] == 2, 'the file is expected to start at l=2'
            _lmax = min(lmax, int(max(l)))  # make sure lmax doesn't exceed model lmax
            
            for p1, pol1 in enumerate('TEB'):
                for p2, pol2 in enumerate('TEB'):
                    if input_type == 'Dl':
                        fg_theory[pol1 + pol2] *=  2 * np.pi / (l * (l + 1))
                    fg_mat[sac1, p1, sac2, p2, 2:(_lmax+1)] = fg_theory[pol1 + pol2][:(_lmax+1) - 2]

    fg_mat[..., _lmax+1:] = fg_mat[..., _lmax][..., None] # extend with last val
    
    return fg_mat


# numba can help speed up the basic array operations ~2x
@numba.njit(parallel=True)
def add_term_to_pseudo_cov(pseudo_cov, C12, C34, coupling):
    """Accumulates terms of a pseudocovariance block. Each term looks like:

    (C12_2d + C12_2d.T) * (C34_2d + C34_2d.T) * coupling

    where C_2d indicates a spectrum that has been broadcast to two dimensions.
    This form indicates we are doing "arithmetic" symmetrization under the NKA.

    Parameters
    ----------
    pseudo_cov : (nell, nell) np.ndarray
        The accumulating array that contains the given term. Updated inplace.
    C12 : (nell,) np.ndarray
        The first spectrum in the covariance block under the NKA.
    C34 : (nell,) np.ndarray
        The second spectrum in the covariance block under the NKA.
    coupling : (nell, nell) np.ndarray
        The 4-point coupling term.

    Notes
    -----
    By wrapping in numba.njit(parallel=True), we are asserting that all
    operations in this function support parallel semantics, and that 
    the dtype and shape of the inputs is fixed.
    """
    C12_2d = np.expand_dims(C12, 0)
    C12_2d = np.broadcast_to(C12_2d, coupling.shape)

    C34_2d = np.expand_dims(C34, 0)
    C34_2d = np.broadcast_to(C34_2d, coupling.shape)
    pseudo_cov += (C12_2d + C12_2d.T) * (C34_2d + C34_2d.T) * coupling


def get_binning_matrix(bin_lo, bin_hi, lmax, cltype='Dl'):
    """Returns P_bl, the binning matrix that turns C_ell into C_b."""
    l = np.arange(2, lmax) # assumes 2:lmax ordering
    if cltype == 'Dl':
        fac = (l * (l + 1) / (2 * np.pi))
    elif cltype == 'Cl':
        fac = l * 0 + 1
    n_bins = len(bin_lo)  # number of bins is same for all spectra in block
    Pbl = np.zeros((n_bins, lmax-2))
    for ibin in range(n_bins):
        loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))[0]
        Pbl[ibin, loc] = fac[loc] / len(loc)
    return Pbl


def read_covariance(cov_file,
                    beam_error_corrections,
                    mc_error_corrections):
    """
    Read the covariance matrix from `cov_file`
    applying some corrections if requested.

    Parameters
    ----------
    cov_file: str
        Path to the .npy file
    beam_error_corrections: bool
        Flag used to include beam error corrections
    mc_error_corrections: bool
        Flag used to include MC error corrections
    """
    cov = np.load(cov_file)

    if mc_error_corrections:
        mc_corr_cov_file = misc.str_replace(cov_file, "analytic_cov", "analytic_cov_with_mc_corrections")
        cov = np.load(mc_corr_cov_file)

    if beam_error_corrections:
        beam_cov_file = misc.str_replace(cov_file, "analytic_cov", "analytic_beam_cov")
        beam_cov = np.load(beam_cov_file)
        cov = cov + beam_cov

    return cov


def get_x_ar_to_x_freq_P_mat(x_ar_cov_list, x_freq_cov_list, binning_file, lmax):
    """
    Create a projector matrix from x_freq spectra to x_array spectra
    To understand why we need this projector let's imagine the following problem.
    We have x_array spectra Dl_{x_ar}, and x_freq spectra Dl_{x_freq}, the idea of combining all x_ar spectra into a set of x_freq spectra
    rely on the following assumption Dl_{x_ar} = Dl_{x_freq} + noise; that is: all array spectra are noisy measurement of
    underlying x_freq spectra.
    if we want to generalize the equation to many x_freq and x_ar spectra we can write \vec{Dl_{x_ar}} = P \vec{Dl_{x_freq}} + \vec{n}
    the idea of this routine is to build the P_matrix, that is to associate x_freq spectra to each x_ar spectra.

    Parameters
     ----------
    x_ar_cov_list: list of tuples
        this list represent the order of spectra entering the x_ar covariance matrix
        it also give some other relevant information such as the effective frequency of each spectra
        expected format is (spec, name, nu_pair)
        e.g ('TT', 'dr6_ar1xdr6_ar1', (150,150))
    x_freq_cov_list: list of tuples
        this list represent the order of spectra entering the x_freq covariance matrix
        expected format is (spec, nu_pair)
        e.g ('TT', (150,150))
    binning_file: str
      a binning file with format bin low, bin high, bin mean
    lmax: int
      the maximum multipole to consider
    """

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)

    n_el_x_ar = len(x_ar_cov_list) # number of block in the x_ar cov mat
    n_el_x_freq = len(x_freq_cov_list) # number of block in the x_freq cov mat

    P_mat = np.zeros((n_el_x_ar * n_bins, n_el_x_freq * n_bins))

    for id_ar, x_ar_cov_el in enumerate(x_ar_cov_list):
        spec1, _, nu_pair1 = x_ar_cov_el # spec1 here is TT,TE,...,BB, nupair1 is the associated effective freq in format (freq1, freq2)

        for id_freq, x_freq_cov_el in enumerate(x_freq_cov_list):
            spec2, nu_pair2 = x_freq_cov_el # spec2 here is TT,TE,...,BB, nupair2 is the associated effective freq in format (freq1, freq2)

            # so the first part if for spectra such as TT, EE, BB
            if (spec1[0] == spec1[1]) and (spec1 == spec2):
                # for these guys we want to check that the freq pair is the same (or inverted since <TT_90x150> = <TT_150x90>)
                if (nu_pair1 == nu_pair2) or (nu_pair1 == nu_pair2[::-1]):
                    # if that's the case we will include it in the projector, what this mean is that we say that this
                    # particular x_freq spectrum will project into this particular x_ar spectrum
                    P_mat[id_ar * n_bins: (id_ar + 1) * n_bins, id_freq * n_bins: (id_freq + 1) * n_bins] = np.identity(n_bins)

            # for cross field spectra such as TE, TB, ET, BT, EB, BE we need a bit more work
            # the idea is to construct a xfreq cov mat with only TE, TB, EB
            # so x_freq_cov_list do not contains any ET, BT, BE
            # the idea is to associate ET_90x150 to TE_150x90, so reverting the frequency pair ordering

            # we start with the  TE, TB, EB case
            if (spec1[0] != spec1[1]) and (spec1 == spec2):
                if (nu_pair1 == nu_pair2):
                    P_mat[id_ar * n_bins: (id_ar + 1) * n_bins, id_freq * n_bins: (id_freq + 1) * n_bins] = np.identity(n_bins)

            # for the ET, BT, BE case we reverse the order of the freq pair E_90 T_150 = T_150 x E_90
            if (spec1[0] != spec1[1]) and (spec1 == spec2[::-1]):
                if (nu_pair1 == nu_pair2[::-1]):
                    P_mat[id_ar * n_bins: (id_ar + 1) * n_bins, id_freq * n_bins: (id_freq + 1) * n_bins] = np.identity(n_bins)
    return P_mat


def get_x_freq_to_final_P_mat(x_freq_cov_list, final_cov_list, binning_file, lmax):
    """
    Create a projector matrix from final spectra to x_freq spectra
    To understand why we need this projector let's imagine the following problem.
    We have x_freq spectra Dl_{x_freq}, and final spectra Dl_{final}, the idea of combining all x_freq spectra into a set of final spectra
    rely on the following assumption Dl_{x_freq} = Dl_{x_final} + noise; that is: all x_freq spectra are noisy measurement of
    underlying final spectra.
    Of course this make no sense in Temperature, since different x_freq spectra see different level of foreground.
    Given how small the fg are in polarisation, it does make sense to look at spectra combination of all cross frequency
    (mostly for plotting)

    we can write \vec{Dl_{x_freq}} = P \vec{Dl_{final}} + \vec{n}
    the idea of this routine is to build the P matrix, which associates x_freq spectra to each final spectra.

    Parameters
     ----------
    x_freq_cov_list: list of tuples
        this list represents the order of spectra entering the x_freq covariance matrix
        expected format is (spec, nu_pair)
        e.g ('TT', (150,150))
    final_cov_list: list of tuples
        this list represents the order of spectra entering the final covariance matrix
        expected format is (spec, nu_pair) for TT
        or (spec, None) for other spectra since the final polarisation spectra are combined across the different frequency
        pair
    binning_file: str
      a binning file with format bin low, bin high, bin mean
    lmax: int
      the maximum multipole to consider
    """
    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)

    n_el_x_freq = len(x_freq_cov_list)
    n_el_final = len(final_cov_list)

    P_mat = np.zeros((n_el_x_freq * n_bins, n_el_final * n_bins))

    for id_freq, x_freq_cov_el in enumerate(x_freq_cov_list):
        spec1, nu_pair1 = x_freq_cov_el
        for id_final, final_cov_el in enumerate(final_cov_list):
            spec2, nu_pair2 = final_cov_el
            if (spec1 == spec2) and (spec1 == "TT"):
                if (nu_pair1 == nu_pair2):
                    P_mat[id_freq * n_bins: (id_freq + 1) * n_bins, id_final * n_bins: (id_final + 1) * n_bins] = np.identity(n_bins)
            else:
                if (spec1 == spec2):
                    P_mat[id_freq * n_bins: (id_freq + 1) * n_bins, id_final * n_bins: (id_final + 1) * n_bins] = np.identity(n_bins)

    return P_mat


def read_x_ar_spectra_vec(spec_dir,
                          spec_name_list,
                          end_of_file,
                          spectra_order=["TT", "TE", "ET", "EE"],
                          type="Dl"):


    """
    This function read spectra files on disk to create a vector that correspond
    to the full covariance matrix corresponding to spec_name_list

    Parameters
     ----------
     spec_name_list: list of str
         list of the cross spectra
     spec_dir: str
         path to the folder with the spectra
     end_of_file: str
         the str at the end of the spectra file
    spectra_order: list of str
         the order of the spectra e.g  ["TT", "TE", "ET", "EE"]
     type: str
         the spectra type can be "Dl" or "Cl"
     """

    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    data_vec = []

    for spec in spectra_order:
        for spec_name in spec_name_list:
            na, nb = spec_name.split("x")
            lb, Db = so_spectra.read_ps(f"{spec_dir}/{type}_{spec_name}_{end_of_file}.dat", spectra=spectra)
            if (spec == "ET" or spec == "BT" or spec == "BE") & (na == nb): continue
            data_vec = np.append(data_vec, Db[spec])

    return data_vec

def read_x_ar_theory_vec(bestfit_dir,
                         mcm_dir,
                         spec_name_list,
                         lmax,
                         spectra_order=["TT", "TE", "ET", "EE"]):

    """
    This function read the theory and fg model from disk and bin it to create a vector that correspond
    to the full covariance matrix corresponding to spec_name_list
    Parameters
    ----------
    bestfit_dir: str
        path to the folder with the theory and fg model
    mcm_dir: str
        path to the folder with the binning matrix
    spec_name_list: list of str
        list of the cross spectra
    freq_pair_list: list of two-d list
        list of the frequencies corresponding the the spec_name in spec_name_list
    lmax: interger
        the maximum multipole to consider
    spectra_order: list of str
         the order of the spectra e.g  ["TT", "TE", "ET", "EE"]
     """

    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    theory_vec = []
    for spec in spectra_order:
        for spec_name in spec_name_list:
            na, nb = spec_name.split("x")

            l, Dl = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)
            l, Dfl = so_spectra.read_ps(f"{bestfit_dir}/fg_{na}x{nb}.dat", spectra=spectra)

            # this is slightly incaccurate in some cases
            Bbl = np.load(f"{mcm_dir}/{spec_name}_Bbl_spin0xspin0.npy")
            Db = np.dot(Bbl, Dl[spec][:lmax] + Dfl[spec][:lmax])

            if (spec == "ET" or spec == "BT" or spec == "BE") & (na == nb): continue
            theory_vec = np.append(theory_vec, Db)

    return theory_vec


def get_indices(
    bin_low,
    bin_high,
    bin_mean,
    spec_name_list,
    spectra_cuts=None,
    spectra_order=["TT", "TE", "ET", "EE"],
    selected_spectra=None,
    excluded_spectra=None,
    excluded_map_set=None,
    only_TT_map_set=None,
    use_bin_edges=True
):
    """
    This function returns the covariance and spectra indices selected given a set of multipole cuts

    Parameters
    ----------
    bin_mean: 1d array
        the center values of the data binning
    spec_name_list: list of str
        list of the cross spectra
    spectra_cuts: dict
        the dictionnary holding the multipole cuts. Its general form must be
        '{"array1": {"T": [lmin1, lmax1], "P": [lmin2, lmax2]}...}'
    spectra_order: list of str
        the order of the spectra e.g  ["TT", "TE", "ET", "EE"]
    selected_spectra: list of str
        the list of spectra to be kept
    excluded_spectra: list of str
        the list of spectra to be excluded
    excluded_map_set: list of str
        the list of map set to be excluded
    only_TT_map_set: list of str
        map_set for which we only wish to use the TT power spectrum
    use_bin_edges: bool
        use bin_low and bin_high together with spectra_cuts (or 0 and inf) to
        cut spectra, otherwise use bin_mean
    """
    if selected_spectra and excluded_spectra:
        raise ValueError("Both 'selected_spectra' and 'excluded_spectra' can't be set together!")
    if selected_spectra:
        excluded_spectra = [spec for spec in spectra_order if spec not in selected_spectra]
    excluded_spectra = excluded_spectra or []

    excluded_map_set = excluded_map_set or []
    
    only_TT_map_set = only_TT_map_set or []

    spectra_cuts = spectra_cuts or {}
    indices_in = np.array([])

    nbins = len(bin_low)
    shift_indices = 0
    
    bin_out_dict = {}
    id_min = 0
    for spec in spectra_order:
        X, Y = spec
        for spec_name in spec_name_list:
            na, nb = spec_name.split("x")
            if spec in ["ET", "BT", "BE"] and na == nb:
                continue
            if spec in excluded_spectra:
                shift_indices += nbins
                continue

            if na in excluded_map_set or nb in excluded_map_set:
                shift_indices += nbins
                continue
                
            if na in only_TT_map_set or nb in only_TT_map_set:
                if spec != "TT":
                    shift_indices += nbins
                    continue
                
            ca, cb = spectra_cuts.get(na, {}), spectra_cuts.get(nb, {})

            if X != "T": X = "P"
            if Y != "T": Y = "P"

            if not ca: #return True if ca is empty
                lmin_Xa, lmax_Xa = 0, np.inf
            else:
                lmin_Xa, lmax_Xa = ca[X][0],  ca[X][1]
            
            if not cb:
                lmin_Yb, lmax_Yb = 0, np.inf
            else:
                lmin_Yb, lmax_Yb = cb[Y][0],  cb[Y][1]

            lmin = np.maximum(lmin_Xa, lmin_Yb)
            lmax = np.minimum(lmax_Xa, lmax_Yb)

            if use_bin_edges:
                idx = np.arange(nbins)[(lmin < bin_low) & (bin_high < lmax)]
            else:
                idx = np.arange(nbins)[(lmin < bin_mean) & (bin_mean < lmax)]
            
            indices_in = np.append(indices_in, idx + shift_indices)
            
            if lmin != lmax:
                bin_out_dict[f"{spec_name}", f"{spec}"] = (np.arange(id_min, id_min + len(idx)), bin_mean[idx])
                id_min += len(idx)

            shift_indices += nbins
                
    return bin_out_dict,  indices_in.astype(int)

def compute_chi2(
    data_vec,
    theory_vec,
    cov,
    binning_file,
    lmax,
    spec_name_list,
    spectra_cuts=None,
    spectra_order=["TT", "TE", "ET", "EE"],
    selected_spectra=None,
    excluded_spectra=None,
    excluded_map_set=None,
    only_TT_map_set=None,
):
    """
    This function computes the chi2 value between data/sim spectra wrt theory spectra given
    the data covariance and a set of multipole cuts

    Parameters
    ----------
    data_vec: 1d array
      the vector holding the data spectra
    theory_vec: 1d array
      the vector holding the theoritical spectra
    cov: 2d array
      the covariance matrix
    binning_file: str
      a binning file with format bin low, bin high, bin mean
    lmax: int
      the maximum multipole to consider
    spec_name_list: list of str
        list of the cross spectra
    spectra_cuts: dict
        the dictionnary holding the multipole cuts. Its general form must be
        '{"array1": {"T": [lmin1, lmax1], "P": [lmin2, lmax2]}...}'
    spectra_order: list of str
        the order of the spectra e.g  ["TT", "TE", "ET", "EE"]
    selected_spectra: list of str
        the list of spectra to be kept
    excluded_spectra: list of str
        the list of spectra to be excluded
    excluded_map_set: list of str
        the list of map_set to be excluded
    only_TT_map_set: list of str
        map_set for which we only wish to use the TT power spectrum

    """
    bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    _, indices = get_indices(
        bin_low,
        bin_high,
        bin_mean,
        spec_name_list,
        spectra_cuts=spectra_cuts,
        spectra_order=spectra_order,
        selected_spectra=selected_spectra,
        excluded_spectra=excluded_spectra,
        excluded_map_set=excluded_map_set,
        only_TT_map_set=only_TT_map_set,
    )

    delta = data_vec[indices] - theory_vec[indices]
    cov = cov[np.ix_(indices, indices)]

    chi2 = delta @ np.linalg.inv(cov) @ delta
    ndof = len(indices)
    return chi2, ndof

def get_max_likelihood_cov(P, inv_cov, check_pos_def=False, force_sim=False):

    """
    Get the maximum likelihood cov mat associated to inv_cov and the passage matrix P
    Parameters
     ----------
    P: 2d array
        the passage matrix (e.g from the x_ar space to the x_freq space)
    inv_cov: 2d_array
        the original inverse covariance matrix
    check_pos_def: boolean
        check wether the ml matrix is pos def and symmetric
    """

    cov_ml = np.linalg.inv(P.T @ inv_cov @ P)

    if force_sim == True:
        temp = cov_ml.copy()
        cov_ml = np.tril(temp) + np.triu(temp.T, 1)

    if check_pos_def == True:
        pspy_utils.is_pos_def(cov_ml)
        pspy_utils.is_symmetric(cov_ml)

    return cov_ml

def max_likelihood_spectra(cov_ml, inv_cov, P, data_vec):
    """
    Get the maximum likelihood vector associated to data_vec and the passage matrix P
    Parameters
     ----------
    cov_ml : 2d array
        the maximum likelihood covariance resulting
        of the combination done with the P matrix
    inv_cov: 2d_array
        the original inverse covariance matrix
    P: 2d array
        the passage matrix (e.g from the x_ar space to the x_freq space)
    data_vec : 1d array
        a vector made from the original spectra with order corresponding to inv_cov
    """
    return cov_ml @ P.T @ inv_cov @ data_vec

def from_vector_and_cov_to_ps_and_std_dict(vec, cov, spectra_order, spec_block_order, binning_file, lmax):
    """
    Take a vector of power spectra and their associated covariance and extract a dictionnary
    of power spectra and a dictionnary of std.
    The organisation of the vector and cov should be the following

    ps[spectra_order[0], spec_block_order[0]] = vec[0:nbins]
    ps[spectra_order[0], spec_block_order[1]] = vec[nbins:2*nbins]
    ....

    spectra_order[0] is usually TT, spec_block_order can be anything depending on
    how was aranged the vector and cov.

    Parameters
     ----------
    vec: 1d array
        vector of power spectrum
        the arangment of the vector should
        ( (spectra_order[0], spec_block_order[0]), (spectra_order[0], spec_block_order[1]),
           ........, (spectra_order[0], spec_block_order[n]), ...., (spectra_order[m], spec_block_order[n]))
    cov: 2d array
        the covariance associated to the vector of power spectrum
    spectra_order: list of string
        e.g ["TT","TE", EE"] or ["TT","TE","ET","EE"]
        the order of the spectra block
    spec_block_order: list of string
        e.g ["pa4_f150xpa4_f150", "pa4_f150xpa5_f150", "pa5_f150xpa5_f150"]
        the arrangment of the spectra within a block
    binning_file: str
      a binning file with format bin low, bin high, bin mean
    lmax: int
      the maximum multipole to consider
    """

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)
    var = cov.diagonal()

    spec_dict, std_dict = {}, {}
    count = 0
    for spec in spectra_order:
        for el in spec_block_order[spec]:
            spec_dict[spec, el] = vec[count * n_bins:(count + 1) * n_bins]
            std_dict[spec, el] = np.sqrt(var[count * n_bins:(count + 1) * n_bins])
            count += 1

    return bin_c, spec_dict, std_dict


def plot_P_matrix(P_mat,
                  cov_list_in,
                  cov_list_out,
                  file_name="test"):

    n_spec_in = len(cov_list_in)
    n_spec_out = len(cov_list_out)

    nbins = int(P_mat.shape[0]/n_spec_out)
    x_tick_loc = np.arange(n_spec_in) * nbins + nbins/2 - 1
    y_tick_loc = np.arange(n_spec_out) * nbins + nbins/2 - 1

    name_in = []
    for el in cov_list_in:
        name_in += [f"{el[0]} {el[1]}"]
    name_out = []
    for el in cov_list_out:
        name_out += [f"{el[0]} {el[1]}"]

    fig, ax = plt.subplots(figsize=(8, 12))
    plt.imshow(P_mat)
    plt.xticks(ticks=x_tick_loc, labels=name_in, rotation=90)
    plt.yticks(ticks=y_tick_loc, labels=name_out)
    plt.savefig(f"{file_name}.pdf")
    plt.clf()
    plt.close()
