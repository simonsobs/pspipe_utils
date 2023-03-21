import numpy as np
import pylab as plt
from pspy import so_cov, so_spectra
from pspy import pspy_utils
from itertools import combinations_with_replacement as cwr
from itertools import product


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
                           mc_full_cov,
                           only_diag_corrections = False):
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

    rescaling_var = np.where(mc_var>=an_var, mc_var, an_var)

    if only_diag_corrections:
        corrected_cov = an_full_cov - np.diag(an_var) + np.diag(rescaling_var)
    else:
        an_full_corr = so_cov.cov2corr(an_full_cov)
        corrected_cov = so_cov.corr2cov(an_full_corr, rescaling_var)

    return corrected_cov


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

def get_x_ar_to_x_freq_P_mat(freq_list, spec_name_list, nu_tag_list, binning_file, lmax):

    """
    The goal of this function is to build the passage matrix that will
    help combining the cross array spectra
    (e.g dr6&pa5_f150xdr6&pa5_f150, dr6&pa6_f150xdr6&pa6_f150, dr6&pa5_f150xdr6&pa6_f150)
    into cross frequency spectra (150x150)
    note that this function work for the "auto" case, i.e the TT, EE, BB cases
    another function take care of the "cross" TE, ...,

    Parameters
    ----------
    freq_list: list of str
        the frequency we consider
    spec_name_list: list of str
        list of the cross spectra
    nu_tag_list: list of tuple
        the effective frequency associated to each cross spectra
    binning_file: str
      a binning file with format bin low, bin high, bin mean
    lmax: int
      the maximum multipole to consider
    """

    n_freq = len(freq_list)
    n_ps = len(spec_name_list)
    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)
    n_cross_freq =  int(n_freq * (n_freq + 1) / 2)

    P_mat = np.zeros((n_cross_freq * n_bins, n_ps * n_bins))

    cross_freq_list = [f"{f0}x{f1}" for f0, f1 in cwr(freq_list, 2)]
    for c_id, cross_freq in enumerate(cross_freq_list):
        id_start_cf = n_bins * (c_id)
        id_stop_cf = n_bins * (c_id + 1)
        for ps_id, (nu_tag, ps) in enumerate(zip(nu_tag_list, spec_name_list)):
            na, nb = ps.split("x")
            nu_tag_a, nu_tag_b = nu_tag
            id_start_n = n_bins * (ps_id)
            id_stop_n =  n_bins * (ps_id + 1)
            if cross_freq in [f"{nu_tag_a}x{nu_tag_b}", f"{nu_tag_b}x{nu_tag_a}"]:
                P_mat[id_start_cf:id_stop_cf, id_start_n:id_stop_n] = np.identity(n_bins)

    return P_mat

def get_x_ar_to_x_freq_P_mat_cross(freq_list, spec_name_list_AB, nu_tag_list_AB, binning_file, lmax, char="&"):

    """
    The goal of this function is to build the passage matrix that will
    help combining the x_array spectra
    (e.g dr6&pa5_f150xdr6&pa5_f150, dr6&pa6_f150xdr6&pa6_f150, dr6&pa5_f150xdr6&pa6_f150)
    into cross frequency spectra (150x150)
    note that this function work for the " cross" case i.e TE, ...
    the function take into account that TE == ET in the case of the cross spectra
    between the same survey and array and therefore have been removed from the cov matrix
    The goal is for the final matrix to go from an TE-ET block
    to a TE only block, so we treat the remaining ET block as TE block with
    reverted nu_eff order (ET 90x150 -> TE 150x90)
    the TE block has n_freq ** 2 element instead of n_freq * (n_freq+1)/2 because
    TE 150x90 and TE 90x150 can have different fg model

    Parameters
    ----------
    freq_list: list of str
        the frequency we consider
    spec_name_list_AB: list of str
        list of the cross spectra corresponding to AB so for example TE
    nu_tag_list_AB: list of tuple
        the effective frequency associated to each cross spectra
    binning_file: str
      a binning file with format bin low, bin high, bin mean
    lmax: int
      the maximum multipole to consider
    """
    n_freq = len(freq_list)
    n_ps = len(spec_name_list_AB)
    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)

    n_cross_freq = n_freq ** 2
    cross_freq_list = [f"{f0}x{f1}" for f0, f1 in product(freq_list, freq_list)]

    # we need to remove ET spectrum with same sv same ar since TE == ET for these guys
    spec_name_list_BA = spec_name_list_AB.copy()
    nu_tag_list_BA = []
    n_ps_same = 0
    for nu_tag, ps in zip(nu_tag_list_AB, spec_name_list_AB):
        na, nb = ps.split("x")
        if (na == nb):
            spec_name_list_BA.remove(ps)
            n_ps_same += 1
        else:
            nu_tag_list_BA += [nu_tag[::-1]]

    spec_name_list = np.append(spec_name_list_AB, spec_name_list_BA)
    nu_tag_list = nu_tag_list_AB + nu_tag_list_BA

    n_ps = 2 * n_ps - n_ps_same
    P_mat_cross = np.zeros((n_cross_freq * n_bins, n_ps * n_bins))

    for c_id, cross_freq in enumerate(cross_freq_list):
        id_start_cf = n_bins * (c_id)
        id_stop_cf = n_bins * (c_id + 1)
        count = 0
        for nu_tag, ps in zip(nu_tag_list, spec_name_list):
            nu_tag_a, nu_tag_b = nu_tag
            spec_cf_list = [f"{nu_tag_a}x{nu_tag_b}"]
            id_start_n = n_bins * (count)
            id_stop_n =  n_bins * (count + 1)
            if cross_freq in spec_cf_list:
                P_mat_cross[id_start_cf:id_stop_cf, id_start_n:id_stop_n] = np.identity(n_bins)
            count += 1

    return P_mat_cross

def combine_P_mat(P_mat, P_mat_cross):

    """
    Cheap function to combine the different passage matrix of the TT - TE - EE spectra
    """
    shape_x, shape_y = P_mat.shape
    shape_cross_x, shape_cross_y = P_mat_cross.shape
    P = np.zeros((2 * shape_x + shape_cross_x, 2 * shape_y + shape_cross_y))
    P[:shape_x,:shape_y] = P_mat
    P[shape_x: shape_x + shape_cross_x, shape_y: shape_y + shape_cross_y] = P_mat_cross
    P[shape_x + shape_cross_x: 2 * shape_x + shape_cross_x, shape_y + shape_cross_y: 2 * shape_y + shape_cross_y] = P_mat
    return P

def read_x_ar_spectra_vec(spec_dir,
                          spec_name_list,
                          end_of_file,
                          spectra_order = ["TT", "TE", "ET", "EE"],
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
                         spectra_order = ["TT", "TE", "ET", "EE"]):

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
            Db = np.dot(Bbl, Dl[spec][:lmax + 2] + Dfl[spec][:lmax + 2])

            if (spec == "ET" or spec == "BT" or spec == "BE") & (na == nb): continue
            theory_vec = np.append(theory_vec, Db)

    return theory_vec


def get_max_likelihood_cov(P, inv_cov, check_pos_def = False, force_sim = False):

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

    cov_ml = np.linalg.inv(P @ inv_cov @ P.T)

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
    return cov_ml @ P @ inv_cov @ data_vec

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


def get_x_freq_to_final_P_mat(freq_list, binning_file, lmax):

    """
    The goal of this function is to build the passage matrix that will
    help combining the cross frequency spectra into final spectra
    the TT block is unchanged since it doesn't make sense to combine different TT together
    given the mismatch in fg for different cross frequency
    but it could make sense to combine TE and EE (at least for plotting) since foreground is not
    that important

    Parameters
    ----------
    freq_list: list of str
        the frequency we consider
    binning_file: str
      a binning file with format bin low, bin high, bin mean
    lmax: int
      the maximum multipole to consider
    """

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)

    n_freq = len(freq_list)
    n_cross_freq =  int(n_freq * (n_freq + 1) / 2)
    n_cross_freq_TE = n_freq ** 2

    P_mat_TT = np.zeros((n_cross_freq * n_bins, n_cross_freq * n_bins))
    P_mat_EE = np.zeros((n_bins, n_cross_freq * n_bins))
    P_mat_TE = np.zeros((n_bins, n_cross_freq_TE * n_bins))

    for i in range(n_cross_freq):
        P_mat_EE[:, n_bins * i:n_bins * (i + 1)] = np.identity(n_bins)
        P_mat_TT[n_bins * i:n_bins * (i + 1), n_bins * i:n_bins * (i + 1)] = np.identity(n_bins)
    for i in range(n_cross_freq_TE):
        P_mat_TE[:, n_bins * i:n_bins * (i + 1)] = np.identity(n_bins)

    s_TT_x, s_TT_y = P_mat_TT.shape
    s_EE_x, s_EE_y = P_mat_EE.shape
    s_TE_x, s_TE_y = P_mat_TE.shape

    P_mat = np.zeros((s_TT_x + s_TE_x + s_EE_x, s_TT_y + s_TE_y + s_EE_y))
    P_mat[:s_TT_x,:s_TT_y] = P_mat_TT
    P_mat[s_TT_x: s_TT_x + s_TE_x, s_TT_y: s_TT_y + s_TE_y] = P_mat_TE
    P_mat[s_TT_x + s_TE_x: s_TT_x + s_TE_x + s_EE_x, s_TT_y + s_TE_y: s_TT_y + s_TE_y + s_EE_y] = P_mat_EE

    return P_mat
