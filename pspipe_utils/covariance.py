import numpy as np
import pylab as plt
from pspy import pspy_utils, so_cov, so_spectra
from pixell import utils


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


def full_cov_to_cov_dict(full_cov,
                         spec_name_list,
                         n_bins,
                         spectra_order=["TT", "TE", "ET", "EE"]):

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

def correct_analytical_cov_skew(an_full_cov, mc_full_cov, nkeep=50, return_S=False):
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
    v  = np.diag(res_clean)
    res_clean = res_clean / (v[:,None] ** 0.5 * v[None,:] ** 0.5)
    corrected_cov = so_cov.corr2cov(res_clean, mc_var)

    if return_S:
        return S, corrected_cov
    else:
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

            idx = np.arange(nbins)[(lmin < bin_low) & (bin_high < lmax)]
            
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
