import numpy as np
from scipy import interpolate

from itertools import combinations as comb
from itertools import combinations_with_replacement as comb_wr
from itertools import product
from functools import reduce

def is_equal_array_pair(array_pair, delimeter='x'):
    """Whether two arrays in an array pair are the same.

    Parameters
    ----------
    array_pair : str
        '{array1}{delimeter}{array2}' formatted pair of two arrays.
    delimeter : str, optional
        The string that separates the pair of arrays.
    
    Returns
    -------
    bool
        array1 == array2
    """
    na, nb = array_pair.split(delimeter)
    return na == nb

def is_equal_pol_pair(spec):
    """Whether two pols in a spec are the same.

    Parameters
    ----------
    spec : str
        '{pol1}{pol2}' formatted pair of two pols.

    Returns
    -------
    bool
        pol1 == pol2
    """
    pa, pb = spec
    return pa == pb

def get_data_vec_indices_into_sacc(specs, array_crosses, nbins, sacc_file,
                                   pspipe_indices=None):
    """Get an indexing array that, when used over a sacc data vector and
    covariance, will output the vector and covariane in the "pspipe-like"
    order convention.
    
    The "pspipe-like" order convention is "spec-major": for instance, if
    'specs' contains ['TT', 'TE', 'ET', 'EE'], then all unique 'TT' crosses
    between arrays are listed first, followed by all 'TE' crosses between
    arrays, and then all 'ET' crosses between arrays that are not equivalent
    to a 'TE' cross, and then all 'EE' crosses between arrays. 
    
    This function makes two assumptions: (1) the list of specs and
    array_crosses do not "cut" any data that is packaged in the sacc. If this
    is not the case, this function will raise an AssertionError. (2) The data
    in the sacc were built by pspipe_utils.io.port2sacc; this function assumes
    the tracer naming convention of that function.

    Parameters
    ----------
    specs : iterable
        Iterable of polarization pairs, e.g. ('TT', 'TE, 'TB', ...).
    array_crosses : iterable
        Iterable of array crosses in the following format: '{array1}x{array2}'.
        Together with specs, must iterate over exactly the spectra in the sacc;
        i.e., the sacc must contain exactly the spectra given by the outer
        product of specs and array_crosses, minus 'ET', 'BT', and 'BE'
        doublons.
    nbins : int
        The number of bins in each spectrum, must be the same for all spectra.
    sacc_file : sacc.sacc.Sacc
        The sacc DataSet object.
    pspipe_indices : array of integers, optional
        If given, the integers that select the desired elements of the data 
        vector and covariance in the pspipe order, by default None. These
        would be the second return value of the function
        psipe_utils.covariance.get_indices.

    Returns
    -------
    array of integers
        The integer indexing array that would give the desired data vector and 
        covariance in the pspipe ordering when applied to the contents of the 
        sacc file.

    Raises
    ------
    ValueError
        If a spectrum should contain data in the sacc but is empty.
    ValueError
        If a spectrum should not be in the sacc but the sacc has data for it.
    AssertionError
        If the total size of the uncut data vector implied by specs,
        array_crosses, and nbins does not match the size of the sacc data.

    Examples
    --------
    >>> # given s is a sacc.sacc.Sacc DataSet object
    >>> # and given all_indices selects the elements of a pspipe-ordered
    >>> # data vector that I want to keep
    >>> vec = s.mean
    >>> cov = s.covariance.covmat
    >>> sacc_indices = get_data_vec_indices_into_sacc(specs, array_crosses,
                                                      nbins, s, all_indices)
    >>> pspipe_ordered_and_cut_vec = vec[sacc_indices]
    >>> pspipe_ordered_and_cut_cov = cov[np.ix_(sacc_indices, sacc_indices)]
    """
    # assumes sacc file makes no cuts of any kind, given the specs and crosses!
    # that means specs, array_crosses, bins must also not omit anything

    # the following assumes the sacc was constructed using
    # pspipe_utils.io.port2sacc, because it assumes a tracer naming convention etc.

    dts = dict(TT='00', TE='0e', TB='0b', ET='0e', BT='0b', EE='ee', EB='eb', BE='be', BB='bb')
    ts = dict(T=0, E=2, B=2)

    idx_dict = {}
    for spec in specs:
        dt = dts[spec]
        ta, tb = ts[spec[0]], ts[spec[1]]
        
        for array_cross in array_crosses:
            na, nb = array_cross.split('x')
            indices = sacc_file.indices(f'cl_{dt}', (f'{na}_s{ta}', f'{nb}_s{tb}'), warn_empty=False)
            
            if len(indices) > 0:
                assert len(indices) == nbins, \
                    f'expected {nbins=} but got {len(indices)=}'
                
                idx_dict[spec, array_cross] = indices

            # check sacc file for bad entries
            should_be_empty = False
            if na == nb and spec in ['ET', 'BT', 'BE']:
                should_be_empty = True

            if not should_be_empty and len(indices) == 0:
                raise ValueError(f'{spec}  {array_cross} should not be empty but sacc file has 0 elements')
            if should_be_empty and len(indices) > 0:
                raise ValueError(f'{spec}  {array_cross} should be empty but sacc file has {len(indices)} elements')

    # check sacc file size against the inputs (ie, must give it all specs and array crosses)
    total_indices = 0
    for v in idx_dict.values():
        total_indices += len(v)
    assert total_indices == len(sacc_file.mean), \
        f'expected {total_indices=} but found {len(sacc_file.mean)=}'
    
    if pspipe_indices is None:
        pspipe_indices = np.arange(total_indices)
    
    # get the indices by iterating over the canonical pspipe order
    sacc_indices = []
    for spec in specs:
        for array_cross in array_crosses:
            na, nb = array_cross.split("x")
            if (na == nb) & (spec == "ET" or spec == "BT" or spec == "BE"):
                continue

            sacc_indices = np.append(sacc_indices, idx_dict[spec, array_cross])
    sacc_indices = sacc_indices.astype(int)
    
    return sacc_indices[pspipe_indices]

def get_data_vec_indices_and_bin_means(bin_out_dict):
    """Construct a two-level data-vector-indexing dictionary with the
    following structure:
    
    data_idx_dict[spec][array_cross] = (idxs, bin_means)

    where spec is a polarization pair in canonical order (e.g., no 'ET' keys),
    array_cross is the '{array1}x{array2}' formatted cross of two arrays 
    corresponding to the polarization legs, idxs are the locations within the
    pspipe-ordered and cut data vector for this spectrum, and bin_means are the
    corresponding central values of the bins for the spectrum.

    This function is essentially a wrapper around bin_out_dict that reformats
    its keys.

    Parameters
    ----------
    bin_out_dict : dict
        The first return value of psipe_utils.covariance.get_indices. 

    Returns
    -------
    dict
        The data_idx_dict.
    """
    idx_dict = {}
    for k, v in bin_out_dict.items():
        array_cross, spec = k # this is the bin_out_dict convention, flipped from canon

        if spec in ['ET', 'BT', 'BE']: # put in spec-major order
            spec = spec[::-1]
            na, nb = array_cross.split('x')
            array_cross = 'x'.join((nb, na))

        if spec not in idx_dict:
            idx_dict[spec] = {}
        
        idxs, bin_means = v
        
        idx_dict[spec][array_cross] = (idxs, bin_means)
   
    return idx_dict

def get_spectrum_null_vec_indices_and_bin_means(bin_out_dict):
    """Construct a two-level null-test-vector-indexing dictionary with the
    following structure:

    null_idx_dict[spec][array_cross1, array_cross2] = (idxs, bin_means)

    where spec is a polarization pair in canonical order (e.g., no 'ET' keys),
    array_cross1 and array_cross2 are the '{array1}x{array2}' formatted crosses
    of two pairs of arrays forming the null test of the spec, idxs are the
    locations within the vector of all null tests for this null test, and
    bin_means are the intersection of the bin_means for the two spectra
    entering the null test.

    Parameters
    ----------
    bin_out_dict : dict
        The first return value of psipe_utils.covariance.get_indices.

    Returns
    -------
    dict
        The null_idx_dict.

    Notes
    -----
    Given two pairs of fields (W, X), (Y, Z), a spectrum-level null test
    will be C(W, X) - C(Y, Z). Therefore, we can have W==X or Y==Z, but not
    (W, X)==(Y, Z). Likewise, spec(W, X) must be equal to spec(Y, Z).
    """
    data_vec_indices_and_bin_means = get_data_vec_indices_and_bin_means(
        bin_out_dict
        )
    
    idx_dict = {f'{spec}': {} for spec in data_vec_indices_and_bin_means}
    idx_start = 0
    for spec, array_cross_dict in data_vec_indices_and_bin_means.items():
        # comb without replacement to avoid (W, X)==(Y, Z), but otherwise
        # allow W==X and Y==Z. using the same array_crosses list enforces
        # spec(W, X) equals spec(Y, Z)
        for array_cross1, array_cross2 in comb(array_cross_dict.keys(), 2):
            _, bin_means1 = data_vec_indices_and_bin_means[spec][array_cross1]
            _, bin_means2 = data_vec_indices_and_bin_means[spec][array_cross2]
            
            bin_means = np.intersect1d(bin_means1, bin_means2)
            nbins = len(bin_means)

            idx_dict[spec][array_cross1, array_cross2] = (
                np.arange(idx_start, idx_start + nbins), bin_means
                )
            idx_start += nbins

    return idx_dict

def get_map_null_vec_indices_and_bin_means(bin_out_dict,
                                           auto_map_null_only=False):
    """Construct a two-level null-test-vector-indexing dictionary with the
    following structure:

    null_idx_dict[spec][array_diff1, array_diff2] = (idxs, bin_means)

    where spec is a polarization pair in canonical order (e.g., no 'ET' keys),
    array_diff1 and array_diff2 are the '{array1}-{array2}' formatted crosses
    of two pairs of arrays forming the null test of the spec, idxs are the
    locations within the vector of all null tests for this null test, and
    bin_means are the intersection of the bin_means for the four spectra
    entering the null test.

    Parameters
    ----------
    bin_out_dict : dict
        The first return value of psipe_utils.covariance.get_indices.
    auto_map_null_only : bool, optional
        Whether to only consider map-level null tests for which (W, X)==(Y, Z),
        by default False (see notes).

    Returns
    -------
    dict
        The null_idx_dict.

    Notes
    -----
    Given two pairs of fields (W, X), (Y, Z), a map-level null test
    will be C(W - X, Y - Z) = C(W, Y) - C(W, Z) - C(X, Y) + C(X, Z). Therefore,
    we can have (W, X)==(Y, Z), but not W==X or Y==Z. Likewise, we must have
    pol(W)==pol(X) and pol(Y)==pol(Z).
    """
    data_vec_indices_and_bin_means = get_data_vec_indices_and_bin_means(
        bin_out_dict
        )
    
    idx_dict = {f'{spec}': {} for spec in data_vec_indices_and_bin_means}
    idx_start = 0
    for spec, array_cross_dict in data_vec_indices_and_bin_means.items():
        # given all the array_crosses in the data vector for a given spec,
        # we want to construct all the unique pairs of unique arrays for 
        # each pol leg of the spec
        pol1, pol2 = spec
        pol1arrs = []
        pol2arrs = []
        for array_cross in array_cross_dict.keys():
            na, nb = array_cross.split('x')
            if na not in pol1arrs:
                pol1arrs.append(na)
            if nb not in pol2arrs:
                pol2arrs.append(nb)
        pol1arrs.sort()
        pol2arrs.sort()
        
        if pol1 == pol2:
            assert np.all(pol1arrs == pol2arrs), \
                f'{pol1=}={pol2=}, but {pol1arrs=}!={pol2arrs=}'

        # skip W==X case
        array_pairs1 = comb(pol1arrs, 2)

        # if pol1 = pol2, we'll reuse array_pairs1
        if pol1 != pol2:
            # skip Y==Z case
            array_pairs2 = comb(pol2arrs, 2)

        # with the list of unique pairs of unique arrays for each pol leg of
        # the spec, we can iterate over pairs of array pairs. there are 3 cases:
        # 1. if auto_map_null_only, array_diff2 = array_diff1
        # 2. otherwise, if pol2 == pol1, iteratate over combs with replacement
        # 3. otherwise, iterate over product of array_diff1 and array_diff2
        # NOTE: we allow (W, X)==(Y, Z) in all cases.
        if auto_map_null_only:
            # there is only one array_diff to consider
            array_pairs1 = list(array_pairs1) # exhaust the comb once
            iter = zip(array_pairs1, array_pairs1)
        elif pol1 == pol2:
            # index from i to allow (W, X)==(Y, Z)
            iter = comb_wr(array_pairs1, 2)
        else:
            iter = product(array_pairs1, array_pairs2)
        
        for (arrayW, arrayX), (arrayY, arrayZ) in iter:
            # get the bin_means of the four spectra involved in the null because
            # we need the common bins to form the null
            array_crossWY = f'{arrayW}x{arrayY}'
            array_crossWZ = f'{arrayW}x{arrayZ}'
            array_crossXY = f'{arrayX}x{arrayY}'
            array_crossXZ = f'{arrayX}x{arrayZ}'

            # if spec is auto, then array_cross might be flipped, and
            # we can freely try to flip it back. if, after flipping, the 
            # spectrum is still missing, or if spec is not auto, then it
            # was cut from the data selection for some reason (e.g., spec is 
            # 'TE' but the 'T' leg is from the 'only_TT_map_set') and we can't
            # form this null test
            array_crosses = [array_crossWY, array_crossWZ, array_crossXY, array_crossXZ]
            bin_means_per_array_cross = []
            for array_cross in array_crosses:
                try:
                    _, bin_means = data_vec_indices_and_bin_means[spec][array_cross]
                except KeyError as e:
                    if is_equal_pol_pair(spec):
                        array_cross = 'x'.join(array_cross.split('x')[::-1])
                        _, bin_means = data_vec_indices_and_bin_means[spec][array_cross]
                    else:
                        raise e
                bin_means_per_array_cross.append(bin_means)

            bin_means = reduce(np.intersect1d, bin_means_per_array_cross)
            nbins = len(bin_means)

            # put into readable key 
            array_diff1 = f'{arrayW}-{arrayX}'
            array_diff2 = f'{arrayY}-{arrayZ}'

            idx_dict[spec][array_diff1, array_diff2] = (
                np.arange(idx_start, idx_start + nbins), bin_means
                )
            idx_start += nbins

    return idx_dict

def _mat2cov_diagonalizing_mat(mat, cov):
    # only keep num=rank singular values
    u, sig, vh = np.linalg.svd(mat, full_matrices=False)
    rank = np.linalg.matrix_rank(mat, tol=1e-6)
    
    # svd gives singular values in descending order 
    _, o = np.linalg.eigh((np.diag(sig) @ vh @ cov @ vh.T @ np.diag(sig).T)[:rank, :rank])
    mat = o.T @ u[:, :rank].T @ mat
    return mat

def get_data_vec_to_spectrum_null_vec_matrix(bin_out_dict, cov=None):
    """Construct the full matrix that, when applied to a pspipe-ordered and cut
    data vector, generates the vector of all null tests. The data vector is 
    assumed to follow the ordering given by the output of
    get_data_vec_indices_and_bin_means, while the vector of null tests would
    follow the ordering given by the output of
    get_null_vec_indices_and_bin_means (unless cov is provided).

    Parameters
    ----------
    bin_out_dict : dict
        The first return value of psipe_utils.covariance.get_indices.
    cov : (n_data_vec_bins, n_data_vec_bins) np.ndarray, optional
        If provided, the returned matrix transforms the data vector into 
        a vector of independent null tests with respect to cov. In this case,
        the dimension of the null test vector will be less than that of the 
        data vector; specifically, it will be the rank of the matrix that is
        returned if cov is not provided.

    Returns
    -------
    (n_null_test_bins, n_data_vec_bins) np.ndarray
        The 2-dimensional matrix taking the full data vector to the vector of
        null tests.
    """
    data_vec_indices_and_bin_means = get_data_vec_indices_and_bin_means(
        bin_out_dict
        )
    null_vec_indices_and_bin_means = get_spectrum_null_vec_indices_and_bin_means(
        bin_out_dict
        )

    # get the dimensions of the matrix
    dv_size = 0
    for _, array_cross_dict in data_vec_indices_and_bin_means.items():
        for _, (idxs, _) in array_cross_dict.items():
            dv_size += len(idxs)

    nv_size = 0
    for _, array_cross_null_dict in null_vec_indices_and_bin_means.items():
        for _, (idxs, _) in array_cross_null_dict.items():
            nv_size += len(idxs)

    # populate the matrix
    mat = np.zeros((nv_size, dv_size))
    for spec, array_cross_null_dict in null_vec_indices_and_bin_means.items():
        for (array_cross1, array_cross2), (row_idxs, row_bin_means) in array_cross_null_dict.items():
            col1_idxs, col1_bin_means = data_vec_indices_and_bin_means[spec][array_cross1]
            col2_idxs, col2_bin_means = data_vec_indices_and_bin_means[spec][array_cross2]

            col_to_row_mats = []
            for col_idxs, col_bin_means in zip(
                [col1_idxs, col2_idxs],
                [col1_bin_means, col2_bin_means]
                ):

                col_to_row = np.zeros((len(row_idxs), len(col_idxs)))
                common_bin_means, row_where, col_where = np.intersect1d(
                    row_bin_means, col_bin_means, return_indices=True
                    )
                assert np.all(common_bin_means == row_bin_means)
                assert np.all(row_where == np.arange(len(row_idxs)))
                col_to_row[:, col_where] = np.eye(len(col_where))

                col_to_row_mats.append(col_to_row)
            col1_to_row, col2_to_row = col_to_row_mats

            mat[np.ix_(row_idxs, col1_idxs)] += col1_to_row
            mat[np.ix_(row_idxs, col2_idxs)] -= col2_to_row

    if cov is not None:
        return _mat2cov_diagonalizing_mat(mat, cov)
    else:
        return mat
    
def get_data_vec_to_map_null_vec_matrix(bin_out_dict, auto_map_null_only=False,
                                        cov=None):
    """Construct the full matrix that, when applied to a pspipe-ordered and cut
    data vector, generates the vector of all null tests. The data vector is 
    assumed to follow the ordering given by the output of
    get_data_vec_indices_and_bin_means, while the vector of null tests would
    follow the ordering given by the output of
    get_null_vec_indices_and_bin_means (unless cov is provided).

    Parameters
    ----------
    bin_out_dict : dict
        The first return value of psipe_utils.covariance.get_indices.
    cov : (n_data_vec_bins, n_data_vec_bins) np.ndarray, optional
        If provided, the returned matrix transforms the data vector into 
        a vector of independent null tests with respect to cov. In this case,
        the dimension of the null test vector will be less than that of the 
        data vector; specifically, it will be the rank of the matrix that is
        returned if cov is not provided.

    Returns
    -------
    (n_null_test_bins, n_data_vec_bins) np.ndarray
        The 2-dimensional matrix taking the full data vector to the vector of
        null tests.
    """
    data_vec_indices_and_bin_means = get_data_vec_indices_and_bin_means(
        bin_out_dict
        )
    null_vec_indices_and_bin_means = get_map_null_vec_indices_and_bin_means(
        bin_out_dict, auto_map_null_only=auto_map_null_only
        )

    # get the dimensions of the matrix
    dv_size = 0
    for _, array_cross_dict in data_vec_indices_and_bin_means.items():
        for _, (idxs, _) in array_cross_dict.items():
            dv_size += len(idxs)

    nv_size = 0
    for _, array_diff_null_dict in null_vec_indices_and_bin_means.items():
        for _, (idxs, _) in array_diff_null_dict.items():
            nv_size += len(idxs)

    # populate the matrix
    mat = np.zeros((nv_size, dv_size))
    for spec, array_diff_null_dict in null_vec_indices_and_bin_means.items():
        for (array_diff1, array_diff2), (row_idxs, row_bin_means) in array_diff_null_dict.items():
            # for this null test, what data do I need?
            arrayW, arrayX = array_diff1.split('-')
            arrayY, arrayZ = array_diff2.split('-')

            array_crossWY = f'{arrayW}x{arrayY}'
            array_crossWZ = f'{arrayW}x{arrayZ}'
            array_crossXY = f'{arrayX}x{arrayY}'
            array_crossXZ = f'{arrayX}x{arrayZ}'

            array_crosses = [array_crossWY, array_crossWZ, array_crossXY, array_crossXZ]
            col_idxs_per_array_cross = []
            col_bin_means_per_array_cross = []
            for array_cross in array_crosses:
                try:
                    col_idxs, col_bin_means = data_vec_indices_and_bin_means[spec][array_cross]
                except KeyError:
                    assert is_equal_pol_pair(spec), \
                        f'{spec=}, {array_cross=} not in data_vec_indices_and_bin_means'
                    array_cross = 'x'.join(array_cross.split('x')[::-1])
                    col_idxs, col_bin_means = data_vec_indices_and_bin_means[spec][array_cross]
                col_idxs_per_array_cross.append(col_idxs)
                col_bin_means_per_array_cross.append(col_bin_means)
            colWY_idxs, colWZ_idxs, colXY_idxs, colXZ_idxs = col_idxs_per_array_cross

            col_to_row_mats = []
            for col_idxs, col_bin_means in zip(
                col_idxs_per_array_cross,
                col_bin_means_per_array_cross
                ):

                col_to_row = np.zeros((len(row_idxs), len(col_idxs)))
                common_bin_means, row_where, col_where = np.intersect1d(
                    row_bin_means, col_bin_means, return_indices=True
                    )
                assert np.all(common_bin_means == row_bin_means)
                assert np.all(row_where == np.arange(len(row_idxs)))
                col_to_row[:, col_where] = np.eye(len(col_where))

                col_to_row_mats.append(col_to_row)
            colWY_to_row, colWZ_to_row, colXY_to_row, colXZ_to_row = col_to_row_mats

            mat[np.ix_(row_idxs, colWY_idxs)] += colWY_to_row
            mat[np.ix_(row_idxs, colWZ_idxs)] -= colWZ_to_row
            mat[np.ix_(row_idxs, colXY_idxs)] -= colXY_to_row
            mat[np.ix_(row_idxs, colXZ_idxs)] += colXZ_to_row

    if cov is not None:
        return _mat2cov_diagonalizing_mat(mat, cov)
    else:
        return mat

def get_th_vec_from_th_vec_dict(bin_out_dict, th_vec_dict):
    """Construct a theory vector corresponding to the data vector
    specified by the output of get_data_vec_indices_and_bin_means. The
    theory vector must already be binned and specified by a two-level
    theory-vector-indexing-dictionary with the following form:

    th_vec_dict[spec][array_cross] = binned_th_vec_values
    th_vec_dict['bin_mean'] = bin_mean

    where spec is a polarization pair in canonical order (e.g., no 'ET' keys),
    array_cross is the '{array1}x{array2}' formatted cross of two arrays 
    corresponding to the polarization legs, and binned_th_vec_values is the
    theory for that spectrum at the locations given by bin_mean. The theory
    vector is assumed to not have any cuts applied, so it has a value at 
    every point in bin_mean.

    Parameters
    ----------
    bin_out_dict : dict
        The first return value of psipe_utils.covariance.get_indices.
    th_vec_dict : dict
        The theory-vector-indexing-dictionary.

    Returns
    -------
    (n_data_vec_bins) np.ndarray
        The pspipe-ordered and cut theory vector, corresponding to the output
        of get_data_vec_indices_and_bin_means.
    """
    data_vec_indices_and_bin_means = get_data_vec_indices_and_bin_means(
        bin_out_dict
        )

    # get the dimensions of the vector
    dv_size = 0
    for spec, array_cross_dict in data_vec_indices_and_bin_means.items():
        for _, (idxs, _) in array_cross_dict.items():
            dv_size += len(idxs)

    bin_mean = th_vec_dict['bin_mean']
    
    # populate the vec
    vec = np.zeros(dv_size)
    for spec, array_cross_dict in data_vec_indices_and_bin_means.items():
        for array_cross, (idxs, out_bin_means) in array_cross_dict.items():
            th_vec = th_vec_dict[spec][array_cross]

            common_bin_means, out_where, in_where = np.intersect1d(
                out_bin_means, bin_mean, return_indices=True
                )
            assert np.all(common_bin_means == out_bin_means)
            assert np.all(out_where == np.arange(len(out_bin_means)))

            vec[idxs] = th_vec[in_where]

    return vec