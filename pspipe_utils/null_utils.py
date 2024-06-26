import numpy as np
from scipy import interpolate

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

def get_null_vec_indices_and_bin_means(bin_out_dict):
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
        The data_idx_dict.
    """
    data_vec_indices_and_bin_means = get_data_vec_indices_and_bin_means(
        bin_out_dict
        )
    
    idx_dict = {}
    idx_start = 0
    for spec, array_cross_dict in data_vec_indices_and_bin_means.items():
        if spec not in idx_dict:
            idx_dict[spec] = {}

        array_crosses = list(array_cross_dict.keys())
        for i, array_cross1 in enumerate(array_crosses):
            _, bin_means1 = data_vec_indices_and_bin_means[spec][array_cross1]
            for array_cross2 in array_crosses[i+1:]:
                _, bin_means2 = data_vec_indices_and_bin_means[spec][array_cross2]
                
                bin_means = np.intersect1d(bin_means1, bin_means2)
                nbins = len(bin_means)

                idx_dict[spec][array_cross1, array_cross2] = (
                    np.arange(idx_start, idx_start + nbins), bin_means
                    )
                idx_start += nbins

    return idx_dict

def get_data_vec_to_null_vec_matrix(bin_out_dict):
    """Construct the full matrix that, when applied to a pspipe-ordered and cut
    data vector, generates the vector of all null tests. The data vector is 
    assumed to follow the ordering given by the output of
    get_data_vec_indices_and_bin_means, while the vector of null tests would
    follow the ordering given by the output of
    get_null_vec_indices_and_bin_means.

    Parameters
    ----------
    bin_out_dict : dict
        The first return value of psipe_utils.covariance.get_indices.

    Returns
    -------
    (n_null_test_bins, n_data_vec_bins) np.ndarray
        The 2-dimensional matrix taking the full data vector to the full vector
        of null tests.
    """
    data_vec_indices_and_bin_means = get_data_vec_indices_and_bin_means(
        bin_out_dict
        )
    null_vec_indices_and_bin_means = get_null_vec_indices_and_bin_means(
        bin_out_dict
        )

    # get the dimensions of the matrix
    dv_size = 0
    for spec, array_cross_dict in data_vec_indices_and_bin_means.items():
        for _, (idxs, _) in array_cross_dict.items():
            dv_size += len(idxs)

    nv_size = 0
    for spec, array_cross_null_dict in null_vec_indices_and_bin_means.items():
        for _, (idxs, _) in array_cross_null_dict.items():
            nv_size += len(idxs)

    # populate the matrix
    mat = np.zeros((nv_size, dv_size))
    for spec, array_cross_null_dict in null_vec_indices_and_bin_means.items():
        for (array_cross1, array_cross2), (row_idxs, row_bin_means) in array_cross_null_dict.items():
            col1_idxs, col1_bin_means = data_vec_indices_and_bin_means[spec][array_cross1]
            col2_idxs, col2_bin_means = data_vec_indices_and_bin_means[spec][array_cross2]

            col1_to_row = np.zeros((len(row_idxs), len(col1_idxs)))
            common1_bin_means, row1_where, col1_where = np.intersect1d(
                row_bin_means, col1_bin_means, return_indices=True
                )
            assert np.all(common1_bin_means == row_bin_means)
            assert np.all(row1_where == np.arange(len(row_idxs)))
            col1_to_row[:, col1_where] = np.eye(len(col1_where))

            col2_to_row = np.zeros((len(row_idxs), len(col2_idxs)))
            common2_bin_means, row2_where, col2_where = np.intersect1d(
                row_bin_means, col2_bin_means, return_indices=True
                )
            assert np.all(common2_bin_means == row_bin_means)
            assert np.all(row2_where == np.arange(len(row_idxs)))
            col2_to_row[:, col2_where] = np.eye(len(col2_where))

            mat[np.ix_(row_idxs, col1_idxs)] = col1_to_row
            mat[np.ix_(row_idxs, col2_idxs)] = -col2_to_row

    return mat

def get_th_vec_from_th_vec_dict(bin_out_dict, th_vec_dict, bin_mean):
    """Construct a theory vector corresponding to the data vector
    specified by the output of get_data_vec_indices_and_bin_means. The
    theory vector must already be binned and specified by a two-level
    theory-vector-indexing-dictionary with the following form:

    th_vec_dict[spec][array_cross] = binned_th_vec_values

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
    bin_mean : iterable of scalar
        The central values of the bins for all the spectra.

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

# forward model funcs
def splines2gammas(x, fields, knots, spline_bin_mean):
    """Project cubic spline knot points over the central bin values given by
    spline_bin_mean.

    Parameters
    ----------
    x : (..., len(fields) * len(knots)) np.ndarray
        The spline values at the knot locations for each sampled field. These
        are the parameters that are sampled. May have any prepended shape, e.g.
        if the sampler has many parallel walkers.
    fields : iterable of two-tuple
        Iterable of (spec, array) pairs, tracking which splines are actually
        being sampled. See note for meaning 'spec' in this context.
    knots : iterable of scalar
        Knot locations in bin_mean space, must be strictly increasing.
    spline_bin_mean : iterable of scalar
        Output locations in bin_mean space to evaluate the spline.

    Returns
    -------
    (..., len(fields), len(spline_bin_mean)) np.ndarray
        The evaluated spline for each field at the spline_bin_mean locations.
        Same prepended shape as x.

    Notes
    -----
    Here, we are hijacking the meaning of 'spec' to refer to the residual
    systematic terms. So 'TT' really refers to the 'delta_T' systematic, 'TE'
    really refers to the 'gamma' systematic, and 'EE' really refers to the 
    'delta_E' systematic. This is done purely for convenience: if we have
    'TT', 'TE', 'EE' spectra in our data vector, we know we will need to
    sample 'delta_T', 'gamma', and 'delta_E' at the alm-level.
    """
    _x = x.reshape(-1, len(fields), len(knots))
    gamma_func = interpolate.CubicSpline(knots, _x, axis=-1, bc_type='natural',
                                         extrapolate=False)
    gammas = gamma_func(spline_bin_mean)
    return gammas.reshape((*x.shape[:-1], len(fields), len(spline_bin_mean)))

def splines2th_vec(x, data_vec_dict, th_vec_dict, fields, knots,
                   spline_bin_mean, bin_mean):
    """Project cubic spline knot points over central bin values, and use the
    evaluated spline to modify a corresponding theory vector. The theory vector
    must already be binned and specified by a two-level
    theory-vector-indexing-dictionary with the following form:

    th_vec_dict[spec][array_cross] = binned_th_vec_values

    where spec is a polarization pair in canonical order (e.g., no 'ET' keys),
    array_cross is the '{array1}x{array2}' formatted cross of two arrays 
    corresponding to the polarization legs, and binned_th_vec_values is the
    theory for that spectrum at the locations given by bin_mean. The theory
    vector is assumed to not have any cuts applied, so it has a value at 
    every point in bin_mean.

    Parameters
    ----------
    x : (..., len(fields) * len(knots)) np.ndarray
        The spline values at the knot locations for each sampled field. These
        are the parameters that are sampled. May have any prepended shape, e.g.
        if the sampler has many parallel walkers.
    data_vec_dict : dict
        The output of get_data_vec_indices_and_bin_means.
    th_vec_dict : dict
        The theory-vector-indexing-dictionary.
    fields : iterable of two-tuple
        Iterable of (spec, array) pairs, tracking which splines are actually
        being sampled. See note for meaning 'spec' in this context.
    knots : iterable of scalar
        Knot locations in bin_mean space, must be strictly increasing.
    spline_bin_mean : iterable of scalar
        Output locations in bin_mean space to evaluate the spline.
    bin_mean : iterable of scalar
        The central values of the bins for all the spectra.

    Returns
    -------
    (..., n_data_vec_bins)
        The modified pspipe-ordered and cut theory vector, corresponding to the
        output of get_data_vec_indices_and_bin_means. Same prepended shape
        as x.
    """
    gammas = splines2gammas(x, fields, knots, spline_bin_mean) # ([num walkers,] num_fields, num_spline_bin_mean)
    gammas = gammas.reshape(-1, len(fields), len(spline_bin_mean)) # (num walkers, num_fields, num_spline_bin_mean)

    # need to go from bin_mean to spline_bin_mean in order
    # to apply gammas to theory. like theory[in_where]
    common_bin_mean, spl_where, in_where = np.intersect1d(
        spline_bin_mean, bin_mean, return_indices=True
        )
    assert np.all(common_bin_mean == spline_bin_mean)
    assert np.all(spl_where == np.arange(len(spline_bin_mean)))

    # get the dimensions of the vector
    dv_size = 0
    for spec, array_cross_dict in data_vec_dict.items():
        for _, (idxs, _) in array_cross_dict.items():
            dv_size += len(idxs)

    out = np.zeros((gammas.shape[0], dv_size)) # (num walkers, dv_size)
    for spec, array_cross_dict in data_vec_dict.items():
        for array_cross, (idxs, bin_means) in array_cross_dict.items():
            na, nb = array_cross.split("x")
            array_cross_ba = 'x'.join((nb, na))
            
            # need to go from spline_bin_mean to this specific spectrum's
            # bin mean in order to construct the data vector here. like
            # theory[spl_where]
            common_bin_mean, out_where, spl_where = np.intersect1d(
                bin_means, spline_bin_mean, return_indices=True
                )
            assert np.all(common_bin_mean == bin_means)
            assert np.all(out_where == np.arange(len(bin_means)))

            if spec == 'TT':
                TT_ab = th_vec_dict['TT'][array_cross][in_where]
                
                try:
                    delta_T_a = gammas[..., fields.index(('TT', na)), :]
                except ValueError:
                    delta_T_a = 0

                try:
                    delta_T_b = gammas[..., fields.index(('TT', nb)), :]
                except ValueError:
                    delta_T_b = 0

                th_spl = (1+delta_T_a)*(1+delta_T_b)*TT_ab
            
            elif spec == 'TE':
                # because for TE the TT arrays might be backwards
                try:
                    TT_ab = th_vec_dict['TT'][array_cross][in_where]
                except KeyError:
                    TT_ab = th_vec_dict['TT'][array_cross_ba][in_where]
                
                TE_ab = th_vec_dict['TE'][array_cross][in_where]
                
                try:
                    delta_T_a = gammas[..., fields.index(('TT', na)), :]
                except ValueError:
                    delta_T_a = 0
                
                try:
                    gamma_b = gammas[..., fields.index(('TE', nb)), :]
                except ValueError:
                    gamma_b = 0

                try:
                    delta_E_b = gammas[..., fields.index(('EE', nb)), :]
                except ValueError:
                    delta_E_b = 0
                
                th_spl = (1+delta_T_a)*(1+delta_E_b)*TE_ab + (1+delta_T_a)*gamma_b*TT_ab
            
            elif spec == 'EE':
                TT_ab = th_vec_dict['TT'][array_cross][in_where]
                TE_ab = th_vec_dict['TE'][array_cross][in_where]
                TE_ba = th_vec_dict['TE'][array_cross_ba][in_where]
                EE_ab = th_vec_dict['EE'][array_cross][in_where]

                try:
                    gamma_a = gammas[..., fields.index(('TE', na)), :]
                except ValueError:
                    gamma_a = 0
                
                try:
                    gamma_b = gammas[..., fields.index(('TE', nb)), :]
                except ValueError:
                    gamma_b = 0
                
                try:
                    delta_E_a = gammas[..., fields.index(('EE', na)), :]
                except ValueError:
                    delta_E_a = 0

                try:
                    delta_E_b = gammas[..., fields.index(('EE', nb)), :]
                except ValueError:
                    delta_E_b = 0
                
                th_spl = (1+delta_E_a)*(1+delta_E_b)*EE_ab + gamma_a*(1+delta_E_b)*TE_ab + \
                         gamma_b*(1+delta_E_a)*TE_ba + gamma_a*gamma_b*TT_ab

            out[..., idxs] = th_spl[..., spl_where]

    return out.reshape((*x.shape[:-1], -1)) # ([num walkers,] dv_size)

# posterior funcs
def log_prob(x, data_vec, inv_cov, data_vec_dict, th_vec_dict, fields, knots,
             spline_bin_mean, bin_mean):
    """Evaluate the unnormalized log posterior for given cubic spline knot
    points, data vector and covariance, and theory vector (to be modified
    by the splines). There is a fixed prior on the knot values with sigma = 1;
    this is very wide.

    Parameters
    ----------
    x : (..., len(fields) * len(knots)) np.ndarray
        The spline values at the knot locations for each sampled field. These
        are the parameters that are sampled. May have any prepended shape, e.g.
        if the sampler has many parallel walkers.
    data_vec : (n_data_vec_bins) np.ndarray
        The fixed data vector.
    inv_cov : (n_data_vec_bins, n_data_vec_bins) np.ndarray
        The fixed inverse covariance matrix of the data vector.
    data_vec_dict : dict
        The output of get_data_vec_indices_and_bin_means.
    th_vec_dict : dict
        The theory-vector-indexing-dictionary.
    fields : iterable of two-tuple
        Iterable of (spec, array) pairs, tracking which splines are actually
        being sampled. See note for meaning 'spec' in this context.
    knots : iterable of scalar
        Knot locations in bin_mean space, must be strictly increasing.
    spline_bin_mean : iterable of scalar
        Output locations in bin_mean space to evaluate the spline.
    bin_mean : iterable of scalar
        The central values of the bins for all the spectra.

    Returns
    -------
    (...) np.ndarray
        Unnormalized log posterior values. Same prepended shape as x.
    """
    lp = -0.5 * np.sum(x**2, axis=-1) # fixed posterior with sigma=1 for each knot value
    th_vec = splines2th_vec(x, data_vec_dict, th_vec_dict, fields, knots,
                            spline_bin_mean, bin_mean)
    res = data_vec - th_vec
    return lp - 0.5 * np.einsum('...a,ab,...b->...', res, inv_cov,
                                res, optimize='greedy')