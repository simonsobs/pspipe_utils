"""
Some utility functions for the kspace filter.
"""
from itertools import combinations_with_replacement as cwr
from pspy import so_spectra, so_map, so_map_preprocessing, pspy_utils
import numpy as np

def build_kspace_filter_matrix(lb, ps_sims, n_sims, spectra, return_dict=False):

    """This function compute the kspace filter transfer matrix using
    a bunch of simulations,
    The matrix acts on unfiltered power spectra and return filtered power spectra.
    We will use the inverse of the matrix to remove the biais from the kspace filter.
    the elements that we estimate explicitely are
     "TT_to_TT", "EE_to_EE", "BB_to_BB", "EE_to_BB", "BB_to_EE"
     
     ps_filt_x = \sum_y x_to_y ps_unfilt_y
     
     we then set "TE_to_TE", "ET_to_ET", "TB_to_TB", "BT_to_BT" to sqrt(TT_to_TT * EE_to_EE)
     and "EB_to_EB", "BE_to_BE" to "EE_to_EE"
     
    Parameters
    ----------
    lb : 1d array
        the binned multipoles
    ps_sims: dict
        a dictionnary with all simulated power spectrum, form should be
        ps[[key_a, key_b][spec]
        key_a is "filter" or "nofilter"
        key_b is "standard", "noE", or "noB"
        spec is the spectra list ["TT","TE".....]
    n_sims: integer
        the number of simulations
    spectra: list of str
        the spectra list ["TT","TE".....]
    return dict: boolean
        wether to return a dictionnary with the term of the matrix in
        addition to the matrix
    """

    n_bins = len(lb)
    kspace_matrix = np.zeros((9 * n_bins, 9 * n_bins))
    kspace_dict, std = {}, {}

    for spec1 in spectra:
        for spec2 in spectra:
            kspace_dict[f"{spec1}_to_{spec2}"] = np.zeros(n_bins)
        
    elements = ["TT_to_TT", "EE_to_EE", "BB_to_BB", "EE_to_BB", "BB_to_EE"]
    for el in elements: kspace_dict[el] = []
    for i in range(n_sims):
        kspace_dict["TT_to_TT"] += [ps_sims["filter", "standard"][i]["TT"]/ps_sims["nofilter", "standard"][i]["TT"]]
        
        kspace_dict["EE_to_EE"] += [ps_sims["filter", "noB"][i]["EE"]/ps_sims["nofilter", "noB"][i]["EE"]]
        kspace_dict["BB_to_BB"] += [ps_sims["filter", "noE"][i]["BB"]/ps_sims["nofilter", "noE"][i]["BB"]]
        
        kspace_dict["EE_to_BB"] += [ps_sims["filter", "noB"][i]["BB"]/ps_sims["nofilter", "noB"][i]["EE"]]
        kspace_dict["BB_to_EE"] += [ps_sims["filter", "noE"][i]["EE"]/ps_sims["nofilter", "noE"][i]["BB"]]
 
    for el in elements:
        std[el] = np.std(kspace_dict[el], axis=0)
        kspace_dict[el] = np.mean(kspace_dict[el], axis=0)

    elements = ["TE_to_TE", "ET_to_ET", "TB_to_TB", "BT_to_BT"]
    for el in elements:
        kspace_dict[el] = np.sqrt(kspace_dict["TT_to_TT"] * kspace_dict["EE_to_EE"])
        
    elements = ["EB_to_EB", "BE_to_BE"]
    for el in elements:
        kspace_dict[el] = kspace_dict["EE_to_EE"]

    for i, spec1 in enumerate(spectra):
        for j, spec2 in enumerate(spectra):
            for k in range(n_bins):
                kspace_matrix[k + i * n_bins, k + j * n_bins] = kspace_dict[f"{spec1}_to_{spec2}"][k]

    if return_dict:
        return kspace_dict, std, kspace_matrix
    else:
        return kspace_matrix

def build_analytic_kspace_filter_diag(sv1, sv2, lmax, templates, filter_dicts,
                                      dtype=np.float64, binning_file=None):
    """Get the analytic kspace filter (binned or unbinned) for two surveys,
    given a dictionary of survey geometries and their filters.

    Parameters
    ----------
    sv1 : str
        The first survey.
    sv2 : str
        The second survey
    lmax : int
        Maximum multipole of the filter.
    templates : dict
        A dictionary for each survey pointing to an so_map.so_map object giving
        the geometry of that survey.
    filter_dicts : dict
        A dictionary pointing to another dictionary of kspace filter info. Keys
        are surveys, but does not have to be every survey (equivalent to 
        filter_dicts[sv] = None if sv not in filter_dicts).
    dtype : np.dtype, optional
        dtype of output array, by default np.float64.
    binning_file : path-like, optional
        If supplied, apply this binning with 2l+1 weights, by default None.

    Returns
    -------
    (x, tf)
        The ell (or binned-ell) points, and the value of the function at those 
        points.
    """
    geometry_sv1 = templates[sv1].data.geometry
    geometry_sv2 = templates[sv2].data.geometry
    geometries = (geometry_sv1, geometry_sv2)
    
    vk_masks = []
    hk_masks = []
    geometries_to_filt = [] 
    for i, filter_dict_sv in enumerate((filter_dicts.get(sv1), filter_dicts.get(sv2))):
        if filter_dict_sv is not None:
            assert filter_dict_sv["type"] == "binary_cross", \
                f'filter must be binary cross, got {filter_dict_sv["type"]}'
            
            vk_masks.append(filter_dict_sv.get('vk_mask'))
            hk_masks.append(filter_dict_sv.get('hk_mask'))
            geometries_to_filt.append(i)

    return so_map_preprocessing.analytical_std_tf(
        lmax, vk_masks=vk_masks, hk_masks=hk_masks, geometries=geometries,
        geometries_to_filt=geometries_to_filt, dtype=dtype, 
        binning_file=binning_file
        )    

def build_analytic_kspace_filter_matrices(surveys, arrays, templates, filter_dicts, binning_file, lmax,
                                          method='new'):
    """This function compute the analytical kspace filter transfer matrices
    
    Parameters
    ----------
    surveys : list
        a list of the survey
    arrays: dict
        a dictionnary with entry "survey" that list the
        different arrays included in the given survey
    templates: dict
        a dictionnary with entry "survey" that contains
        a so_map template corresponding to the given survey
    filter_dict: dict
        a dictionnary that contains the filter properties
        e.g filter_dict = {..., "type":"binary_cross","vk_mask":[-90, 90], "hk_mask":[-50, 50], ...}
    binning_file: data file
      a binning file with format bin low, bin high, bin mean
    lmax: int
        the maximum multipole to consider
    """    
    transfer_func = {}
    if method == 'new':
        for sv1, sv2 in cwr(surveys, 2):
            _, tfb = build_analytic_kspace_filter_diag(sv1, sv2, lmax, templates, filter_dicts, binning_file)
            transfer_func[sv1, sv2] = tfb       
    
    if method == 'old':
        kf_tfs = {}
        for sv in surveys:                
            filter_sv = get_kspace_filter(templates[sv], filter_dicts[sv])
            _, kf_tfs[sv] = so_map_preprocessing.analytical_tf(templates[sv], filter_sv, binning_file, lmax)

        for sv1 in surveys:
            for sv2 in surveys:
                transfer_func[sv1, sv2] = np.minimum(kf_tfs[sv1], kf_tfs[sv2])

    transfer_mat = {}
    mapnames = [f'{sv}_{m}' for sv in surveys for m in arrays[sv]]
    for mapname1, mapname2 in cwr(mapnames, 2): # like spec_name_list
        sv1 = mapname1.split('_')[0]
        sv2 = mapname2.split('_')[0]
        diag = np.tile(transfer_func[sv1, sv2], 9)
        transfer_mat[f"{mapname1}x{mapname2}"] = np.diag(diag)

    return transfer_mat


def deconvolve_kspace_filter_matrix(lb, ps, kspace_filter_matrix, spectra, xtra_corr=None):

    """This function deconvolve the kspace filter transfer matrix
     
     since
     ps_filt_x = \sum_y M_{x_to_y} ps_unfilt_y
     we have
     ps_unfilt_x = \sum_y (M)**-1_{x_to_y} ps_filt_y
     
     
    Parameters
    ----------
    lb : 1d array
        the binned multipoles
    ps: dict
        a dictionnary with the spectra
        ps[spec]
        spec is the spectra list ["TT","TE".....]
    kspace_filter_matrix: 2d array
        a 9 * n_bins, 9 * n_bins matrix that encode the effect of the kspace filter
    spectra: list of str
        the spectra list ["TT","TE".....]
    xtra_corr: a dictionnary with spectra
        this term account for an xtra correction for the effect of kspace filter
        in particular, tf_TE is not perfectly equal to sqrt(tf_TT * tf_EE)
        so we might want to correct for this
    """

    n_bins = len(lb)

    inv_kspace_mat = np.linalg.inv(kspace_filter_matrix)
    vec = []
    for f in spectra:
        vec = np.append(vec, ps[f])
    vec = np.dot(inv_kspace_mat, vec)
    ps = so_spectra.vec2spec_dict(n_bins, vec, spectra)

    if xtra_corr is not None:
        for f in spectra:
            ps[f] -= xtra_corr[f]
    return lb, ps


def filter_map(map, filter, window, inv_pixwin=None, weighted_filter=False, tol=1e-4, ref=0.9, use_ducc_rfft=False):

    """Filter the map in Fourier space using a predefined filter. Note that we mutliply the maps by a window  before
    doing this operation in order to remove pathological pixels
    We also include an option for removing the pixel window function

    Parameters
    ---------
    map: ``so_map``
        the map to be filtered
    filter: 2d array
        a filter applied in fourier space
    window:  ``so_map``
        a window removing pathological pixels
    inv_pixwin: 2d array
        the inverse of the pixel window function in fourier space
    weighted_filter: boolean
        wether to use weighted filter a la sigurd
    tol, ref: floats
        only in use in the case of the weighted filter, these arg
        remove crazy pixels value in the weight applied
    use_ducc_rfft: boolean
        wether to use ducc real fft instead of enmap complex fft

    """

    if weighted_filter == False:
        if inv_pixwin is not None:
            map = so_map.fourier_convolution(map, filter * inv_pixwin, window, use_ducc_rfft=use_ducc_rfft)
        else:
            map = so_map.fourier_convolution(map, filter, window, use_ducc_rfft=use_ducc_rfft)

    else:
    
        if use_ducc_rfft == True:
            print("ducc fft not implemented for weighted filter")
        map.data *= window.data
        one_mf = (1 - filter)
        rhs    = enmap.ifft(one_mf * enmap.fft(map.data, normalize=True), normalize=True).real
        div    = enmap.ifft(one_mf * enmap.fft(window.data, normalize=True), normalize=True).real
        del one_mf
        div    = np.maximum(div, np.percentile(window.data[::10, ::10], ref * 100) * tol)
        map.data -= rhs / div
        del rhs
        del div
    
        if inv_pixwin is not None:
            ft = enmap.fft(map.data, normalize=True)
            ft  *= inv_pixwin
            map.data = enmap.ifft(ft, normalize=True).real

    return map


def get_kspace_filter(template, filter_dict, dtype=np.float64):

    """build the kspace filter according to a dictionnary specifying the filter parameters
    Parameters
    ---------
    template: ``so_map``
        a template of the CAR map we want to filter
    filter_dict: dict
        a dictionnary that contains the filter properties
        e.g filter_dict = {..., "type":"binary_cross","vk_mask":[-90, 90], "hk_mask":[-50, 50], ...}
        
    """

    shape, wcs = template.data.shape, template.data.wcs
    if filter_dict["type"] == "binary_cross":
        filter = so_map_preprocessing.build_std_filter(shape, wcs, vk_mask=filter_dict["vk_mask"], hk_mask=filter_dict["hk_mask"], dtype=dtype)
    elif filter_dict["type"] == "gauss":
        filter = so_map_preprocessing.build_sigurd_filter(shape, wcs, filter_dict["lbounds"], dtype=dtype)
    else:
        print("you need to specify a valid filter type")

    return filter
