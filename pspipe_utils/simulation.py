"""
Some utility functions for the generation of simulations.
"""
import numpy as np
from mflike import theoryforge_MFLike as th_mflike
from pixell import curvedsky
from pspy import so_spectra


def get_noise_matrix_spin0and2(f_name_tmp, survey, arrays, lmax, nsplits, input_type="Dl"):
    
    """This function uses the measured noise power spectra
    and generate a three dimensional array of noise power spectra [n_arrays, n_arrays, lmax] for temperature
    and polarisation.
    The different entries ([i,j,:]) of the arrays contain the noise power spectra
    for the different array pairs.
    for example nl_array_t[0,0,:] =>  nl^{TT}_{ar_{0},ar_{0}),  nl_array_t[0,1,:] =>  nl^{TT}_{ar_{0},ar_{1})
    this allows to consider correlated noise between different arrays.
    Note the the function return noise power spectra in "Cl", so apply an extra correction if the input is "Dl"
    
    Parameters
    ----------
    f_name_tmp : string
      a template name of the noise power spectra
    survey : string
      the survey to consider
    arrays: 1d array of string
      the arrays we consider
    lmax: integer
      the maximum multipole for the noise power spectra
    n_splits: integer
      the number of data splits we want to simulate
      nl_per_split= nl * n_{splits}
    input_type: str
      "Cl" or "Dl"

    """
    
    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    n_arrays = len(arrays)
    nl_array_t = np.zeros((n_arrays, n_arrays, lmax))
    nl_array_pol = np.zeros((n_arrays, n_arrays, lmax))
    
    for c1, ar1 in enumerate(arrays):
        for c2, ar2 in enumerate(arrays):
            if c1 > c2 : continue
            
            l, nl = so_spectra.read_ps(f_name_tmp.format(ar1, ar2, survey), spectra=spectra)
            
            nl_t = nl["TT"][:lmax]
            nl_pol = (nl["EE"][:lmax] + nl["BB"][:lmax])/2
            l = l[:lmax]

            nl_array_t[c1, c2, :] = nl_t * nsplits
            nl_array_pol[c1, c2, :] = nl_pol * nsplits
            
            if input_type == "Dl":
                nl_array_t[c1, c2, :]  *= 2 * np.pi / (l * (l + 1))
                nl_array_pol[c1, c2, :]  *= 2 * np.pi / (l * (l + 1))

    for i in range(lmax):
        nl_array_t[:, :, i] = fill_sym_mat(nl_array_t[:, :, i])
        nl_array_pol[:, :, i] = fill_sym_mat(nl_array_pol[:, :, i])

    return l, nl_array_t, nl_array_pol




def get_foreground_matrix(fg_dir, all_freqs, lmax):

    """This function uses the best fit foreground power spectra
    and generate a three dimensional array of foregroung power spectra [nfreqs, nfreqs, lmax].
    The different entries ([i,j,:]) of the array contains the fg power spectra for the different
    frequency channel pairs.
    for example fl_array_T[0,0,:] =>  fl_{f_{0},f_{0}),  fl_array_T[0,1,:] =>  fl_{f_{0},f_{1})
    this allows to have correlated fg between different frequency channels.
    (Not that for now, no fg are including in pol)

    Parameters
    ----------
    fg_dir : string
      the folder containing the foreground power spectra
    all_freqs: 1d array of string
      the frequencies we consider
    lmax: integer
      the maximum multipole for the noise power spectra
    """

    nfreqs = len(all_freqs)
    fl_array = np.zeros((nfreqs, nfreqs, lmax))

    for c1, freq1 in enumerate(all_freqs):
        for c2, freq2 in enumerate(all_freqs):
            if c1 > c2 : continue

            l, fl_all = np.loadtxt("%s/fg_%sx%s_TT.dat"%(fg_dir, freq1, freq2), unpack=True)
            fl_all *=  2 * np.pi / (l * (l + 1))

            fl_array[c1, c2, 2:lmax] = fl_all[:lmax-2]

    for i in range(lmax):
        fl_array[:,:,i] = fill_sym_mat(fl_array[:,:,i])

    return l, fl_array


def get_foreground_dict(ell, frequencies, fg_components, fg_params, fg_norm=None):

    """This function computes the foreground power spectra for a given set of multipoles,
    foreground components and parameters.

    Parameters
    ----------
    ell : 1d array of float
      the multipole array
    frequencies: 1d array of float or string
      the frequencies we consider
    fg_components: dict
      the foreground components, one per spectrum mode for instance
      fg_components = {"tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
                       "te": ["radio", "dust"],
                       "ee": ["radio", "dust"]}
    fg_params: dict
      the foreground parameter values for instance
      fg_params = {"a_tSZ": 3.30, "a_kSZ": 1.60, "a_p": 6.90, "beta_p": 2.08, "a_c": 4.90,
                   "beta_c": 2.20, "a_s": 3.10, "a_gtt": 2.79, "a_gte": 0.36, "a_gee": 0.13,
                   "a_psee": 0.05, "a_pste": 0, "xi": 0.1, "T_d": 9.60}

    fg_norm: dict
      the foreground normalisation. By default, {"nu_0": 150.0, "ell_0": 3000, "T_CMB": 2.725}
    """

    ThFo = th_mflike.TheoryForge_MFLike()

    frequencies = np.asarray(frequencies, dtype=float)
    ThFo.bandint_freqs = frequencies
    ThFo.freqs = ThFo.bandint_freqs

    fg_norm = fg_norm or {"nu_0": 150.0, "ell_0": 3000, "T_CMB": 2.725}
    fg_model = {"normalisation": fg_norm, "components": fg_components}
    ThFo.foregrounds = fg_model
    ThFo._init_foreground_model()
    fg_dict = ThFo._get_foreground_model(ell=ell, freqs_order=frequencies,  **fg_params)


    # Let's add other foregrounds not available in mflike
    ell_0 = fg_norm["ell_0"]
    nu_0 = fg_norm["nu_0"]

    # Normalisation of radio sources
    ell_clp = ell * (ell + 1.0)
    ell_0clp = ell_0 * (ell_0 + 1.0)

    models = {}
    models["bb", "radio"] = fg_params["a_psbb"] * ThFo.radio(
        {"nu": frequencies, "nu_0": nu_0, "beta": -0.5 - 2.0},
        {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1},
    )
    models["bb", "dust"] = fg_params["a_gbb"] * ThFo.dust(
        {"nu": frequencies, "nu_0": nu_0, "temp": 19.6, "beta": 1.5},
        {"ell": ell, "ell_0": 500.0, "alpha": -0.4},
    )

    models["tb", "radio"] = fg_params["a_pstb"] * ThFo.radio(
        {"nu": frequencies, "nu_0": nu_0, "beta": -0.5 - 2.0},
        {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1},
    )
    models["tb", "dust"] = fg_params["a_gtb"] * ThFo.dust(
        {"nu": frequencies, "nu_0": nu_0, "temp": 19.6, "beta": 1.5},
        {"ell": ell, "ell_0": 500.0, "alpha": -0.4},
    )
    for c1, f1 in enumerate(frequencies):
        for c2, f2 in enumerate(frequencies):
            for s in ["bb", "tb"]:
                fg_dict[s, "all", f1, f2] = np.zeros(len(ell))
                for comp in fg_components[s]:
                    fg_dict[s, comp, f1, f2] = models[s, comp][c1, c2]
                    fg_dict[s, "all", f1, f2] += fg_dict[s, comp, f1, f2]

    # Add ET, BT, BE where ET[f1, f2] = TE[f2, f1]
    for c1, f1 in enumerate(frequencies):
        for c2, f2 in enumerate(frequencies):
            for s in ["te", "tb", "eb"]:
                s_r = s[::-1]
                fg_dict[s_r, "all", f1, f2] = np.zeros(len(ell))
                for comp in fg_components[s]:
                    fg_dict[s_r, comp, f1, f2] = fg_dict[s, comp, f2, f1]
                    fg_dict[s_r, "all", f1, f2] += fg_dict[s, comp, f2, f1]

    return fg_dict

def generate_noise_alms(nl_array_t, lmax, n_splits, ncomp, nl_array_pol=None, dtype=np.complex128):

    """This function generates the alms corresponding to the noise power spectra matrices
    nl_array_t, nl_array_pol. The function returns a dictionnary nlms["T", i].
    The entry of the dictionnary are for example nlms["T", i] where i is the index of the split.
    note that nlms["T", i] is a (narrays, size(alm)) array, it is the harmonic transform of
    the noise realisation for the different frequencies.

    Parameters
    ----------
    nl_array_t : 3d array [narrays, narrays, lmax]
      noise power spectra matrix for temperature data

    lmax : integer
      the maximum multipole for the noise power spectra
    n_splits: integer
      the number of data splits we want to simulate
    ncomp: interger
      the number of components
      ncomp = 3 if T,Q,U
      ncomp = 1 if T only
    nl_array_pol : 3d array [narrays, narrays, lmax]
      noise power spectra matrix for polarisation data
      (in use if ncomp==3)
    """

    nlms = {}
    if ncomp == 1:
        for k in range(n_splits):
            nlms[k] = curvedsky.rand_alm(nl_array_t,lmax=lmax, dtype=dtype)
    else:
        for k in range(n_splits):
            nlms["T", k] = curvedsky.rand_alm(nl_array_t, lmax=lmax, dtype=dtype)
            nlms["E", k] = curvedsky.rand_alm(nl_array_pol, lmax=lmax, dtype=dtype)
            nlms["B", k] = curvedsky.rand_alm(nl_array_pol, lmax=lmax, dtype=dtype)

    return nlms



def fill_sym_mat(mat):
    """Make a upper diagonal or lower diagonal matrix symmetric

    Parameters
    ----------
    mat : 2d array
    the matrix we want symmetric
    """
    return mat + mat.T - np.diag(mat.diagonal())
