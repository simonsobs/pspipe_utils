"""
Some utility functions for the generating best fit power spectrum.
"""
import numpy as np
from mflike import theoryforge as th_mflike
from pspy import pspy_utils, so_spectra

def cmb_dict_from_file(f_name_cmb, lmax, spectra, lmin=2):
    """
    create a cmb power spectrum dict from file

    Parameters
    __________
    f_name_cmb: string
      the name of the cmb power spectra file
    lmax: integer
      the maximum multipole to consider (not inclusive)
    spectra: list
      the list of spectra ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    lmin: integer
      the minimum multipole to consider
    """

    l_cmb, cmb_dict = so_spectra.read_ps(f_name_cmb, spectra=spectra)
    id_cmb = np.where((l_cmb >= lmin) & (l_cmb < lmax))
    for spec in spectra:
        cmb_dict[spec] = cmb_dict[spec][id_cmb]

    l_cmb = l_cmb[id_cmb]

    return l_cmb, cmb_dict


def fg_dict_from_files(f_name_fg, array_list, lmax, spectra, lmin=2, f_name_cmb=None):
    """
    create a fg power spectrum dict from files

    Parameters
    __________
    f_name_fg: string
      a template for the name of the fg power spectra files
    array_list: list
      list of all arrays
    spectra: list
      the list of spectra ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    lmax: integer
      the maximum multipole to consider (not inclusive)
    spectra: list
      the list of spectra ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    lmin: integer
      the minimum multipole to consider
    f_name_cmb: str
      optionnaly include the cmb
    """

    if f_name_cmb is not None:
        l_cmb, cmb_dict = cmb_dict_from_file(f_name_cmb, lmax, spectra, lmin)

    fg_dict = {}
    for i, ar1 in enumerate(array_list):
        for j, ar2 in enumerate(array_list):
            if i > j: array_tuple = (ar2, ar1)
            else: array_tuple = (ar1, ar2)

            l_fg, fg = so_spectra.read_ps(f_name_fg.format(*array_tuple), spectra=spectra)
            id_fg = np.where((l_fg >= lmin) & (l_fg < lmax))
            fg_dict[ar1, ar2] = {}
            for spec in spectra:
                if i > j:
                    spec = spec[::-1]
                fg_dict[ar1, ar2][spec] = fg[spec][id_fg]
                if f_name_cmb is not None:
                    fg_dict[ar1, ar2][spec] += cmb_dict[spec]

    l_fg = l_fg[id_fg]

    return l_fg, fg_dict


def noise_dict_from_files(f_name_noise, sv_list, arrays, lmax, spectra, n_splits=None, lmin=2):
    """
    create a noise power spectrum dict from files

    Parameters
    __________
    f_name_noise: string
        a template for the name of the noise power spectra files
    sv_list: list
        list of the surveys to consider
    arrays: dict
        dict with the different array to consider, key correspond to the different surveys
    lmax: integer
        the maximum multipole to consider (not inclusive)
    spectra: list
        the list of spectra ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    lmin: integer
        the minimum multipole to consider
    n_splits: dict
        the number of splits, this is useful if we want to consider the noise
        power spectrum per split rather than the average noise power spectrum

    """

    nl_dict = {}
    for sv in sv_list:
        for ar_a in arrays[sv]:
            for ar_b in arrays[sv]:
                l_noise, nl = so_spectra.read_ps(
                    f_name_noise.format(ar_a, ar_b, sv), spectra=spectra
                )
                id_noise = np.where((l_noise >= lmin) & (l_noise < lmax))
                nl_dict[sv, ar_a, ar_b] = {}
                for spec in spectra:
                    nl_dict[sv, ar_a, ar_b][spec] = nl[spec][id_noise]
                    if n_splits is not None:
                        nl_dict[sv, ar_a, ar_b][spec] *= n_splits[sv]

    l_noise = l_noise[id_noise]

    return l_noise, nl_dict


def beam_dict_from_files(f_name_beam, sv_list, arrays, lmax, lmin=2):
    """
    create a beam dict from files

    Parameters
    __________
    f_name_beam: string
        a template for the name of the beam files
    sv_list: list
        list of the surveys to consider
    arrays: dict
        dict with the different array to consider, key correspond to the different surveys
    lmax: integer
        the maximum multipole to consider (not inclusive)
    lmin: integer
        the minimum multipole to consider
    """

    bl_dict = {}
    for sv in sv_list:
        for ar in arrays[sv]:
            l_beam, bl = pspy_utils.read_beam_file(f_name_beam.format(sv, ar))
            id_beam = np.where((l_beam >= lmin) & (l_beam < lmax))
            bl_dict[sv, ar] = bl[id_beam]

    l_beam = l_beam[id_beam]

    return l_beam, bl_dict


def get_all_best_fit(spec_name_list, l_th, cmb_dict, fg_dict, spectra, nl_dict=None, bl_dict=None):
    """
    This function prepare all best fit corresponding to the spec_name_list.
    the ps_all_th and nl_all_th are in particular useful for the analytical covariance computation
    the expected format for spec_name is ar_a&sv_bxar_c&sv_d

    Parameters
    ----------
    spec_name_list : list
        a list of the name of all spectra we consider
    l_th : 1d array
        the array of multipole
    cmb_dict: dict
        the cmb ps (format is [spec]
    fg_dict: dict of dict
        the fg ps (format is [sv1_ar1,sv2_ar2][spec])
    nl_dict: dict of dict
        the noise ps (format is [sv, ar1, ar2][spec])
    bl_dict: dict
        the beam ps (format is [sv,ar])
    """

    ps_all_th, nl_all_th = {}, {}

    for spec_name in spec_name_list:
        na, nb = spec_name.split("x")
        if len(na.split("&")) == 2:
            sv_a, ar_a = na.split("&")
            sv_b, ar_b = nb.split("&")
            noise_key_a = ar_a
            noise_key_b = ar_b
        elif len(na.split("&")) == 3:
            sv_a, ar_a, split_a = na.split("&")
            sv_b, ar_b, split_b = nb.split("&")
            noise_key_a = f"{ar_a}_{split_a}"
            noise_key_b = f"{ar_b}_{split_b}"

        for spec in spectra:
            ps_all_th[na, nb, spec] = cmb_dict[spec] + fg_dict[f"{sv_a}_{ar_a}", f"{sv_b}_{ar_b}"][spec]

            if bl_dict is not None:
                ps_all_th[na, nb, spec] *= bl_dict[sv_a, ar_a] * bl_dict[sv_b, ar_b]

            ps_all_th[nb, na, spec] = ps_all_th[na, nb, spec]

            if nl_dict is not None:
                if sv_a == sv_b:
                    nl_all_th[na, nb, spec] = nl_dict[sv_a, noise_key_a, noise_key_b][spec]
                else:
                    nl_all_th[na, nb, spec] = 0

                nl_all_th[nb, na, spec] = nl_all_th[na, nb, spec]

    if nl_dict is not None:
        return l_th, ps_all_th, nl_all_th
    else:
        return l_th, ps_all_th


def get_foreground_dict(ell,
                        external_bandpass,
                        fg_components,
                        fg_params,
                        fg_norm=None,
                        band_shift_dict=None):
    """This function computes the foreground power spectra for a given set of multipoles,
    foreground components and parameters. It uses mflike, note that mflike do not
    support foreground in tb, and bb therefore we include it here.
    The foreground are given in term of Dl

    Parameters
    ----------
    ell : 1d array of float
        the multipole array
    external_bandpass: dict
        external bandpass for each wafer
        example : external_bandpass = {"pa4_f150": [nu_ghz, passband], ...}
    fg_components: dict
        the foreground components, one per spectrum mode for instance
        fg_components = {"tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
                        "te": ["radio", "dust"],
                        "ee": ["radio", "dust"],
                        "bb": ["radio", "dust"],
                        "tb": ["radio", "dust"],
                        "eb": []}
    fg_params: dict
        the foreground parameter values for instance
    fg_params = {
            "a_tSZ": 3.30,
            "a_kSZ": 1.60,
            "a_p": 6.90,
            "beta_p": 2.08,
            "a_c": 4.90,
            "beta_c": 2.20,
            "a_s": 3.10,
            "xi": 0.1,
            "T_d": 9.60,
            "a_gtt": 2.79,
            "a_gte": 0.36,
            "a_gtb": 0.36,
            "a_gee": 0.13,
            "a_gbb": 0.13,
            "a_psee": 0.05,
            "a_psbb": 0.05,
            "a_pste": 0,
            "a_pstb": 0}

    fg_norm: dict
        the foreground normalisation. By default, {"nu_0": 150.0, "ell_0": 3000, "T_CMB": 2.725}
    band_shift_dict: dict
        a dictionnary with bandpass shift parameter
    """

    ThFo = th_mflike.TheoryForge()

    # The following lines defines ThFo.bands and params to follow
    # MFLike conventions.
    ThFo.bands = {f"{k}_s0": {"nu": v[0], "bandpass": v[1]} for k, v in external_bandpass.items()}
    ThFo.experiments = external_bandpass.keys()
    band_shift_dict = band_shift_dict or {f"bandint_shift_{exp}": 0.0 for exp in ThFo.experiments}
    ThFo._bandpass_construction(**band_shift_dict)

    fg_norm = fg_norm or {"nu_0": 150.0, "ell_0": 3000, "T_CMB": 2.725}
    fg_model = {"normalisation": fg_norm, "components": fg_components}
    ThFo.foregrounds = fg_model
    ThFo._init_foreground_model()

    fg_dict = ThFo._get_foreground_model(ell=ell, **fg_params)

    # Let's add other foregrounds not available in mflike (BB and TB fg)
    ell_0 = fg_norm["ell_0"]
    nu_0 = fg_norm["nu_0"]

    # Normalisation of radio sources
    ell_clp = ell * (ell + 1.0)
    ell_0clp = ell_0 * (ell_0 + 1.0)

    models = {}
    models["bb", "radio"] = fg_params["a_psbb"] * ThFo.radio(
        {"nu": ThFo.bandint_freqs, "nu_0": nu_0, "beta": -0.5 - 2.0},
        {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1},
    )

    models["bb", "dust"] = fg_params["a_gbb"] * ThFo.dust(
        {"nu": ThFo.bandint_freqs, "nu_0": nu_0, "temp": 19.6, "beta": 1.5},
        {"ell": ell, "ell_0": 500.0, "alpha": -0.4},
    )

    models["tb", "radio"] = fg_params["a_pstb"] * ThFo.radio(
        {"nu": ThFo.bandint_freqs, "nu_0": nu_0, "beta": -0.5 - 2.0},
        {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1},
    )

    models["tb", "dust"] = fg_params["a_gtb"] * ThFo.dust(
        {"nu": ThFo.bandint_freqs, "nu_0": nu_0, "temp": 19.6, "beta": 1.5},
        {"ell": ell, "ell_0": 500.0, "alpha": -0.4},
    )
    for c1, f1 in enumerate(ThFo.experiments):
        for c2, f2 in enumerate(ThFo.experiments):
            for s in ["eb", "bb", "tb"]:
                fg_dict[s, "all", f1, f2] = np.zeros(len(ell))
                for comp in fg_components[s]:
                    fg_dict[s, comp, f1, f2] = models[s, comp][c1, c2]
                    fg_dict[s, "all", f1, f2] += fg_dict[s, comp, f1, f2]

    # Add ET, BT, BE where ET[f1, f2] = TE[f2, f1]
    for c1, f1 in enumerate(ThFo.experiments):
        for c2, f2 in enumerate(ThFo.experiments):
            for s in ["te", "tb", "eb"]:
                s_r = s[::-1]
                fg_dict[s_r, "all", f1, f2] = np.zeros(len(ell))
                for comp in fg_components[s]:
                    fg_dict[s_r, comp, f1, f2] = fg_dict[s, comp, f2, f1]
                    fg_dict[s_r, "all", f1, f2] += fg_dict[s, comp, f2, f1]

    return fg_dict
