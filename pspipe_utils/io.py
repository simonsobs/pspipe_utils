import importlib
import logging
import os
from copy import deepcopy

import numpy as np
import sacc
from pspy import pspy_utils


def port2sacc(
    data_vec,
    cov,
    cov_order,
    binning_file,
    lmax,
    bbls=None,
    passbands=None,
    beams=None,
    metadata=None,
    sacc_file_name="data_sacc.fits",
    log=None,
):
    """
    This function computes the chi2 value between data/sim spectra wrt theory spectra given
    the data covariance and a set of multipole cuts

    Parameters
    ----------
    data_vec: 1d array
      the flatten vector holding the data spectra
    cov: 2d array
      the covariance matrix
    cov_order: tuple
      the order of the covariance matrix e.g  [("TT", "sv_ar1xsv_ar1"), ("TT", "sv_ar1xsv_ar2")...]
    binning_file: str
      a binning file with format bin low, bin high, bin mean
    lmax: int
      the maximum multipole to consider for binning
    bbls: dict of bbl
      the dictionary indexed on cross spectrum name holding the bbl
    passbands: dict of (frequencies, passbands)
      the dictionary indexed on array detector holding the couple of (frequencies, passbands) values
    beams: dict of beam
      the dictionary indexed on array detector holding the beam information
    metadata: dict
      the metadata to be stored within sacc file
    sacc_file_name: path
      the sacc file name
    log: logger
      the logger to print message (default gets back to logging library with debug level)
    """

    passbands = passbands or {}
    beams = beams or {}
    metadata = metadata or {}

    if not log:
        log = logging.getLogger(__name__)
        log.setLevel(logging.DEBUG)

    bin_low, bin_high, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(lb)
    # Sanity check
    if n_bins != len(data_vec) // len(cov_order):
        raise ValueError("Number of bins does not match the data size and the cov order values!")

    # Compute unique list of (survey, array)
    map_set_list = set(sum([cross.split("x") for spec, cross, *_ in cov_order], []))
    log.debug(f"Survey list : {map_set_list} \n")

    # Saving into sacc format
    s = sacc.Sacc()
    for map_set in map_set_list:
        for spin, quantity in zip([0, 2], ["temperature", "polarization"]):
            nus, passband = passbands.get(f"{map_set}", ([], []))
            ell, beam = beams.get(f"{map_set}", ([], {"T": [], "E": []}))
            bfield = "T" if quantity == "temperature" else "E"

            tracer_kwargs = dict(
                tracer_type="NuMap",
                name=f"{map_set}_s{spin}",
                quantity=f"cmb_{quantity}",
                spin=spin,
                nu=nus,
                bandpass=passband,
                ell=ell,
                beam=beam[bfield],
            )

            s.add_tracer(**tracer_kwargs)

    for count, (spec, cross, *_) in enumerate(cov_order):

        # Define tracer names and cl type
        p1, p2 = spec
        tracer1, tracer2 = cross.split("x")
        tracer1 += "_s0" if p1 == "T" else "_s2"
        tracer2 += "_s0" if p2 == "T" else "_s2"

        map_types = {"T": "0", "E": "e", "B": "b"}
        if p2 == "T":
            data_type = "cl_" + map_types[p2] + map_types[p1]
        else:
            data_type = "cl_" + map_types[p1] + map_types[p2]

        # Add ell/cl to sacc
        Db = data_vec[count * n_bins : (count + 1) * n_bins]

        # Add Bbl
        bp_window = None
        if bbls is not None:
            if (bbl := bbls.get(cross)) is None:
                raise ValueError(f"Missing bbl for '{cross}' cross spectra!")
            ls_w = np.arange(2, bbl.shape[-1] + 2)
            bp_window = sacc.BandpowerWindow(ls_w, bbl.T)

        log.debug(f"Adding '{cross}', {spec} spectrum as {data_type} {tracer1} {tracer2}")

        kwargs = dict(
            data_type=data_type, tracer1=tracer1, tracer2=tracer2, ell=lb, x=Db, window=bp_window
        )
        s.add_ell_cl(**kwargs)

    # Add metadata
    s.metadata = deepcopy(metadata)

    # Finally add covariance
    if cov is not None:
        log.info("Adding covariance")
        s.add_covariance(cov)

    log.info(f"Writing {sacc_file_name} \n")
    s.save_fits(sacc_file_name, overwrite=True)


def extract_sacc_spectra(likelihood_name, input_file, cov_Bbl_file=None):
    """
    This function extracts spectra from a sacc file through an "mflike"-like likelihood.
    It returns spectra and covariance block as python dictionnary the same way
    the likelihood reads and parses the sacc file content.

    Parameters
    ----------
    likelihood_name: str
      the likelihood name. Must equivalent to the name set in the cobaya yaml file
      i.e. "mflike.MFLike" for default SO likelihood or "act_dr6_mflike.ACTDR6MFLike"
      for ACT DR6 likelihood
    input_file: path
      the path to the input sacc file
    cov_Bbl_file: path
      the path to the covariance file if not inside the input file (default: None).
      The dirname **must** be the same as the input file.
    """
    if cov_Bbl_file and os.path.dirname(cov_Bbl_file) != os.path.dirname(input_file):
        raise ValueError(
            "The directory path of the covariance file is different from the input file!"
        )

    likelihood_module, likelihood_class = likelihood_name.rsplit(".", 1)
    likelihood_module = importlib.import_module(likelihood_module)
    likelihood_class = getattr(likelihood_module, likelihood_class)

    # Do not check installation of likelihood
    likelihood_class.install_options = None

    my_like = likelihood_class(
        {
            "data_folder": os.path.dirname(input_file),
            "input_file": os.path.basename(input_file),
            "cov_Bbl_file": os.path.basename(cov_Bbl_file) if cov_Bbl_file else None,
        }
    )

    spectra = {}
    for data in my_like.spec_meta:
        lb, db = data.get("leff"), data.get("cl_data")
        cross = (data.get("t1"), data.get("t2"))
        mode = data.get("pol") if not data.get("hasYX_xsp") else "et"
        ids = data.get("ids")
        cov = my_like.cov[np.ix_(ids, ids)]
        spectra.setdefault((mode, *cross), []).append(dict(lb=lb, db=db, cov=cov))

    return spectra
