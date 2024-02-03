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
