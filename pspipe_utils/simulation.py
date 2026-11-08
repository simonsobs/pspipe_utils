"""
Some utility functions for the generation of simulations.
"""
from pspipe_utils import dict_utils

import numpy as np
from pixell import enmap, curvedsky, lensing
from pspy import so_spectra, so_map
from mnms import noise_models as nm, utils


CMB_BASE_SEED = 0
FG_BASE_SEED = 1 
BEAM_BASE_SEED = 2
LEAKAGE_BASE_SEED = 3

def cmb_matrix_from_file(f_name, lmax, spectra, input_type="Dl"):
    """This function read the cmb power spectra from disk and return a
     [3, 3, lmax] matrix with the cmb power spectra.

    Parameters
    ----------
    f_name : string
        the file_name of the power spectra
    lmax: integer
        the maximum multipole for the cmb power spectra

    """
    l, ps_theory = so_spectra.read_ps(f_name, spectra=spectra)
    assert l[0] == 2, "the file is expected to start at l=2"

    ps_mat = np.zeros((3, 3, lmax))
    for i, f1 in enumerate("TEB"):
        for j, f2 in enumerate("TEB"):
            if input_type == "Dl":
                ps_theory[f1+f2] *= 2 * np.pi / (l * (l + 1))
            ps_mat[i, j][2:lmax] = ps_theory[f1+f2][:lmax-2]

    return ps_mat

def unlensed_cmb_and_lensing_matrix_from_file(f_name, lmax, spectra, input_type="Dl"):
    """This function read the cmb power spectra from disk and return a
     [4, 4, lmax] matrix with the unlensed cmb power spectra and lensing
     potential spectrum.

    Parameters
    ----------
    f_name : string
        the file_name of the power spectra
    lmax: integer
        the maximum multipole for the cmb power spectra

    """
    l, ps_theory = so_spectra.read_ps(f_name, spectra=spectra)
    assert l[0] == 2, "the file is expected to start at l=2"

    ps_mat = np.zeros((4, 4, lmax))
    for i, f1 in enumerate("TEBP"):
        for j, f2 in enumerate("TEBP"):
            if input_type == "Dl":
                ps_theory[f1+f2] *= 2 * np.pi / (l * (l + 1))
            ps_mat[i, j][2:lmax] = ps_theory[f1+f2][:lmax-2]

    return ps_mat

def noise_matrix_from_files(f_name_tmp, survey, arrays, lmax, nsplits, spectra, input_type="Dl"):

    """This function uses the measured noise power spectra from disk
    and generate a three dimensional array of noise power spectra [3 * n_arrays, 3 * n_arrays, lmax]
    The measured noise spectra is supposed to be the "mean noise" so to get the split noise we have to multiply by nsplits
    Note the the function return noise matrix in "Cl", so apply an extra correction if the input is "Dl"

    Parameters
    ----------
    f_name_tmp : string
        a template name of the noise power spectra
    survey : string
        the survey to consider
    arrays: list of string
        the arrays we consider
    lmax: integer
        the maximum multipole for the noise power spectra
    n_splits: integer
        the number of data splits we want to simulate
        nl_per_split= nl * n_{splits}
    input_type: str
        "Cl" or "Dl"

    """

    n_arrays = len(arrays)
    nl_array = np.zeros((3 * n_arrays, 3 * n_arrays, lmax))

    for c1, ar1 in enumerate(arrays):
        for c2, ar2 in enumerate(arrays):
            l, nl_dict = so_spectra.read_ps(f_name_tmp.format(ar1, ar2, survey), spectra=spectra)
            assert l[0] == 2, "the file is expected to start at l=2"

            for s1, field1 in enumerate("TEB"):
                for s2, field2 in enumerate("TEB"):
                    if input_type == "Dl":
                        nl_dict[field1 + field2] *=  2 * np.pi / (l * (l + 1))

                    nl_array[c1 + n_arrays * s1, c2 + n_arrays * s2, 2:lmax] = nl_dict[field1 + field2][:lmax-2] * nsplits

    return l, nl_array


def foreground_matrix_from_files(f_name_tmp, arrays_list, lmax, spectra, input_type="Dl"):

    """This function read the best fit foreground power spectra from disk
    and generate a three dimensional array of foregroung power spectra [3 * nfreqs, 3 * nfreqs, lmax].
    The file on disk are expected to start at l=2 while the matrix will start at l=0

    Parameters
    ----------
    f_name_tmp : string
      a template name of the fg power spectra
    arrays_list: 1d array of string
      the arrays we consider
    lmax: integer
      the maximum multipole for the foreground power spectra
    spectra: list of strings
      the arrangement of the spectra for example:
      ['TT','TE','TB','ET','BT','EE','EB','BE','BB']
    input_type: str
        "Cl" or "Dl"

    """

    narrays = len(arrays_list)
    fl_array = np.zeros((narrays, 3, narrays, 3, lmax))

    for c1, array1 in enumerate(arrays_list):
        for c2, array2 in enumerate(arrays_list):
            try:
                l, fl_dict = so_spectra.read_ps(f_name_tmp.format(array1, array2), spectra=spectra)
            except FileNotFoundError:
                # possible the fg is saved with the array order reversed.
                # if so, also need to reverse the pol order in the spectra
                spectra_reversed = [spec[::-1] for spec in spectra]
                l, fl_dict = so_spectra.read_ps(f_name_tmp.format(array2, array1), spectra=spectra_reversed)

            assert l[0] == 2, "the file is expected to start at l=2"

            for s1, field1 in enumerate("TEB"):
                for s2, field2 in enumerate("TEB"):
                    if input_type == "Dl":
                        fl_dict[field1 + field2] *=  2 * np.pi / (l * (l + 1))

                    fl_array[c1, s1, c2, s2, 2:lmax] = fl_dict[field1 + field2][:lmax-2]

    return l, fl_array

class DataModel:

    def __init__(self, signal_model_args, noise_model_args,
                 signal_model_kwargs=None, noise_model_kwargs=None):
        """A wrapper around a SignalModel and NoiseModel instance, allowing
        interfacing with just one object. This also makes it easy to apply a 
        mask_obs from a mnms.NoiseModel to a signal simulation (for extra
        realism of the data), without the mess of dealing with mnms inside 
        SignalModel instances too.

        Parameters
        ----------
        signal_model_args : iterable
            Positional arguments to pass to the SignalModel constructor.
        noise_model_args : iterable
            Positional arguments to pass to the NoiseModel constructor.
        signal_model_kwargs : dict, optional
            Keyword arguments to pass to the SignalModel constructor, by default
            None.
        noise_model_kwargs : dict, optional
            Keyword arguments to pass to the NoiseModel constructor, by default
            None.
        """
        if signal_model_kwargs is None:
            signal_model_kwargs = {}
        if noise_model_kwargs is None:
            noise_model_kwargs = {}

        self._signal_model = SignalModel(*signal_model_args, **signal_model_kwargs)
        self._noise_model = NoiseModel(*noise_model_args, **noise_model_kwargs)

    def get_signal_sim(self, mapname, sim_num):
        """Get the signal sim for the given map and sim realization index. If
        the currently-stored sim for the model is not this realization, a new one
        will need to be drawn. Thus, it makes sense for a loop to iterate 
        slowly over sim_num, and faster over mapname.

        Parameters
        ----------
        mapname : str
            The full survey and map information in format '{sv}_{m}'
        sim_num : int
            A sim realization index, used in setting the seed.

        Returns
        -------
        (3, ny, nx) or (3, npix) so_map.so_map
            A polarized realization for this signal map.

        Notes
        -----
        Unlike SignalModel.get_sim, this will multiply the output map by the 
        mask of observed pixels (as computed by the mnms.NoiseModel for this
        map).
        """
        out = self._signal_model.get_sim(mapname, sim_num)

        tag = self._noise_model._mapnames2modeltags[mapname]
        m_nm = self._noise_model._modeltags2modelinfos[tag]['noise_model']
        mask_obs_dg1 = m_nm.get_from_cache('mask_obs', downgrade=1)

        # NOTE: extraction only works if full res nm data and geometry are
        # compatible. this should be the case if people are careful
        shape, wcs = out.data.geometry
        mask_obs_dg1 = enmap.extract(mask_obs_dg1, shape, wcs)
        
        out.data *= mask_obs_dg1

        return out
    
    def get_noise_sim(self, mapname, split_num, sim_num):
        """Get the noise sim for the map, split, and sim realization index. If 
        the currently-stored sim for the model is not this realization, a new one
        will need to be drawn. Thus, it makes sense for a loop to iterate 
        slowly over sim_num, and faster over mapname.

        Parameters
        ----------
        mapname : str
            The full survey and map information in format '{sv}_{m}'
        split_num : int
            The split to draw a simulation for.
        sim_num : int
            A sim realization index, used in setting the seed.

        Returns
        -------
        (3, ny, nx) or (3, npix) so_map.so_map
            A polarized realization for this noise map.

        Notes
        -----
        This is just a wrapper for NoiseModel.get_sim.
        """
        return self._noise_model.get_sim(mapname, split_num, sim_num)
        
    def get_sim(self, mapname, split_num, sim_num):
        """Get the signal + noise sim for the map, split, and sim realization
        index. If the currently-stored sim for the model is not this realization,
        a new one will need to be drawn. Thus, it makes sense for a loop to
        iterate slowly over sim_num, and faster over mapname.

        Parameters
        ----------
        mapname : str
            The full survey and map information in format '{sv}_{m}'
        split_num : int
            The split to draw a simulation for.
        sim_num : int
            A sim realization index, used in setting the seed.

        Returns
        -------
        (3, ny, nx) or (3, npix) so_map.so_map
            A polarized realization for this signal + noise map.
        """
        out = self.get_signal_sim(mapname, sim_num)
        out += self.get_noise_sim(mapname, split_num, sim_num)
        return out

class SignalModel:

    def __init__(self, mapnames2minfos, lmax, cmb_mat, fg_mat, bl, cal,
                 pol_eff, bl_err=None, gl=None, gl_err=None,
                 gl_includes_poleff=False, dtype=np.float32, pixwin_apod_deg=1):
        """Simulate a set of signal maps including the lensed CMB, Gaussian
        foregrounds, and potentially beam errors, leakage, and leakage errors.
        The lensing can be via CMB power spectrum being lensed, or can be
        performed via a realization of phi. The beam is always convolved. The
        pixel windows are also always convolved. The cal and poleff is applied
        (in the opposite sense of the correction, so the sims are "like" data),
        but can not be made stochastic. 

        Parameters
        ----------
        mapnames2minfos : (nmap,) dict
            A dictionary with keys of mapnames ('{sv}_{m}') and value of a dict
            with keys:
            * 'geometry': shape, wcs for that map.
            * 'pixwin': the full 2d pixel window for that map.
            * 'pixwin_ell': optional (nl,) harmonic pixel window for that map
            (e.g. a HEALPIX map reprojected to CAR), applied to T and P like
            a beam. If None or missing, not applied.
        lmax : int
            Max multipole of signal sims.
        cmb_mat : (3, 3, nl) or (4, 4, nl) array-like
            TEB or TEBP cross-spectra. If lensing potential (P) is included,
            unlensed cmb maps are drawn from the first [3, 3] part, and a lensing
            realization from the last part, and the cmb map is lensed using
            lenspyx.
        fg_mat : (nmap, 3, nmap, 3, nl) array-like
            Polarized foreground cross-spectra.
        bl : [nmap, (2, nl)] list of array-like
            Polarized beams for each map, T and P.
        cal : (nmap,) array-like, optional
            Cal factors for each map.
        pol_eff : (nmap,) array-like, optional
            Pol_eff factors for each map.
        bl_err : [nmap, ({1,2}, nmode, nl)] list of array-like, optional
            The error modes for the beam, by default None. Assumed to be
            independent over maps. If the second axis has shape 1, assumed to be
            covariant error over all pols. If the second axis has shape 2, 
            then independent error for T and P. If None, no beam error
            realization is drawn.
        gl : [nmap, (2, nl)] list of array-like, optional
            Beam leakage, T2E and T2B, by default None. If None, no leakage is
            applied to this simulation.
        gl_err : [nmap, (2, nmode, nl)] list of array-like, optional
            The error modes for the leakage, by default None. Assumed to be 
            independent over maps. If None, no leakage error realization is 
            drawn. An error is raised if gl_err is provided but gl is None.
        gl_includes_poleff : bool, optional
            Whether the leakage products are still carring a pol_eff factor, by
            default False. If True, they are divided by the provided pol_eff. 
        dtype : np.dtype, optional
            The dtype of the simulation, by default single precision.
        pixwin_apod_deg : scalar, optional
            The signal is provided at the level of the map-on-disk, so it needs
            to be convolved with the pixel window (see mapnames2minfos). Each
            map is thus apodized by this amount on the edge before the pixwin is
            applied. Apodized pixels are then cut from the final realization.
        """
        nmap = len(mapnames2minfos)

        # are we lensing or not?
        assert cmb_mat.ndim == 3, f'{cmb_mat.ndim=} must equal 3'
        assert cmb_mat.shape[0] == cmb_mat.shape[1], \
            f'{cmb_mat.shape[0]=} must equal {cmb_mat.shape[1]=}'
        ncomp = cmb_mat.shape[0]
        assert ncomp in (3, 4), f'{ncomp=} must be 3 or 4'
        self._simulate_lens = ncomp == 4

        # which systematics are we simulating, if any?
        simulate_beam_err = bl_err is not None 
        simulate_leakage = gl is not None 
        simulate_leakage_err = gl_err is not None
        if simulate_leakage_err:
            assert simulate_leakage, \
                'only simulate leakage err if simulate leakage'

        self._simulate_beam_err = simulate_beam_err 
        self._simulate_leakage = simulate_leakage
        self._simulate_leakage_err = simulate_leakage_err

        # sanity check map_list and products. also need to compile beam products
        # into arrays with a minimal lmax
        assert fg_mat.ndim == 5, f'{fg_mat.ndim=} must equal 5'
        assert fg_mat.shape[0] == fg_mat.shape[2], \
            f'{fg_mat.shape[0]=} must equal {fg_mat.shape[2]=}'
        assert fg_mat.shape[1] == fg_mat.shape[3], \
            f'{fg_mat.shape[1]=} must equal {fg_mat.shape[3]=}'
        assert fg_mat.shape[0] == nmap, \
            f'{fg_mat.shape[0]=} must equal {nmap=}'
        assert fg_mat.shape[1] == 3, \
            f'{fg_mat.shape[0]=} must equal 3'

        def get_minimal_lmax(list_of_elly_things):
            _lmax = np.inf
            for elly_thing in list_of_elly_things: 
                _lmax = min(_lmax, elly_thing.shape[-1] - 1)
            return _lmax

        def prep_array_of_elly_things(list_of_elly_things):
            assert len(list_of_elly_things) == nmap, \
                f'{len(list_of_elly_things)=} must equal {nmap=}'
            preshape = list_of_elly_things[0].shape[:-1]
            elly_max = get_minimal_lmax(list_of_elly_things)
            
            array_of_elly_things = np.zeros((nmap, *preshape, elly_max+1), dtype=dtype)
            for i, elly_thing in enumerate(list_of_elly_things):
                _preshape = elly_thing.shape[:-1]
                assert _preshape == preshape, \
                    f'{_preshape=} does not match {preshape=}'
                array_of_elly_things[i] = elly_thing[..., :elly_max+1]
            return array_of_elly_things

        bl = prep_array_of_elly_things(bl)
        assert len(cal) == nmap, \
            f'{len(cal)=} must equal {nmap=}'
        assert len(pol_eff) == nmap, \
            f'{len(pol_eff)=} must equal {nmap=}'
        if simulate_beam_err:
            bl_err = prep_array_of_elly_things(bl_err)
        if simulate_leakage:
            gl = prep_array_of_elly_things(gl)
        if simulate_leakage_err:
            gl_err = prep_array_of_elly_things(gl_err)
        
        # sanity check lmax of all products, assume start from l=0
        elly_things = [cmb_mat, fg_mat, bl]
        if simulate_beam_err:
            elly_things.append(bl_err)
        if simulate_leakage:
            elly_things.append(gl)
        if simulate_leakage_err:
            elly_things.append(gl_err)
        elly_things += [info['pixwin_ell'] for info in mapnames2minfos.values()
                        if info.get('pixwin_ell') is not None]
        _lmax = get_minimal_lmax(elly_things)

        assert lmax <= _lmax, \
            f"{lmax=} must be less than or equal to the lowest lmax of each of " + \
            f"this model's components ({_lmax})"
        
        self._lmax = lmax
        self._dtype = dtype

        # hold metadata about surveys, maps, ps, beams, leakages etc
        self._mapnames2minfos = mapnames2minfos
        self._nmap = nmap
        self._cmb_mat = cmb_mat[..., :lmax+1].astype(dtype, copy=False)
        self._fg_mat = fg_mat[..., :lmax+1].reshape(nmap*3, nmap*3, -1).astype(dtype, copy=False)
        self._bl = np.repeat(bl[..., :lmax+1], [1, 2], axis=1).astype(dtype, copy=False) # (nmap, 2, nl) -> (nmap, 3, nl)
        self._cal = np.asarray(cal).astype(dtype, copy=False)
        self._pol_eff = np.asarray(pol_eff).astype(dtype, copy=False)
        if simulate_beam_err:
            self._bl_err = bl_err[..., :lmax+1].astype(dtype, copy=False)
        if simulate_leakage:
            if gl_includes_poleff:
                gl = gl / self._pol_eff.reshape(nmap, 1, 1) # new copy
            self._gl = gl[..., :lmax+1].astype(dtype, copy=False)
        if simulate_leakage_err:
            if gl_includes_poleff:
                gl_err = gl_err / self._pol_eff.reshape(nmap, 1, 1, 1) # new copy
            self._gl_err = gl_err[..., :lmax+1].astype(dtype, copy=False)

        # harmonic pixwins (e.g. HEALPIX maps reprojected to CAR) act like a beam
        self._pixwin_ell = None
        if any(info.get('pixwin_ell') is not None for info in mapnames2minfos.values()):
            self._pixwin_ell = np.ones((nmap, lmax+1), dtype=dtype)
            for i, info in enumerate(mapnames2minfos.values()):
                if info.get('pixwin_ell') is not None:
                    self._pixwin_ell[i] = info['pixwin_ell'][:lmax+1]

        # hold current sims so that multiple calls to get_sim can consume maps
        # from one correlated signal sim
        self._current_sims = {}
        self._current_sim_num = -1 # start with dummy

        # get the apodized mask that will be used to apply a pixwin. this is
        # only relevant for CAR maps for which the pixwin is being deconvolved
        for info in mapnames2minfos.values():
            if info['pixwin'] is not None:
                shape, wcs = info['geometry']

                # construct the mask for the convolution. the top and bottom pixels
                # are always discontinuous even on the fullsky, but left and right
                # are continuous on fullsky
                mask = enmap.ones(shape[-2:], wcs, dtype)
                mask[..., 0, :] = 0
                mask[..., -1, :] = 0
                if not np.round(360/abs(wcs.wcs.cdelt[0])).astype(int) == shape[-1]:
                    mask[..., :, 0] = 0
                    mask[..., :, -1] = 0
                mask = enmap.apod_mask(mask, width=np.deg2rad(pixwin_apod_deg), edge=False)
                mask = mask.astype(dtype, copy=False)

                info['pixwin_mask'] = mask
                info['mask_obs'] = mask == 1

    def _get_all_sim(self, sim_num):
        # TODO: use parallelized mnms functions
        cdtype = np.result_type(1j, self._dtype)

        # first get cmb sim, possibly lensed
        alms = utils.rand_alm(self._cmb_mat, lmax=self._lmax,
                              seed=(CMB_BASE_SEED, sim_num), dtype=cdtype,
                              nchunks=None)
        if self._simulate_lens:
            unlensed_cmb_alms = alms[:3]
            phi_alm = alms[3]

            # lensing needs to project onto map first, but then we want to 
            # convolve with beam, so we want to use the lowest-res intermediate
            # map possible before going back to alm. for exact alms, we want
            # to use the 2d method, which requires lmax+1 rings
            ishape, iwcs = enmap.fullsky_geometry(shape=(self._lmax+1, 2*(self._lmax+1)),
                                                  variant='fejer1')
            
            # FIXME: both pixell and lenspyx needs double-prec for some reason
            phi_alm = phi_alm.astype(np.complex128, copy=False)
            unlensed_cmb_alms = unlensed_cmb_alms.astype(np.complex128, copy=False)
            lensed_cmb = lensing.lens_map_curved((3, *ishape), iwcs, phi_alm,
                                                  unlensed_cmb_alms,
                                                  dtype=np.float64, spin=[0, 2],
                                                  output='l', method='lenspyx')[0]
            lensed_cmb = lensed_cmb.astype(self._dtype, copy=False)

            alms = curvedsky.map2alm(lensed_cmb, lmax=self._lmax, method='2d')
            unlensed_cmb_alms = None
            phi_alm = None
            lensed_cmb = None

        # next add les fgs
        fg_alms = utils.rand_alm(self._fg_mat, lmax=self._lmax, 
                                 seed=(FG_BASE_SEED, sim_num), dtype=cdtype,
                                 nchunks=100)
        fg_alms = fg_alms.reshape(self._nmap, 3, -1)
        alms = alms + fg_alms # (3, nlm) + (nmap, 3, nlm)
        fg_alms = None

        # next apply beam (TODO: beam only commutes with leakage if bT = bP. If
        # not, better to construct the full 3x3 operator). beam is diagonal
        # and preshape broadcasts exactly with alms (nmap, 3, nl). if necessary,
        # get beam realization
        bl = self._bl
        if self._simulate_beam_err:
            npol = self._bl_err.shape[1] # 1 or 2
            bl_delta = np.zeros((self._nmap, npol, self._lmax+1), dtype=self._dtype)
            for p in range(npol): # T or T,P
                bl_err = self._bl_err[:, p] # (nmap, p, nmode, nl)
                rng = np.random.default_rng(seed=(BEAM_BASE_SEED, p, sim_num))
                err_modes = rng.standard_normal(size=(self._nmap, bl_err.shape[-2]),
                                                dtype=self._dtype) # (nmap, nmode)
                bl_delta[:, p] = np.einsum('nml,nm->nl', bl_err, err_modes) # (nmap, nl)
            if npol == 2:
                bl_delta = np.repeat(bl_delta, [1, 2], axis=1) # (nmap, 3, nl)
            bl = bl + bl_delta # copy = (nmap, 3, nl) + (nmap, {1,3}, nl)

        # harmonic pixwin is the same for T and P, so it commutes with leakage
        if self._pixwin_ell is not None:
            bl = bl * self._pixwin_ell[:, None] # copy = (nmap, 3, nl) * (nmap, 1, nl)
        curvedsky.almxfl(alms, bl, out=alms)

        # next apply leakage. if necessary, get leakage realization
        if self._simulate_leakage:
            gl = self._gl
            if self._simulate_leakage_err:
                gl_delta = np.zeros((self._nmap, 2, self._lmax+1), dtype=self._dtype)
                for p in range(2): # T2E, T2B independently
                    gl_err = self._gl_err[:, p] # (nmap, p, nmode, nl)
                    rng = np.random.default_rng(seed=(LEAKAGE_BASE_SEED, p, sim_num))
                    err_modes = rng.standard_normal(size=(self._nmap, gl_err.shape[-2]),
                                                    dtype=self._dtype) # (nmap, nmode)
                    gl_delta[:, p] = np.einsum('nml,nm->nl', gl_err, err_modes) # (nmap, nl)
                gl = gl + gl_delta # copy = (nmap, 2, nl) + (nmap, 2, nl)

            # (nmap, E, nlm) += (nmap, T, nlm) x (nmap, T2E, nl)
            # (nmap, B, nlm) += (nmap, T, nlm) x (nmap, T2B, nl)
            alms[:, 1] += curvedsky.almxfl(alms[:, 0], gl[:, 0])
            alms[:, 2] += curvedsky.almxfl(alms[:, 0], gl[:, 1])

        # finally, apply cal and poleff (*poleff/cal, opposite of "correction")
        alms[:, 1:] *= self._pol_eff.reshape(self._nmap, 1, 1)
        alms /= self._cal.reshape(self._nmap, 1, 1)

        # place alms in current_sim
        for i, mapname in enumerate(self._mapnames2minfos):
            self._current_sims[mapname] = alms[i]
        self._current_sim_num = sim_num

        alms = None

    def get_sim(self, mapname, sim_num):
        """Get the sim for the given map and sim realization index. If the 
        currently-stored sim for the model is not this realization, a new one
        will need to be drawn. Thus, it makes sense for a loop to iterate 
        slowly over sim_num, and faster over mapname.

        Parameters
        ----------
        mapname : str
            The full survey and map information in format '{sv}_{m}'
        sim_num : int
            A sim realization index, used in setting the seed.

        Returns
        -------
        (3, ny, nx) or (3, npix) so_map.so_map
            A polarized realization for this signal map.
        """
        assert mapname in self._mapnames2minfos, \
            f"{mapname=} not in this model's set of maps " + \
            f"({list(self._mapnames2minfos.keys())})"

        if self._current_sim_num != sim_num:
            self._get_all_sim(sim_num)

        # finalize sim. project to patch and apply pixwin
        shape, wcs = self._mapnames2minfos[mapname]['geometry']
        pixwin = self._mapnames2minfos[mapname]['pixwin']
        
        omap = enmap.zeros((3, *shape), wcs=wcs, dtype=self._dtype)
        curvedsky.alm2map(self._current_sims[mapname], omap)

        if pixwin is not None:
            pixwin_mask = self._mapnames2minfos[mapname]['pixwin_mask']
            kmap = utils.rfft(omap * pixwin_mask, normalize='forward')
            # get rfft cutout of pixwin
            utils.irfft(kmap * pixwin[..., :kmap.shape[-1]], omap=omap,
                        normalize='forward', n=shape[-1])
            mask_obs = self._mapnames2minfos[mapname]['mask_obs']
            omap *= mask_obs
        
        return so_map.from_enmap(omap)
    
class NoiseModel:

    def __init__(self, mapnames2minfos, modeltags2modelinfos,
                 keep_model=True, dtype=np.float32):
        """Simulate a set of noise maps from mnms. This class handles the 
        mapping between mapnames and mnms noise models. The final noise 
        sims are in the geometry passed into this object, not necessarily the
        geometry of the mnms noise model. It is only required that the dg1 
        geometry in the mnms noise model is compatible with the passed 
        geometry (this should be true if the noise model is built for the maps
        that define the passed geometry). White noise can optionally be added
        above the mnms noise model's lmax from the ivar maps.

        Parameters
        ----------
        mapnames2minfos : (nmap,) dict
            A dictionary with keys of mapnames ('{sv}_{m}') and value of a dict
            with keys:
            * 'geometry': shape, wcs for that map.
            * 'noise_info': A dictionary with keys:
                * 'qid': The qid that corresponds to this mapname 
                * 'subproduct_kwargs': A dictionary with keys for possible 
                subproduct_kwargs and their values, that further specifies this
                mapname. Can be omitted if there are not any.
        modeltags2modelinfos : (nmodel,) dict
            A dictionary with keys of 'noise_model_tag's and values:
            * 'config_name': argument of mnms.noise_models.BaseNoiseModel.from_config
            * 'noise_model_name': argument of mnms.noise_models.BaseNoiseModel.from_config
            * 'qids': argument of mnms.noise_models.BaseNoiseModel.from_config
            * 'subproduct_kwargs': argument of mnms.noise_models.BaseNoiseModel.from_config
            * 'lmax': The lmax that will be used for this noise model
            * 'add_white_noise_above_lmax': argument of mnms.noise_models.BaseNoiseModel.get_sim
            * 'white_noise_ell_taper_width': optional argument of mnms.noise_models.BaseNoiseModel.get_sim
        keep_model : bool, optional
            Keep the noise model files in memory after loading the first time, 
            by default True. This spends memory to save execution time. 
        dtype : np.dtype, optional
            The dtype of the simulation, by default single precision.
        """
                
        self._mapnames2minfos = mapnames2minfos
        self._modeltags2modelinfos = modeltags2modelinfos
        self._mapnames2modeltags = dict_utils.get_mapnames_to_noise_model_tags(
             mapnames2minfos=mapnames2minfos, modeltags2modelinfos=modeltags2modelinfos
        )
        self._keep_model = keep_model
        self._dtype = dtype

        # hold current sims so that multiple calls to get_sim can consume maps
        # from one correlated noise sim. since _get_all_sim calls by tag, split,
        # self._current_sim_num also needs to hold for separate tag, splits (but
        # in practice it is unlikely the separate splits would become out of
        # sync)
        self._current_sims = {}
        self._current_sim_num = {}

        # instantiate all the models
        for tag, info in self._modeltags2modelinfos.items():
            m_nm = self._get_noise_model_from_tag(tag)

            # want to hold full-res mask_obs for projection and extraction
            mask_obs_dg1 = m_nm.get_mask_obs(downgrade=1)
            m_nm.cache_data('mask_obs', mask_obs_dg1, downgrade=1)

            # set the current sim to a dummy value for each split
            for s in range(m_nm._num_splits):
                self._current_sim_num[tag, s] = -1 # start with dummy

            info['noise_model'] = m_nm

    def _get_super_qid_from_mapname(self, mapname, key=False):
        noise_info = self._mapnames2minfos[mapname]['noise_info']
        qid = noise_info['qid']
        subproduct_kwargs = noise_info.get('subproduct_kwargs', {}) # might be empty
        if key:
            return (frozenset(subproduct_kwargs.items()), qid)
        else:
            return (subproduct_kwargs, qid)

    def _get_noise_model_from_tag(self, noise_model_tag):
        noise_model_info = self._modeltags2modelinfos[noise_model_tag]
        config_name = noise_model_info['config_name']
        noise_model_name = noise_model_info['noise_model_name']
        qids = noise_model_info['qids']
        subproduct_kwargs = noise_model_info.get('subproduct_kwargs', {})
        return nm.BaseNoiseModel.from_config(config_name, noise_model_name, *qids, **subproduct_kwargs)
    
    def _get_all_sim(self, tag, split_num, sim_num):
        m_nm = self._modeltags2modelinfos[tag]['noise_model']
        lmax = self._modeltags2modelinfos[tag]['lmax']
        add_white_noise_above_lmax = self._modeltags2modelinfos[tag]['add_white_noise_above_lmax']

        # optional, otherwise use the mnms default
        get_sim_kwargs = {}
        if 'white_noise_ell_taper_width' in self._modeltags2modelinfos[tag]:
            get_sim_kwargs['white_noise_ell_taper_width'] = self._modeltags2modelinfos[tag]['white_noise_ell_taper_width']

        # get the sim in map space
        sim = m_nm.get_sim(split_num, sim_num, lmax, alm=False,
                           check_disk_for_map_and_alm=True,
                           calibrate=False,
                           add_white_noise_above_lmax=add_white_noise_above_lmax,
                           keep_model=self._keep_model, **get_sim_kwargs)
        sim = sim.astype(self._dtype, copy=False)
        
        # for each super_qid in the sim, store it.
        for i, (subproduct_kwargs, qid) in enumerate(m_nm._super_qids):
            key = (frozenset(subproduct_kwargs.items()), qid)
            self._current_sims[key, split_num] = sim[i].squeeze()

        self._current_sim_num[tag, split_num] = sim_num

    def get_sim(self, mapname, split_num, sim_num):
        """Get the sim for the map, split, and sim realization index. If the 
        currently-stored sim for the model is not this realization, a new one
        will need to be drawn. Thus, it makes sense for a loop to iterate 
        slowly over sim_num, and faster over mapname.

        Parameters
        ----------
        mapname : str
            The full survey and map information in format '{sv}_{m}'
        split_num : int
            The split to draw a simulation for.
        sim_num : int
            A sim realization index, used in setting the seed.

        Returns
        -------
        (3, ny, nx) or (3, npix) so_map.so_map
            A polarized realization for this noise map.
        """
        assert mapname in self._mapnames2minfos, \
            f"{mapname=} not in this model's set of maps " + \
            f"({list(self._mapnames2minfos.keys())})"

        tag = self._mapnames2modeltags[mapname]
        key = self._get_super_qid_from_mapname(mapname, key=True)

        # sims are popped once returned to save memory (each map, split is
        # normally requested once per sim_num), so redraw if already popped
        if self._current_sim_num[tag, split_num] != sim_num or \
            (key, split_num) not in self._current_sims:
            self._get_all_sim(tag, split_num, sim_num)

        # finalize sim: project to mask_obs_dg1, extract to patch
        omap = self._current_sims.pop((key, split_num))

        m_nm = self._modeltags2modelinfos[tag]['noise_model']
        mask_obs_dg1 = m_nm.get_from_cache('mask_obs', downgrade=1)

        # sims with white noise are already full-res
        if omap.shape[-2:] != mask_obs_dg1.shape:
            omap = utils.fourier_resample(omap, shape=mask_obs_dg1.shape,
                                          wcs=mask_obs_dg1.wcs)

        utils.parallel_inplace_op(np.multiply, omap, mask_obs_dg1)

        # NOTE: extraction only works if full res nm data and geometry are
        # compatible. this should be the case if people are careful
        shape, wcs = self._mapnames2minfos[mapname]['geometry']
        omap = enmap.extract(omap, shape, wcs)

        return so_map.from_enmap(omap)

def generate_fg_alms(fg_mat, arrays_list, lmax, dtype=np.complex64):
    """
    This function generate the alms corresponding to a fg matrix
    the alms are returned in the form of a dict with key "freq"
    the alms are in the format [T,E,B]

    Parameters
    ----------
    fg_mat : 2d array
      the fg matrix of size [3 * nfreqs, 3 * nfreqs, lmax]
    arrays_list: 1d array of string
      the arrays we consider
    lmax: integer
      the maximum multipole for the noise power spectra
    dtype: str
      the datatype of the alms (e.g complex64)
    """

    narrays = len(arrays_list)

    # fg_mat is like (narray, 3, narray, 3, nl), but we want to make it square
    fg_mat = fg_mat.reshape(narrays*3, narrays*3, -1)

    fglms_all = curvedsky.rand_alm(fg_mat, lmax=lmax, dtype=dtype)

    # fglms_all is now (narray*3, ...), but we want (narray, 3, ...) to make it 
    # easy
    fglms_all = fglms_all.reshape(narrays, 3, -1)

    fglm_dict = {}
    for i, array in enumerate(arrays_list):
        fglm_dict[array] = fglms_all[i]

    return fglm_dict

def generate_noise_alms(noise_mat, array_list, lmax, dtype=np.complex64):
    """
    This function generate the alms corresponding to a noise matrix
    the alms are returned in the form of a dict with key "array"
    the alms are in the format [T,E,B]

    Parameters
    ----------
    noise_mat : 2d array
        the noise matrix of size [3 * narrays, 3 * narrays, lmax]
    array_list: 1d array of string
        the frequencies we consider
    lmax: integer
        the maximum multipole for the noise power spectra
    dtype: str
        the datatype of the alms (e.g complex64)
    """

    narrays = len(array_list)
    nlms_all = curvedsky.rand_alm(noise_mat, lmax=lmax, dtype=dtype)
    nlm_dict = {}
    for i, array in enumerate(array_list):
        nlm_dict[array] = [nlms_all[i + k * narrays] for k in range(3)]

    return nlm_dict

def _fix_lenspyx_result(lenspyx_result, lenspyx_geom_info, shape, wcs):
	"""Lenspyx delivers results for rectangular geometries in not quite the
	format that we'd like to use. First, it delivers a tuple of 1d arrays,
	rather than a 3d array of shape (-1, ny, nx). Then, the bottom-left pixel
	corresponds to the ring with the smallest colatitude and minimum phi value,
	with phi increasing to the right. The result is always full-sky. This
	function rearranges the result into the shape we expect and copies its data
	into the right order, such that it matches the geometry obtained by 
	enmap.fullsky_geometry.

	Parameters
	----------
	lenspyx_result : 1d np.ndarray or tuple of 1d np.ndarray
		The results of a call to lenspyx.lensing functions.
	lenspyx_geom_info : (name, info_dict)
		The argument that would need to be supplied to lenspyx.get_geom:
		* name gives the rectangular projection ('cc' or 'f1')
		* info_dict gives the map shape as {'ntheta': ny, 'nphi': nx}
		Note this is a full-sky geom_info, not restricted.
	shape : (ny, nx) tuple
		Shape of output map (not necessarily full-sky).
	wcs : astropy.wcs.WCS
		WCS of output map (not necessarily full-sky).

	Returns
	-------
	(-1, ny, nx) enmap.ndmap
		The lenspyx_result projected onto the shape, wcs geometry.

	Raises
	------
	AssertionError
		If the lenspyx geometry is not shifted by an integer number of pixels
		from the corresponding pixell geometry.
	"""
	import lenspyx

	# do some sanity checks on the full-sky geom_info
	geom = lenspyx.get_geom(lenspyx_geom_info)
	phi0, nph = geom.phi0, geom.nph
	assert np.all(phi0 == phi0[0]), 'all phi0 must be the same'
	assert np.all(nph == nph[0]), 'all nph must be the same'

	# get the pixell full-sky geometry we will paste the results into
	variant = dict(cc='cc', f1='fejer1')[lenspyx_geom_info[0]]
	fs_shape = (lenspyx_geom_info[1]['ntheta'], lenspyx_geom_info[1]['nphi'])
	fs_shape, fs_wcs = enmap.fullsky_geometry(shape=fs_shape, variant=variant)

	# lenspyx delivers tuples of arrays, not arrays
	fs_inp = np.asarray(lenspyx_result).reshape((-1, *fs_shape))
	fs_out = np.zeros_like(fs_inp)

	# now cut and paste data! :(

	# first, handle the x coordinates. first find the location of phi0
	# in the full-sky geometry
	_, _phi0_ind = enmap.sky2pix(fs_shape, fs_wcs, [0, phi0[0]])
	phi0_ind = np.round(_phi0_ind).astype(int)
	assert np.allclose(phi0_ind, _phi0_ind, rtol=0, atol=1e-5), \
		('we cannot handle the case of a non-integer pixel with cut and paste '
		'but could roll the whole array by a fractional pixel using ffts, this '
		'needs to be implemented')

	# then do copy paste, handling whether phi increases
	if fs_wcs.wcs.cdelt[0] > 0: # wcs is in x,y ordering
		fs_out[..., phi0_ind:] = fs_inp[..., :shape[-1] - phi0_ind]
		fs_out[..., :phi0_ind] = fs_inp[..., shape[-1] - phi0_ind:]
	else:
		fs_out[..., phi0_ind::-1] = fs_inp[..., :phi0_ind+1]
		fs_out[..., :phi0_ind:-1] = fs_inp[..., phi0_ind+1:]

	# next, we know lenspyx delivers colatitudes, but fullsky_geometry
	# is likely the opposite of that
	if fs_wcs.wcs.cdelt[1] > 0: # wcs is in x,y ordering
		fs_out = fs_out[..., ::-1, :] # flip the y coordinates if theta increases

	# finally, extract the cutout we want
	fs_out = enmap.ndmap(fs_out, fs_wcs)
	return enmap.extract(fs_out, shape, wcs)