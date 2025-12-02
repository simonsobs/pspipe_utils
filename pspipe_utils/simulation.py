"""
Some utility functions for the generation of simulations.
"""
import numpy as np
from pixell import enmap, curvedsky, lensing
from pspy import so_spectra, so_map
from mnms import noise_models as nm, utils


CMB_BASE_SEED = 0
FG_BASE_SEED = 1 
BEAM_BASE_SEED = 2
LEAKAGE_BASE_SEED = 3
WHITE_NOISE_BASE_SEED = 4

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

        noise_info = self._noise_model._mapnames2minfos[mapname]['noise_info']
        tag = noise_info['noise_model_tag']
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
                gl = gl / pol_eff.reshape(nmap, 1, 1) # new copy
            self._gl = gl[..., :lmax+1].astype(dtype, copy=False)
        if simulate_leakage_err:
            if gl_includes_poleff:
                gl_err = gl_err / pol_eff.reshape(nmap, 1, 1, 1) # new copy
            self._gl_err = gl_err[..., :lmax+1].astype(dtype, copy=False)

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
            lensed_cmb = _lens_map_curved_lenspyx((3, *ishape), iwcs, phi_alm,
                                                  unlensed_cmb_alms,
                                                  dtype=np.float64, spin=[0, 2],
                                                  output='l')[0]
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
                 add_white_noise_above_lmax=False,
                 white_noise_ell_taper_width=500,
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
                * 'noise_model_tag': The noise model this mapname will be drawn
                from (a key in modeltags2modelinfos).
                * 'qid': The qid that corresponds to this mapname 
                * 'subproduct_kwargs': A dictionary with keys for possible 
                subproduct_kwargs and their values, that further specifies this
                mapname. Can be omitted if there are not any.
        modeltags2modelinfos : (nmodel,) dict
            A dictionary with keys of 'noise_model_tag's (see above) and values:
            * 'config_name': argument of mnms.noise_models.BaseNoiseModel.from_config
            * 'noise_model_name': argument of mnms.noise_models.BaseNoiseModel.from_config
            * 'qids': argument of mnms.noise_models.BaseNoiseModel.from_config
            * 'subproduct_kwargs': argument of mnms.noise_models.BaseNoiseModel.from_config
            * 'lmax': The lmax that will be used for this noise model
        add_white_noise_above_lmax : bool, optional
            Add a white noise realization from ivar above the lmax of the noise
            sims, by default False. Ivars are rescaled to match onto the level
            of the noise model cov_ell at high ell.
        white_noise_ell_taper_width : int, optional
            Width in ell below the lmax of each noise model that will be used
            to smoothly blend in the high-ell white noise realization (using
            a cosine profile)
        keep_model : bool, optional
            Keep the noise model files in memory after loading the first time, 
            by default True. This spends memory to save execution time. 
        dtype : np.dtype, optional
            The dtype of the simulation, by default single precision.
        """
                
        self._mapnames2minfos = mapnames2minfos
        self._modeltags2modelinfos = modeltags2modelinfos
        self._add_white_noise_above_lmax = add_white_noise_above_lmax
        self._white_noise_ell_taper_width = white_noise_ell_taper_width
        self._keep_model = keep_model
        self._dtype = dtype

        # hold current sims so that multiple calls to get_sim can consume maps
        # from one correlated noise sim. since _get_all_sim calls by tag, split,
        # self._current_sim_num also needs to hold for separate tag, splits (but
        # in practice it is unlikely the separate splits would become out of
        # sync)
        self._current_sims = {}
        self._current_sim_num = {}

        self._prep_noise_models()

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
    
    def _prep_noise_models(self):
        # instantiate all the models
        if self._add_white_noise_above_lmax:
            # what to multiply white noise realizations by for each map, in T
            # and pol, to match onto the high-ell noise ps
            self._sigma_facs = {} 
            
        for tag, info in self._modeltags2modelinfos.items():
            m_nm = self._get_noise_model_from_tag(tag)

            # want to hold full-res mask_obs for projection and extraction
            mask_obs_dg1 = m_nm.get_mask_obs(downgrade=1)
            m_nm.cache_data('mask_obs', mask_obs_dg1, downgrade=1)

            # may want to hold full-res sqrt_ivars for white noise (each
            # split of each superqid in this noise model). also get the factors
            # for T and P to match onto the sqrt_cov_ell (this depends on the
            # mask used which should be close to mask_obs usually). also see the
            # comments inside get_sim for more assumptions. also set the current
            # sim to a dummy value for each split
            for s in range(m_nm._num_splits):
                if self._add_white_noise_above_lmax:
                    sqrt_ivar = m_nm._empty(*mask_obs_dg1.geometry, ivar=True, num_splits=1)
                    sqrt_ivar = sqrt_ivar.astype(self._dtype, copy=False)

                    for i, (subproduct_kwargs, qid) in enumerate(m_nm._super_qids):
                        ivar = utils.read_map(
                            m_nm._data_model, qid, split_num=s, ivar=True,
                            maps_subproduct=m_nm._maps_subproduct, srcfree=m_nm._srcfree,
                            **subproduct_kwargs
                            )
                        ivar = enmap.extract(ivar, *mask_obs_dg1.geometry)
                        sqrt_ivar[i, 0] = np.sqrt(ivar).astype(self._dtype, copy=False)
                    
                    m_nm.cache_data('sqrt_ivar', sqrt_ivar, split_num=s, downgrade=1)
                
                    # get cov_ell for model facs 
                    lmax = info['lmax']
                    model_dict = m_nm.get_model(s, lmax, generate=False,
                                                keep_model=self._keep_model)
                    sqrt_cov_ell = model_dict['sqrt_cov_ell']
                    model_dict = None # save space

                    nmap = len(m_nm._super_qids)
                    sqrt_cov_ell = sqrt_cov_ell.reshape(nmap*3, nmap*3, -1)
                    cov_ell = utils.eigpow(sqrt_cov_ell, 2, axes=[0, 1])
                    sqrt_cov_ell = None # save space
                    cov_ell = cov_ell.reshape(nmap, 3, nmap, 3, -1)

                    # also need postfiltreldg to adjust sqrt_cov_ell back to
                    # dg1, with lmax. for postfiltreldg, the cov_ell will be
                    # (dg/postfiltreldg)**2 bigger than the dg1 ivars
                    dg = utils.downgrade_from_lmaxs(m_nm._full_lmax, lmax)
                    pfrd = m_nm._filter_kwargs['post_filt_rel_downgrade']
                    cov_ell /= (dg/pfrd)**2

                    # finally, compare to expected std normal white noise in the
                    # mask_obs patch for T and pol
                    nlw = np.sum(mask_obs_dg1**2 * mask_obs_dg1.pixsizemap()**2)/(4*np.pi) # std normal white noise pseudospectrum in patch
                    nlw /= np.sum(mask_obs_dg1**2 * mask_obs_dg1.pixsizemap())/(4*np.pi) # w2 of patch

                    for i, (subproduct_kwargs, qid) in enumerate(m_nm._super_qids):
                        key = (frozenset(subproduct_kwargs.items()), qid)

                        sigma_fac_T = (cov_ell[i, 0, i, 0, -100:].mean() / nlw)**0.5 # TT
                        sigma_fac_pol = ((cov_ell[i, 1, i, 1, -100:].mean() + cov_ell[i, 2, i, 2, -100:].mean())/2 / nlw)**0.5 ## (EE + BB)/2

                        self._sigma_facs[key, s] = np.array([sigma_fac_T,
                                                             sigma_fac_pol,
                                                             sigma_fac_pol],
                                                             dtype=self._dtype)

                self._current_sim_num[tag, s] = -1 # start with dummy

            info['noise_model'] = m_nm

    def _get_all_sim(self, tag, split_num, sim_num):
        m_nm = self._modeltags2modelinfos[tag]['noise_model']
        lmax = self._modeltags2modelinfos[tag]['lmax']

        # get the sim in map space
        sim = m_nm.get_sim(split_num, sim_num, lmax, alm=False,
                           calibrate=False, keep_model=self._keep_model)
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

        noise_info = self._mapnames2minfos[mapname]['noise_info']
        tag = noise_info['noise_model_tag']

        if self._current_sim_num[tag, split_num] != sim_num:
            self._get_all_sim(tag, split_num, sim_num)

        # finalize sim: project to mask_obs_dg1, extract to patch
        key = self._get_super_qid_from_mapname(mapname, key=True)
        omap = self._current_sims[key, split_num]

        m_nm = self._modeltags2modelinfos[tag]['noise_model']
        mask_obs_dg1 = m_nm.get_from_cache('mask_obs', downgrade=1)
        omap = utils.fourier_resample(omap, shape=mask_obs_dg1.shape,
                                      wcs=mask_obs_dg1.wcs)

        # white noise? NOTE: matching assumes the tiled part of the 
        # model returns noise with a power spectrum of ~1 (ie, ps~sqrt_cov_ell
        # after filtering), which may not be true but should be close at high l
        if self._add_white_noise_above_lmax:
            sqrt_ivar = m_nm.get_from_cache('sqrt_ivar', split_num=split_num,
                                            downgrade=1)
            super_qid = self._get_super_qid_from_mapname(mapname, key=False)
            i = m_nm._super_qids.index(super_qid)
            
            super_qid_str = m_nm._super_qid_strs[i]
            sqrt_ivar = sqrt_ivar[i].squeeze()

            # get the ivar realization (indep in pol)
            sigma_facs = self._sigma_facs[key, split_num][:, None, None]
            seed = utils.get_seed(split_num, sim_num, m_nm._data_model_name,
                                  m_nm._maps_product, m_nm._maps_subproduct,
                                  m_nm.noise_model_name, *m_nm._super_qid_strs,
                                  super_qid_str, str(WHITE_NOISE_BASE_SEED))
            eta = utils.concurrent_normal(size=(3, *mask_obs_dg1.shape),
                                          dtype=self._dtype, seed=seed)
            eta *= sigma_facs
            eta = enmap.samewcs(eta, sqrt_ivar)
            np.divide(eta, sqrt_ivar, where=sqrt_ivar!=0, out=eta) * (sqrt_ivar!=0)

            # blend the realizations
            lmax = self._modeltags2modelinfos[tag]['lmax']
            filt_low, filt_high = utils.get_ell_trans_profiles([lmax - self._white_noise_ell_taper_width],
                                                               [lmax], lmax=lmax, exp=0.5)
            utils.ell_filter(omap, filt_low, omap=omap, lmax=lmax)
            eta -= utils.ell_filter(eta, 1 - filt_high, lmax=lmax)

            omap += eta

        omap *= mask_obs_dg1

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

def _lens_map_curved_lenspyx(shape, wcs, phi_alm, cmb_alm, phi_ainfo=None, 
								dtype=np.float64, spin=[0, 2], output="l",
								epsilon=1e-7, nthreads=0, verbose=False):
	"""Lenses a CMB map given the lensing potential harmonic transform and the
	unlensed CMB harmonic transform.  By default, T, E, B spherical harmonic
	coefficients are accepted and the returned maps are T, Q, U. Unlike 
	lens_map_curved, this implements lensing using lenspyx, which is a more
	optimized/specialized/stable lensing library. This function formats lenspyx
	outputs to be drop-in replacements for lens_map_curved.

	Parameters
	----------
	shape : tuple
		Shape of the output map. Only the first pre-dimension (-3), if passed,
		is kept. 
	wcs : WCS object
		World Coordinate System object describing the map projection.
	phi_alm : array-like
		Spherical harmonic coefficients of the lensing potential.
	cmb_alm : array-like
		Spherical harmonic coefficients of the CMB. If (3, nelem) shaped, the
		coeffients are assumed to be in the form of [T, E, B] in that order,
		unless spin is 0.
	phi_ainfo : alm_info, optional
		alm_info object containing information about the alm layout. Default:
		standard triangular layout.
	dtype : data-type, optional
		Data type of the output maps. Default is np.float64.
	spin : list, optional
		List of spins. These describe how to handle the [ncomp] axis in cmb_alm.
	 	0: scalar transform. Consumes one element in the component axis
	 	not 0: spin transform. Consumes two elements from the component axis.
	 	For example, if you have a TEB alm [3,nelem] and want to transform it
	 	to a TQU map [3,ny,nx], you would use spin=[0,2] to perform a scalar
	 	transform for the T component and a spin-2 transform for the Q,U
	 	components. Another example. If you had an alm [5,nelem] and map
	 	[5,ny,nx] and the first element was scalar, the next pair spin-1
	 	and the next pair spin-2, you woudl use spin=[0,1,2]. default:[0,2]
	output : str, optional
		String which specifies which maps to output, e.g. "lu". Default is "l".
		"l" - lensed CMB map
		"u" - unlensed CMB map
		"p" - lensing potential map
		"k" - convergence map
		"a" - deflection angle maps
	epsilon : float, optional
		Target result accuracy, by default 1e-7. See lenspyx.
	nthreads : int, optional
		number of threads to use, by default 0 (os.cpu_count()). See lenspyx.
	verbose : bool, optional
		If True, prints progress information. Default is False.

	Returns
	-------
	tuple
		A tuple containing the requested output maps in the order specified by
		the `output` parameter.

	Notes
	-----
	This function assumes the cmb_alm is in a triangular layout.

	This function has a restrictive interpretation of spin. If the default
	[0, 2] is passed, the cmb_alm and output shape must have an axis size of
	2 or 3 in the (-3) axis, in which case the inputs are interpreted as T, E
	or T, E, B, respectively (see lenspyx). Otherwise, a fully spin-0 transform
	must be passed. The default should cover the vast majority of use-cases.
	"""
	import lenspyx

	# restrict to target number of components
	oshape  = shape[-3:]
	if len(oshape) == 2:
		oshape = (1, *shape)

	assert np.asarray(phi_alm).ndim == 1, \
		'Can only do 1-dimensional phi_alm, set up a loop if you have many'
	
	cmb_alm = np.atleast_2d(cmb_alm)
	assert cmb_alm.ndim <= 2, \
		'Can only do <=2-dimensional cmb_alm, set up a loop if you have many'

	# map from spin to pol 
	pol = False
	pre_shape = oshape[0]
	pre_cmb = cmb_alm.shape[0]
	if spin == [0, 2]:
		assert pre_cmb in (2, 3), \
			f'{spin=} indicates TEB but number of components in alm {pre_cmb=}' + \
			' not 2 or 3'
		assert pre_shape == 3, \
			f'{spin=} indicates TEB but number of components in map {pre_shape=}' + \
			' not 3'
		pol = True
	else:
		assert np.all(spin) == 0, \
			f'expect spin-0 transform for all {pre_cmb} components'
		assert pre_cmb == pre_shape, \
			f'expect {pre_cmb=} to be the same as {pre_cmb=}'
	
	# we need to get the "lenspyx geometry" from the "pixell geometry".
	# we know pixell will have a ducc-compatible rectangular geometry, so get
	# its parameters: number of pixels in x and y, and the name of the
	# rectangular geometry. pixell capitalizes the names, but lenspyx wants them 
	# lowercase. finally, handle cut-sky
	ducc_geo = curvedsky.analyse_geometry(shape, wcs).ducc_geo
	ny = ducc_geo.ny
	nx = ducc_geo.nx
	name = ducc_geo.name.lower()
	geom_info = (name, dict(ntheta=ny, nphi=nx)) # this is a full-sky geom_info

	# after getting the result from lenspyx, it needs to be fixed to respect
	# pixell conventions
	if 'l' in output:
		phi_lmax = curvedsky.nalm2lmax(phi_alm.shape[-1])
		d_alm = np.empty_like(phi_alm)
		lfilter = np.sqrt(np.arange(phi_lmax + 1) * np.arange(1, phi_lmax + 2))
		curvedsky.almxfl(phi_alm, lfilter, out=d_alm)
		cmb_obs = lenspyx.alm2lenmap(cmb_alm, d_alm, geom_info,
										epsilon=epsilon, verbose=verbose, 
									 	nthreads=nthreads, pol=pol)
		cmb_obs = _fix_lenspyx_result(cmb_obs, geom_info, oshape, wcs)
		cmb_obs = cmb_obs.astype(dtype=dtype, copy=False)

	# possibly get extra outputs
	if 'u' in output:
		cmb_raw = enmap.empty(shape, wcs, dtype=dtype)
		if verbose:
			print("Computing unlensed map")
		curvedsky.alm2map(cmb_alm, cmb_raw, spin=spin)
	if 'p' in output:
		phi_map = enmap.empty(oshape[-2:], wcs, dtype=dtype)
		if verbose:
			print('Computing phi map')
		curvedsky.alm2map(phi_alm, phi_map)
	if 'k' in output:
		kappa_map = enmap.empty(oshape[-2:], wcs, dtype=dtype)
		kappa_alm = lensing.phi_to_kappa(phi_alm, phi_ainfo=phi_ainfo)
		curvedsky.alm2map(kappa_alm, kappa_map)
	if 'a' in output:
		grad_map = enmap.empty((2, *oshape[-2:]), wcs, dtype=dtype)
		curvedsky.alm2map(phi_alm, grad_map, deriv=True)

	# Output in same order as specified in output argument
	res = []
	for c in output:
		if   c == 'l': res.append(cmb_obs.squeeze())
		elif c == "u": res.append(cmb_raw.squeeze())
		elif c == "p": res.append(phi_map)
		elif c == "k": res.append(kappa_map)
		elif c == "a": res.append(grad_map)
	return tuple(res)