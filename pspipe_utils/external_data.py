"""
Some utility functions for handling external data.
"""
import numpy as np

def get_choi_spectra(data_path, spec, survey="deep", return_Dl=True):
    """
    read in the choi et al power spectra

    Parameters
    __________
    data_path: str
      the path to the spectra file
    spec: str
      the spectrum to consider (e.g "TT", "TE"....)
    survey: str
      "deep", "wide"
    """

    if spec in ["TT", "EE", "EB", "BB"]:
        fp_choi = ["90x90", "90x150", "150x150"]
    elif spec in ["TE", "TB"]:
        fp_choi = ["90x90", "90x150", "150x90", "150x150"]
    elif spec in ["ET", "BT"]:
        spec = spec[::-1]
        fp_choi = ["90x90", "150x90", "90x150", "150x150"]
    elif spec == "BE":
         spec = "EB"
         fp_choi = ["90x90", "90x150", "150x150"]

    data = np.loadtxt(f"{data_path}/act_dr4.01_multifreq_{survey}_C_ell_{spec}.txt")
    l = data[:, 0]
    fac = l * (l + 1) / (2 * np.pi)
    l_choi, cl, err = {}, {}, {}

    for count, fp in enumerate(fp_choi):
        cl[fp] = data[:, 1 + 2 * count]
        err[fp] = data[:, 2 + 2 * count]
        if return_Dl:
            cl[fp], err[fp] = cl[fp] * fac, err[fp] * fac
        l_choi[fp] = l
        
    return fp_choi, l_choi, cl, err
    
def get_planck_spectra(data_path, spec, return_Dl=True):

    if spec == "TT":
        fp_planck = ["100x100", "143x143", "143x217", "217x217"]
    elif spec in ["TE", "EE"]:
        fp_planck = ["100x100", "100x143", "100x217", "143x143", "143x217", "217x217"]

    l, cl, err = {}, {}, {}
    for fp in fp_planck:

        l[fp], cl[fp], err[fp] = np.loadtxt(f"{data_path}/planck_spectrum_{spec}_{fp}.dat", unpack=True)
        fac = l[fp] * (l[fp] + 1) / (2 * np.pi)

        if return_Dl:
            cl[fp], err[fp] = cl[fp] * fac, err[fp] * fac

    return fp_planck, l, cl, err

def get_passband_dict_dr6(fname, wafer_list):
    """
    Read and return ACT DR6 passbands for
    wafers in wafer_list

    Parameters
    __________
    fname: str
      the path to passbands (h5 file)
    wafer_list: list[str]
      list of the different wafers considered
    """
    import h5py
    passband_dict = {}

    with h5py.File(fname,'r') as hfile:
        for wafer in wafer_list:
            pa, freq = wafer.split("_")

            nu_ghz = hfile[f'{pa.upper()}_{freq}']['frequencies'][()]
            freq_bounds = hfile[f'{pa.upper()}_{freq}']['integration_bounds'][()]
            passband = hfile[f'{pa.upper()}_{freq}']['mean_band'][()]
            passband /= nu_ghz ** 2

            # Truncate frequencies.
            freq_mask = (freq_bounds.min() < nu_ghz) & (nu_ghz < freq_bounds.max())

            passband_dict[wafer] = [nu_ghz[freq_mask], passband[freq_mask]]

    return passband_dict

def get_passband_dict_npipe(fname, wafer_list, freq_range_list = None):
    """
    Read and return NPIPE passbands for
    wafers in wafer_list

    Parameters
    __________
    fname: str
      the path to passbands (fits file)
    wafer_list: list[str]
      list of the different wafers considered
    freq_range_list: list[tuple]
      list of the frequency range to use for each wafer of wafer_list
    """
    import astropy.io.fits as fits
    passband_dict = {}

    for i, wafer in enumerate(wafer_list):

        freq = wafer.split("_")[1].replace("f", "")

        hdu_list = fits.open(fname)
        data = hdu_list[f"BANDPASS_F{freq}"].data

        nu_ghz = data["WAVENUMBER"] * 1e-7 * 3e8 # conversion from cm^-1 to GHz
        passband = data["TRANSMISSION"]

        if freq_range_list:
            nu_min, nu_max = freq_range_list[i]
            freq_mask = (nu_ghz >= nu_min) & (nu_ghz <= nu_max)
            nu_ghz, passband = nu_ghz[freq_mask], passband[freq_mask]

        passband_dict[wafer] = [nu_ghz, passband]

    return passband_dict
