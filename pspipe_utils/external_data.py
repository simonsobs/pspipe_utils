"""
Some utility functions for handling external data.
"""
import numpy as np
from scipy.io import FortranFile
from pspy import so_spectra

from . import get_data_path

def get_bicep_BB_spectrum():
    """
    Read in the BICEP BB CMB only power spectrum, Fig 16 of https://arxiv.org/pdf/2110.00483.pdf
    """
    
    data = np.loadtxt(f"{get_data_path()}/spectra/bicep_keck/bk_B_modes.txt")
    bin_lo, lb, bin_hi, Db,  Db_low,  Db_high = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:, 5]
    y_err = np.stack([Db - Db_low, Db_high - Db])
    
    return lb, Db, y_err
    
def get_agora_spectrum(spec_name, comp1, comp2, spectrum=None):
    """
    We have computed the agora + pySM spectrum in DR6 bandpass, you can access the result here
    Parameters
    __________
    spec_name: str
      the name of the x_ar spectrum you want to consider e.g dr6_pa4_f220xdr6_pa5_f090
    comp1: str
      the first component in ["tsz","rksz","ksz","radio", "cib", "sync", "dust", "anomalous"]
    comp2: str
      the second component in ["tsz","rksz","ksz","radio", "cib", "sync", "dust", "anomalous"]
    spectrum: str
        optional return either "TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"

    """
    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    l, Dl = so_spectra.read_ps(f"{get_data_path()}/spectra/agora/Dl_{spec_name}_{comp1}x{comp2}.dat", spectra=spectra)
    
    if spectrum is not None:
        return l, Dl[spectrum]
    else:
        return l, Dl




def get_sptpol_BB_spectrum():
    """
    Read in the SPTPol BB CMB only power spectrum, Fig 2 of https://arxiv.org/pdf/1910.05748.pdf
    """
    
    bin_lo, bin_hi, lb, Db, sigmab = np.loadtxt(f"{get_data_path()}/spectra/sptpol/sptpol_B_modes.txt", unpack=True)
    return lb, Db,  sigmab

def get_polarbear_BB_spectrum():
    """
    Read in the POLARBEAR BB CMB only power spectrum, Fig 11 of https://iopscience.iop.org/article/10.3847/1538-4357/aa8e9f/pdf
    """
    
    lb, Db, sigmab = np.loadtxt(f"{get_data_path()}/spectra/polarbear/polarbear_B_modes.txt", unpack=True)
    return lb, Db,  sigmab




def get_choi_spectra(spec, survey="deep", return_Dl=True):
    """
    Read in the choi et al power spectra: https://arxiv.org/abs/2007.07289

    Parameters
    __________
    spec: str
      the spectrum to consider (e.g "TT", "TE"....)
    survey: str
      "deep", "wide"
    return_Dl: bool
      by default return Dl spectra (if False return Cl)
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

    data = np.loadtxt(f"{get_data_path()}/spectra/dr4/act_dr4.01_multifreq_{survey}_C_ell_{spec}.txt")
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

def get_planck_spectra(spec, return_Dl=True):
    """
    Read in the Planck legacy (PR3) power spectra: https://arxiv.org/abs/1907.12875

    Parameters
    __________
    spec: str
      the spectrum to consider (e.g "TT", "TE"....)
    return_Dl: bool
      by default return Dl spectra (if False return Cl)
    """

    if spec == "TT":
        fp_planck = ["100x100", "143x143", "143x217", "217x217"]
    elif spec in ["TE", "EE"]:
        fp_planck = ["100x100", "100x143", "100x217", "143x143", "143x217", "217x217"]

    l, cl, err = {}, {}, {}
    for fp in fp_planck:

        l[fp], cl[fp], err[fp] = np.loadtxt(f"{get_data_path()}/spectra/planck/planck_spectrum_{spec}_{fp}.dat", unpack=True)
        fac = l[fp] * (l[fp] + 1) / (2 * np.pi)

        if return_Dl:
            cl[fp], err[fp] = cl[fp] * fac, err[fp] * fac

    return fp_planck, l, cl, err
    
    
def get_planck_cmb_only_data():
    """
    Read the cmb only data corresponding to the likelihood 'like_cmbonly_plikv22'
    Return the data in Cls (not Dls)
    """

    planck_cmb_only_path =  f"{get_data_path()}/spectra/planck_cmb_only"

    nbin_TT = 215 #30-2508
    nbin_TE = 199 #30-1996
    nbin_EE = 199 #30-1996

    l, cl, sigma, cov_dict = {}, {}, {}, {}

    l["TT"], cl["TT"], sigma["TT"] = np.loadtxt(f"{planck_cmb_only_path}/cl_cmb_plik_v22_TT.dat", unpack=True)
    l["TE"], cl["TE"], sigma["TE"] = np.loadtxt(f"{planck_cmb_only_path}/cl_cmb_plik_v22_TE.dat", unpack=True)
    l["EE"], cl["EE"], sigma["EE"] = np.loadtxt(f"{planck_cmb_only_path}/cl_cmb_plik_v22_EE.dat", unpack=True)

    cov = FortranFile(f"{planck_cmb_only_path}/c_matrix_plik_v22.dat").read_reals()
    i = int(np.sqrt(len(cov)))
    cov = cov.reshape((i, i))
    cov = np.tril(cov) + np.tril(cov, -1).T

    start_TE = nbin_TT
    start_EE = nbin_TT + nbin_TE
    
    cov_dict["TTTT"] = cov[0:start_TE, 0:start_TE]
    cov_dict["TETE"] = cov[start_TE:start_EE, start_TE:start_EE]
    cov_dict["EEEE"] = cov[start_EE:start_EE + nbin_EE, start_EE:start_EE + nbin_EE]

    return l, cl, sigma, cov_dict

def bin_ala_planck_cmb_only(lth, ps_th):
    """
    Bin theory spectra using the weight of the Planck CMB-only likelihood
    Return spectra in Cls (not Dls)
    
    Parameters
    ----------
    lth: 1d integer array
         the multipoles
    ps_th: dict
         a dictionnary with CAMB theory power spectra
    """

    planck_cmb_only_path =  f"{get_data_path()}/spectra/planck_cmb_only"

    bin_low = np.loadtxt(f"{planck_cmb_only_path}/blmin.dat")
    bin_high= np.loadtxt(f"{planck_cmb_only_path}/blmax.dat")
    bin_weight = np.loadtxt(f"{planck_cmb_only_path}/bweight.dat")
    
    plmin  = 30
    nbin = {}
    nbin["TT"] = 215 #30-2508
    nbin["TE"] = 199 #30-1996
    nbin["EE"] = 199 #30-1996

    l_b, ps_b = {}, {}
    
    for spec in ["TT", "TE", "EE"]:
    
        cl = ps_th[spec] * 2 * np.pi / (lth * (lth+1))
    
        l_b[spec]= np.zeros(nbin[spec])
        ps_b[spec]= np.zeros(nbin[spec])
    
        for i in range(nbin[spec]):
            id = np.where( (lth >= plmin + bin_low[i]) & (lth <= plmin + bin_high[i])   )
            ps_select =  cl[id]
            w_select = bin_weight[int(bin_low[i]):int(bin_high[i]+1)]
        
            l_b[spec][i] = np.sum(lth[id] * w_select)
            ps_b[spec][i] = np.sum(ps_select * w_select)
    
    return l_b, ps_b


def get_passband_dict_dr6(wafer_list):
    """
    Read and return ACT DR6 passbands for
    wafers in wafer_list

    Parameters
    __________
    wafer_list: list[str]
      list of the different wafers considered
    """
    import h5py
    passband_dict = {}

    fname = f"{get_data_path()}/passbands/AdvACT_Passbands.h5"
    with h5py.File(fname, "r") as hfile:
        for wafer in wafer_list:
            pa, freq = wafer.split("_")
            data = hfile[f"{pa.upper()}_{freq}"]

            nu_ghz = data['frequencies'][()]
            freq_bounds = data['integration_bounds'][()]
            passband = data['mean_band'][()]
            passband /= nu_ghz ** 2

            # Truncate frequencies.
            freq_mask = (freq_bounds.min() < nu_ghz) & (nu_ghz < freq_bounds.max())

            passband_dict[wafer] = [nu_ghz[freq_mask], passband[freq_mask]]

    return passband_dict

def get_passband_dict_npipe(wafer_list, freq_range_list=None):
    """
    Read and return NPIPE passbands for
    wafers in wafer_list

    Parameters
    __________
    wafer_list: list[str]
      list of the different wafers considered
    freq_range_list: list[tuple]
      list of the frequency range to use for each wafer of wafer_list
    """
    import astropy.io.fits as fits
    passband_dict = {}

    fname = f"{get_data_path()}/passbands/HFI_RIMO_R4.00.fits"
    for i, wafer in enumerate(wafer_list):

        freq = wafer.split("_")[1].replace("f", "")

        with fits.open(fname) as hdu_list:
            data = hdu_list[f"BANDPASS_F{freq}"].data

            nu_ghz = data["WAVENUMBER"] * 1e-7 * 3e8 # conversion from cm^-1 to GHz
            passband = data["TRANSMISSION"]

            if freq_range_list:
                nu_min, nu_max = freq_range_list[i]
                freq_mask = (nu_ghz >= nu_min) & (nu_ghz <= nu_max)
                nu_ghz, passband = nu_ghz[freq_mask], passband[freq_mask]

            passband_dict[wafer] = [nu_ghz, passband]

    return passband_dict
