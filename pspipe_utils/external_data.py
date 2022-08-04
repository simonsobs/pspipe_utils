"""
Some utility functions for handling external data.
"""
import numpy as np

def get_choi_data(data_path, spec, survey="deep", return_Dl=True):
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
    elif spec == "BE"
         spec == "EB"
         fp_choi = ["90x90", "90x150", "150x150"]

    data = np.loadtxt(f"{data_path}/act_dr4.01_multifreq_{survey}_C_ell_{spec}.txt")
    l = data[:, 0]
    fac = l * (l + 1) / (2 * np.pi)
    cl, err = {}, {}
    
    for count, fp in enumerate(fp_choi):
        cl[fp] = data[:, 1 + 2 * count]
        err[fp] = data[:, 2 + 2 * count]
        if return_Dl:
            cl[fp], err[fp] = cl[fp] * fac, err[fp] * fac
        
    return fp_choi, l, cl, err
