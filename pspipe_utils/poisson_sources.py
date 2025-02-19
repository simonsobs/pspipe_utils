"""
this is the pspipe version of Alexander van Engelen code which serve to compute equation 16 and 17 of https://arxiv.org/pdf/1310.7023.pdf at 148 GHz
"""

import numpy as np
import pylab as plt
from pixell import enmap, utils
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import rcParams
import fgspectra.frequency as fgf

from . import get_data_path


rcParams["xtick.labelsize"] = 13
rcParams["ytick.labelsize"] = 13
rcParams["axes.labelsize"] = 16
rcParams["axes.titlesize"] = 16

ref_freq_radio_GHz = 148
ref_freq_dusty_GHz = 217


tucci_file_name = f"{get_data_path()}/radio_source/ns_148GHz_modC2Ex.dat"

bethermin_file_name = f"{get_data_path()}/radio_source/Bethermin2012model_grids.save"

def read_bethermin_source_distrib(lam=1380.0, plot_fname=None):
    """
    Read in the source distribution from Bethermin et al 2012
    The model is described in this paper https://arxiv.org/abs/1208.6512 and the data products come from https://irfu.cea.fr/dap/Phocea/Page/index.php?id=537
    
    Parameters
    ----------
    lam : float
        The wavelenth, in um, at which we would like the counts.  Must be one of 
        160., 250., 350., 500., 550., 850., 1100., 1380., 2097., 3000., 210000.    
        Default is 1380 um, i.e. 217GHz.
    plot_fname : str
        if not None save the plot in plot_fname
    """
    from  scipy.io import readsav

    bethermin = readsav(bethermin_file_name)
    
    #Figure out the index of the Bethermin model grid of the wavelength that we are interested in.  Note that 1100 um = 220 GHz

    lam_idx = np.where(bethermin['lambda'   ] == lam)[0][0]
    
    #bethermin['dndsnudz'] has size [n_lambda, n_z, n_S] or namely [16, 210, 200].
    #Units are gal/Jy/sr, so there is actually a 'dOmega' in the denominator
    #Let's sum over redshift
    dz = np.gradient(bethermin['z'])
    #Note the broadcasting of dz here, from shape=(210) to shape=(210,1)
    dNdSdOmega = np.nansum(bethermin['dndsnudz'][lam_idx, :, :] * dz[:,None], axis = 0)

    
    S = bethermin['snu']
    
    
    if plot_fname is not None:
        plt.figure(figsize=(15, 10))
        plt.loglog()
        plt.ylim(1e-3, 200)
        plt.xlim(0.003, 1.5)
        plt.plot(S, dNdSdOmega * S ** (5/2), "--", color="red")
        plt.xlabel("S (Flux [Jy])", fontsize=16)
        plt.ylabel(r"$S^{5/2}\frac{dN}{dS d\Omega} [Jy^{3/2} sr^{-1}]$", fontsize=22)
        plt.savefig(plot_fname, bbox_inches="tight")
        plt.clf()
        plt.close()
        
    return S, dNdSdOmega

def read_tucci_source_distrib(plot_fname=None):
    """
    Read the source distribution from tucci et al (https://arxiv.org/pdf/1103.5707.pdf) with flux (S) at ref frequency 148 GHz
    and optionnaly plot it. Should replicate the C2Ex model (bottom long dashed line of Fig14)
    
    Parameters
    ----------
    plot_fname : str
        if not None save the plot in plot_fname
    """

    tucci = np.loadtxt(tucci_file_name)
    S = tucci[:, 0]
    dNdSdOmega = tucci[:, 1]
    
    if plot_fname is not None:
        plt.figure(figsize=(12, 10))
        plt.loglog()
        plt.ylim(0.6, 100)
        plt.xlim(0.001, 20)
        plt.plot(S, dNdSdOmega * S ** (5/2), "--", color="red")
        plt.xlabel("S (Flux [Jy])", fontsize=16)
        plt.ylabel(r"$S^{5/2}\frac{dN}{dS d\Omega} [Jy^{3/2} sr^{-1}]$", fontsize=22)
        plt.savefig(plot_fname, bbox_inches="tight")
        plt.clf()
        plt.close()
        
    return S, dNdSdOmega


def convert_Jy_per_str_to_muK_cmb(freq_GHz):
    """
    Convert from Jy/str to muK CMB at the observation frequency in GHz
    
    Parameters
    ----------
    freq_GHz : float
        the frequency of observation in GHz
    """
    K_to_Jy_per_str =  utils.dplanck(freq_GHz * 10 ** 9)
    
    return 10 ** 6 / K_to_Jy_per_str

def get_poisson_power(S, dNdSdOmega, plot_fname=None, ref_freq_GHz=ref_freq_radio_GHz):

    """
    Get the expected poisson power as a function of Smax: the maximal flux to consider in the integral
    optionally plot a_s, the radio source amplitude of D^{rad}_{ell} normalised at nu_0=148 GHz and l_0=3000
    as in Dunkley et al (https://arxiv.org/pdf/1301.0776.pdf)
    
    Parameters
    ----------
    S : 1d array
        source flux at the ref frequency in Jy
    dNdSdOmega : 1d array
        source distribution dN/(dS dOmega) corresponding to S
    plot_fname : str
        if not None save the plot in plot_fname
    ref_freq_GHz : float
        Reference frequency, defaults to ref_freq_ratio_GHz defined above
    """
    
    Jy_per_str_to_muK = convert_Jy_per_str_to_muK_cmb(ref_freq_GHz)
    dS = np.gradient(S)
    poisson_power = (Jy_per_str_to_muK) ** 2 * np.cumsum(dS * S ** 2 * dNdSdOmega)

    if plot_fname is not None:
        l0 = 3000 # pivot scale for the fg amplitude
        fac0 = (l0 * (l0 + 1))/(2 * np.pi)
        
        plt.figure(figsize=(12, 10))
        plt.loglog()
        plt.xlim(1, 1000)
        plt.ylim(10 ** -1, 5 * 10 ** 2)
        plt.plot(S * 10 ** 3, poisson_power * fac0)
        plt.xlabel(r"$S_{max}$ (Flux cut [mJy])", fontsize=16)
        plt.ylabel(r"$a_{s} [ {%d} GHz] $" % ref_freq_GHz, fontsize=22)
        plt.savefig(plot_fname, bbox_inches="tight")
        plt.clf()
        plt.close()
   
    return poisson_power

def get_trispectrum(S, dNdSdOmega, ref_freq_GHz = ref_freq_radio_GHz):

    """
    Get the expected trispectrum  as a function of Smax: the maximal flux to consider in the integral
    
    Parameters
    ----------
    S : 1d array
        source flux at the ref frequency in Jy
    dNdSdOmega : 1d array
        source distribution dN/(dS dOmega) corresponding to S
    ref_freq_GHz : Reference frequency at which to convert from Jy/Sr to uK.  If not explicitly given, this is assumed to be the value of ref_freq_radio_GHz above.
    """
    
    Jy_per_str_to_muK = convert_Jy_per_str_to_muK_cmb(ref_freq_GHz)
    dS = np.gradient(S)
    trispectrum = (Jy_per_str_to_muK) ** 4 * np.cumsum(dS * S ** 4 * dNdSdOmega)

    return trispectrum

def get_power_and_trispectrum_at_Smax(S, poisson_power, trispectrum, Smax):
    """
    Get the expected poisson power and trispectrum  evaluated at Smax by interpolating them
    
    Parameters
    ----------
    S : 1d array
        source flux at the ref frequency in Jy
    poisson_power : 1d array
        poisson power as a function of S=Smax
    trispectrum : 1d array
        trispectrum as a function of S=Smax
    Smax : float
        the maximum flux we consider, this correspond to the flux cut we use to define the mask
    """
    
    poisson_power_of_S = InterpolatedUnivariateSpline(S, poisson_power)
    trispectrum_of_S = InterpolatedUnivariateSpline(S, trispectrum)
    
    return poisson_power_of_S(Smax), trispectrum_of_S(Smax)

def get_mean_number_of_source(template, S, dNdSdOmega, plot_fname=None):

    """
    Get the mean number of sources in the patch defined by template
    
    Parameters
    ----------
    template : so_map
        the map template that will contain the source
    S : 1d array
        source flux at the ref frequency in Jy
    dNdSdOmega : 1d array
        source distribution dN/(dS dOmega) corresponding to S
    plot_fname : str
        if not None save the plot in plot_fname
    """

    dS = np.gradient(S)
    mean_numbers_per_patch = dNdSdOmega * enmap.area(template.data.shape, template.data.wcs) * dS
    
    if plot_fname is not None:
        plt.figure(figsize=(12, 10))
        plt.loglog()
        plt.xlim(1, 1000)
        plt.ylim(1, 10**4)
        plt.plot(S * 10 ** 3, mean_numbers_per_patch)
        plt.xlabel(r"$S$ (Flux [mJy])", fontsize=16)
        plt.ylabel(r"$N_{source}$", fontsize=22)
            
        plt.savefig(plot_fname, bbox_inches="tight")
        plt.clf()
        plt.close()

    return mean_numbers_per_patch
    
def radio_scaling(freq_GHz, alpha = -.5):

    """
    Scale the intensity of the signal as a function of frequency
    For example for power spectrum you will need this number ** 2 and for trispectrum this number ** 4
    
    Parameters
    ----------
    freq_GHz : float
        the frequency you want to scale the signal at
    alpha : float
        the slope of the radio SED
    """
    
    return (freq_GHz / ref_freq_radio_GHz) ** alpha * convert_Jy_per_str_to_muK_cmb(freq_GHz) / convert_Jy_per_str_to_muK_cmb(ref_freq_radio_GHz)


def CIB_scaling(freq_GHz, temp = 9.7, beta = 2.0):

    """
    Scale the intensity of the signal as a function of frequency
    For example for power spectrum you will need this number ** 2 and for trispectrum this number ** 4
    
    Parameters
    ----------
    freq_GHz : float
        the frequency you want to scale the signal at
    temp : float
        the CIB dust temperature
    beta: float
        Spectral index.
    """
    if not isinstance(freq_GHz, (float, int)):
        raise ValueError("You should pass a float or an integer not a list")

    sed_cib = fgf.CIB(**{'nu': freq_GHz, 'nu_0': ref_freq_dusty_GHz , 'temp': temp, 'beta': beta})
    
    return sed_cib.eval()[0]

