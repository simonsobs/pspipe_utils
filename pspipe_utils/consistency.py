from itertools import combinations_with_replacement as cwr
from itertools import product
from getdist.mcsamples import loadMCSamples
from pspipe_utils import misc, covariance
from pspy import so_spectra, so_cov
import matplotlib.pyplot as plt
from cobaya.run import run
import scipy.stats as ss
import numpy as np
import re

def append_spectra_and_cov(ps_dict,
                           cov_dict,
                           spec_list):
    """
    Get the power spectra vector containing
    the spectra in `spec_list` with the
    associated covariance matrix.

    Parameters
    ----------
    ps_dict: dict
      dict containing the different power spectra
    cov_list: dict
      dict containing the covariances
    spec_list: list
      list containing the name of the spectra

    """
    n_bins = len(ps_dict[spec_list[0]])
    n_spec = len(spec_list)

    spectra_vec = np.zeros(n_bins * n_spec)
    full_cov = np.zeros((n_bins * n_spec,
                         n_bins * n_spec))

    for i, key1 in enumerate(spec_list):
        spectra_vec[i*n_bins:(i+1)*n_bins] = ps_dict[key1]
        for j, key2 in enumerate(spec_list):
            if j < i: continue
            try:
                full_cov[i*n_bins:(i+1)*n_bins,
                         j*n_bins:(j+1)*n_bins] = cov_dict[key1, key2]
            except:
                full_cov[i*n_bins:(i+1)*n_bins,
                         j*n_bins:(j+1)*n_bins] = cov_dict[key2, key1]

    full_cov = np.triu(full_cov)
    transpose_cov = full_cov.T
    full_cov += transpose_cov - np.diag(full_cov.diagonal())

    return spectra_vec, full_cov


def get_projector(n_bins,
                  proj_pattern):
    """
    Get the projection operator
    to apply to a spectra vector to
    get the residual power spectra

    Parameters
    ----------
    n_bins: int
      number of bins
    proj_pattern: 1D array
      If the spectra vector is a concatenation
      of three spectra psA, psB, psC; and
      proj_pattern is [A, B, C], applying the
      projector to a spectra vector will give :
        A * psA + B * psB + C * psC
    """
    N = len(proj_pattern)
    identity = np.identity(n_bins)

    projector = (np.tile(identity, (1, N)) *
                 np.tile(np.repeat(proj_pattern, n_bins), (n_bins, 1)))
    return projector


def project_spectra_vec_and_cov(spectra_vec,
                                full_cov,
                                proj_pattern,
                                calib_vec = None):
    """
    Get the residual spectrum and the associated
    covariance matrix from the spectra vector and
    the full covmat.

    Parameters
    ----------
    spectra_vec: 1D array
      spectra vector
    full_cov: 2D array
      full covariance matrix associated with spectra_vec
    proj_pattern: 1D array
      If the spectra vector is a concatenation
      of three spectra psA, psB, psC; and
      proj_pattern is [A, B, C], applying the
      projector to a spectra vector will give :
        A * psA + B * psB + C * psC
    calib_vec: 1D array
      calibration amplitudes to apply to the
      different power spectra
    """
    n_bins = len(spectra_vec) // len(proj_pattern)
    projector_uncal = get_projector(n_bins, proj_pattern)
    if calib_vec is None:
        calib_vec = np.ones(len(proj_pattern))
    projector_cal = get_projector(n_bins, proj_pattern * calib_vec)

    res_spectrum = projector_cal @ spectra_vec
    res_cov = projector_uncal @ full_cov @ projector_uncal.T

    return res_spectrum, res_cov

def get_chi2(spectra_vec,
             full_cov,
             proj_pattern,
             calib_vec = None,
             lrange = None):
    """
    Compute the chi2 of the residual
    power spectrum

    Parameters
    ----------
    spectra_vec: 1D array
      spectra vector
    full_cov: 2D array
      full covariance matrix associated with spectra_vec
    proj_pattern: 1D array (len = 3)
      If the spectra vector is a concatenation
      of three spectra psA, psB, psC; and
      proj_pattern is [A, B, C], applying the
      projector to a spectra vector will give :
        A * psA + B * psB + C * psC
    lrange: 1D array
    calib_vec: 1D array
      calibration amplitudes to apply to the
      different power spectra
    """
    res_spec, res_cov = project_spectra_vec_and_cov(spectra_vec, full_cov,
                                                    proj_pattern, calib_vec)
    if lrange is not None:
        return res_spec[lrange] @ np.linalg.inv(res_cov[np.ix_(lrange, lrange)]) @ res_spec[lrange]
    else:
        return res_spec @ np.linalg.inv(res_cov) @ res_spec

def plot_residual(lb,
                  res_ps_dict,
                  res_cov_dict,
                  mode,
                  title,
                  file_name,
                  lrange=None,
                  ylims=None,
                  l_pow=0,
                  overplot_theory_lines=None,
                  expected_res=0.,
                  return_chi2=False,
                  remove_dof=0):
    """
    Plot the residual power spectrum and
    save it at a png file

    Parameters
    ----------
    lb: 1D array
    res_ps_dict: dict
      Dict containing residual power spectra
    res_cov_dict: dict
      Dict containing residual covariance matrices
    mode: string
    title: string
    fileName: string
    lrange: 1D array
      selected multipole indices
    ylims: tuple
    l_pow: float
      apply a ell^{l_pow} scaling to the plot
    overplot_theory_lines: tuple (lb, Cl)
    expected_res: float
      Expected value for the residual
      ex: 0 for ps differences and 1 for a ratio of two ps
    return_chi2: bool
    """
    colors = ["darkorange", "navy", "forestgreen"]

    if overplot_theory_lines:
        lb_th, res_th = overplot_theory_lines
        assert len(lb) == len(lb_th), "Mismatch between expected residual and data"
    else:
        res_th = np.ones(len(lb)) * expected_res

    if return_chi2:
        chi2_dict = {}

    pte_list = []
    plt.figure(figsize=(8, 6))
    plt.axhline(expected_res, color="k", ls="--")
    for i, (name, res_cov) in enumerate(res_cov_dict.items()):
        if isinstance(res_ps_dict, dict):
            res_spec = res_ps_dict[name]
        else:
            res_spec = res_ps_dict
        if lrange is not None:
            chi2 = (res_spec[lrange] - res_th[lrange]) @ np.linalg.inv(res_cov[np.ix_(lrange, lrange)]) @ (res_spec[lrange] - res_th[lrange])
            ndof = len(lb[lrange])
        else:
            chi2 = (res_spec - res_th) @ np.linalg.inv(res_cov) @ (res_spec - res_th)
            ndof = len(lb)

        ndof -= remove_dof

        pte = 1 - ss.chi2(ndof).cdf(chi2)
        color = colors[i] if isinstance(res_ps_dict, dict) else "k"
        plt.errorbar(lb, res_spec * lb ** l_pow,
                     yerr=np.sqrt(res_cov.diagonal()) * lb ** l_pow,
                     ls="None", marker = ".", ecolor = colors[i],
                     color=color,
                     label=f"{name} [$\chi^2 = {{{chi2:.1f}}}/{{{ndof}}}$ (${{{pte:.4f}}}$)]")

        if return_chi2:
            chi2_dict[name] = {"chi2": chi2, "ndof": ndof}

        pte_list += [pte]

    if lrange is not None:
        xleft, xright = lb[lrange][0], lb[lrange][-1]
        plt.axvspan(xmin=0, xmax=xleft,
                    color="gray", alpha=0.7)
        if xright != lb[-1]:
            plt.axvspan(xmin=xright, xmax=lb[-1],
                        color="gray", alpha=0.7)

    if overplot_theory_lines:

        l_th, ps_th = overplot_theory_lines
        plt.plot(l_th, ps_th * l_th ** l_pow, color="gray")


    plt.title(title)
    plt.xlim(0, 1.05*lb[-1])
    if ylims is not None:
        plt.ylim(*ylims)
    plt.xlabel(r"$\ell$", fontsize=18)
    plt.ylabel(r"$\ell^{%d} \Delta D_\ell^\mathrm{%s}$" % (l_pow, mode), fontsize=18)
    plt.tight_layout()


    leg = plt.legend()
    for pte, text in zip(pte_list,leg.get_texts()):

        text.set_color("green")

        if (pte < 0.01) or (pte > 0.99):
            text.set_color("orange")
        if (pte < 0.001) or (pte > 0.999):
            text.set_color("red")



    plt.savefig(f"{file_name}.png", dpi=300)
    plt.clf()
    plt.close()

    if return_chi2:
        return chi2_dict

def get_calibration_amplitudes(spectra_vec,
                               full_cov,
                               proj_pattern,
                               mode,
                               lrange,
                               chain_name):
    """
    Get the calibration amplitude and the
    associated error

    Parameters
    ----------
    spectra_vec: 1D array
    full_cov: 2D array
    proj_pattern: 1D array (len = 3)
      If the spectra vector is a concatenation
      of three spectra psA, psB, psC; and
      proj_pattern is [A, B, C], applying the
      projector to a spectra vector will give :
        A * psA + B * psB + C * psC
    mode: string
    lrange: 1D array
    chain_name: string
    """
    cal_vec = {# Global calibration :
               #  multiplicative factor to apply to
               # the different cross spectra to obtain
               # a calibration amplitude. Here the cross
               # spectra are [AxA, AxR, RxR] with A the
               # array you want to calibrate, and R the
               # reference array
              "TT": lambda c: np.array([c**2, c, 1]),
              # Pol. Eff.
              "EE": lambda e: np.array([e**2, e, 1]),
              "TE": lambda e: np.array([e, 1, 1]),
              "ET": lambda e: np.array([e, e, 1])}

    def logL(cal):
        if (proj_pattern == np.array([1, -1, 0])).all():
            if mode == "TT" or mode == "EE":
                # If we want to fit for the calib (resp. pol.eff.)
                # using the combination (c**2)AxA - c AxR,
                # we set the multiplicative factor to be
                # [c, 1, 1] such that we are minimizing
                # the residual c AxA - AxR
                calib_vec = np.array([cal, 1, 1])
        else:
            calib_vec = cal_vec[mode](cal)

        chi2 = get_chi2(spectra_vec, full_cov, proj_pattern, calib_vec, lrange)
        return -0.5 * chi2

    info = {
        "likelihood": {"my_like": logL},
        "params": {
            "cal": {
                "prior": {
                    "min": 0.5,
                    "max": 1.5
                         },
                "proposal": 1e-2,
                "latex": "c"
                 }
                  },
        "sampler": {
            "mcmc": {
                "max_tries": 1e4,
                "Rminus1_stop": 0.03,
                "Rminus1_cl_stop": 0.07,
                    }
                   },
        "output": chain_name,
        "force": True
           }

    updated_info, sampler = run(info)
    samples = loadMCSamples(chain_name, settings = {"ignore_rows": 0.5})
    cal_mean = samples.mean("cal")
    cal_std = np.sqrt(samples.cov(["cal"])[0, 0])

    return cal_mean, cal_std


def get_ps_and_cov_dict(ar_list,
                        ps_template,
                        cov_template,
                        spectra_order=["TT", "TE", "ET", "EE"],
                        skip_auto=False):
    """
    Load power spectra and covariances for
    arrays listed in `ar_list`.

    Parameters
    ----------
    ar_list: 1d array [str]
        List of the arrays we want to use
    ps_template: str
        Template for the name of the power spectra files
        ex : "spectra/Dl_{}x{}_cross.dat"
    cov_template: str
        Template for the name of the covariance files
        ex : "covariances/analytic_cov_{}x{}_{}x{}.npy"
    spectra_order: list
    skip_auto: bool
    """
    ps_dict = {}
    cov_dict = {}

    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    for i, (ar1, ar2) in enumerate(cwr(ar_list, 2)):

        if skip_auto and (ar1==ar2): continue
        try:
            tuple_name1 = (ar1, ar2)
            ps_file = ps_template.format(*tuple_name1)
            lb, ps = so_spectra.read_ps(ps_file, spectra = spectra)
        except:
            tuple_name1 = (ar2, ar1)
            ps_file = ps_template.format(*tuple_name1)
            lb, ps = so_spectra.read_ps(ps_file, spectra = spectra)

        ps_dict = {**ps_dict, **{(*tuple_name1, m): ps[m] for m in spectra_order}}

        for j, (ar3, ar4) in enumerate(cwr(ar_list, 2)):
            if skip_auto and (ar3==ar4): continue
            if i < j: continue

            try:
                tuple_name2 = (ar3, ar4)
                try:
                    tuple_order = [tuple_name1, tuple_name2]
                    cov_file = cov_template.format(*tuple_name1, *tuple_name2)
                    cov = np.load(cov_file)
                except:
                    tuple_order = [tuple_name2, tuple_name1]
                    cov_file = cov_template.format(*tuple_name2, *tuple_name1)
                    cov = np.load(cov_file)
            except:
                tuple_name2 = (ar4, ar3)
                try:
                    tuple_order = [tuple_name1, tuple_name2]
                    cov_file = cov_template.format(*tuple_name1, *tuple_name2)
                    cov = np.load(cov_file)
                except:
                    tuple_order = [tuple_name2, tuple_name1]
                    cov_file = cov_template.format(*tuple_name2, *tuple_name1)
                    cov = np.load(cov_file)
            t1, t2 = tuple_order
            for m1, m2 in product(spectra_order, spectra_order):
                cov_dict[(*t1, m1), (*t2, m2)] = so_cov.selectblock(cov, spectra_order, n_bins=len(lb), block = m1+m2)

    ps_dict["ell"] = lb

    return ps_dict, cov_dict


def compute_ps_and_cov_ratio(ps_dict,
                             cov_dict,
                             spec_list):
    """
    Compute the ratio between two power spectra
    and the associated covariance matrix.

    Parameters
    ----------
    ps_dict: dict
    cov_dict: dict
    spec_list: 1d array [str] (len = 2)
        List containing the two power spectra for
        which we want to compute the ratio
    """
    XY, WZ = spec_list

    snr = ps_dict[WZ] / np.sqrt(cov_dict[WZ, WZ].diagonal())
    snr_cut = np.where(snr >= 3)[0]

    bias = cov_dict[WZ, WZ] / np.outer(ps_dict[WZ], ps_dict[WZ])
    try:
        cross_name = (XY, WZ)
        bias -= cov_dict[cross_name] / np.outer(ps_dict[XY], ps_dict[WZ])
    except:
        cross_name = (WZ, XY)
        bias -= cov_dict[cross_name] / np.outer(ps_dict[XY], ps_dict[WZ])

    ratio = ps_dict[XY] / ps_dict[WZ] * (1 - bias.diagonal())

    cov = cov_dict[WZ, WZ] / np.outer(ps_dict[WZ], ps_dict[WZ])
    cov += cov_dict[XY, XY] / np.outer(ps_dict[XY], ps_dict[XY])
    cov -= 2 * cov_dict[cross_name] / np.outer(ps_dict[XY], ps_dict[WZ])

    cov *= np.outer(ratio, ratio)

    return ratio[snr_cut], cov[np.ix_(snr_cut, snr_cut)], snr_cut



def compare_spectra(ar_list,
                    op,
                    ps_dict,
                    cov_dict,
                    mode = "TT",
                    return_chi2 = True):
    """
    Compare two power spectra according
    to the operation specified in `op`.

    Parameters
    ----------
    ar_list: 1d array [str]
        List of the arrays [arA, arB, ...]
    op: str
        Symbolic operation to perform on the power spectra.
        ex: "ab-aa" where "a" corresponds to the first element of
            `ar_list` and "b" corresponds to the second.
        Supported operations are :
            - PS differences : "ab-cd"
            - Map differences : "aa+bb-2ab"
            - PS ratio : "ab/cd"
    ps_dict: dict
    cov_dict: dict
    mode: str
        Default: "TT"
    """
    required_names = sorted(set(re.findall("[A-Za-z]", op)))
    if len(required_names) > len(ar_list):
        raise ValueError(f"You have to provide {len(required_names)} names to perform this comparison.")

    names_dict = {alias: ar_list[i] for i, alias in enumerate([chr(i) for i in range(ord("a"),ord("a")+len(ar_list))])}
    spec_list = []

    op_is_ratio = False

    # Power spectra difference
    if re.match("[a-z]{2}-[a-z]{2}", op):
        proj_pattern = np.array([1, -1])
        for cross in op.split("-"):
            ps_name = (names_dict[cross[0]], names_dict[cross[1]])
            if ps_name + (mode,) in ps_dict.keys():
                spec_list.append(ps_name + (mode,))
            else:
                spec_list.append(ps_name[::-1] + (mode,))

    # Map difference
    elif re.match("[a-z]{2}\+[a-z]{2}-2[a-z]{2}", op):
        proj_pattern = np.array([1, -2, 1])
        map1 = names_dict[op[0]]
        map2 = names_dict[op[3]]

        if (map1, map2, mode) in ps_dict.keys():
            spec_list = [(map1, map1, mode), (map1, map2, mode), (map2, map2, mode)]
        else:
            spec_list = [(map1, map1, mode), (map2, map1, mode), (map2, map2, mode)]

    # Power spectra ratio
    elif re.match("[a-z]{2}/[a-z]{2}", op):
        op_is_ratio = True
        for cross in op.split("/"):
            ps_name = (names_dict[cross[0]], names_dict[cross[1]])
            if ps_name + (mode,) in ps_dict.keys():
                spec_list.append(ps_name + (mode,))
            else:
                spec_list.append(ps_name[::-1] + (mode,))
    else:
        raise ValueError(f"Invalid operation : {op}")

    ps_vec, full_cov = append_spectra_and_cov(ps_dict, cov_dict, spec_list)
    lb = ps_dict["ell"]

    if op_is_ratio:
        expect = 1
        res_ps, res_cov, snr_cut = compute_ps_and_cov_ratio(ps_dict, cov_dict, spec_list)
        lb = lb[snr_cut]
    else:
        expect = 0
        res_ps, res_cov = project_spectra_vec_and_cov(ps_vec, full_cov, proj_pattern)


    if return_chi2:
        chi2 = (res_ps - expect) @ np.linalg.inv(res_cov) @ (res_ps - expect)
        pte = 1 - ss.chi2.cdf(chi2, len(lb))
        return lb, res_ps, res_cov, chi2, pte
    else:
        return lb, res_ps, res_cov
