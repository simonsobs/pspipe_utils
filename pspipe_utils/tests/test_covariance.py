import tempfile
import unittest

import numpy as np
from pspipe_utils.covariance import compute_chi2


class CovarianceTest(unittest.TestCase):
    def setUp(self):
        self.spectra_order = ["TT", "TE", "EE"]
        self.spec_name_list = ["ar1xar1", "ar1xar2", "ar2xar2"]
        nbins = 50
        nspec = nbins * len(self.spectra_order) * len(self.spec_name_list)
        self.data_vec = np.random.randn(nspec)
        self.theory_vec = np.zeros_like(self.data_vec)
        self.cov = np.identity(nspec)

        self.binning_file = tempfile.NamedTemporaryFile().name
        bin_low = np.arange(2, nbins + 2)
        bin_high = bin_low + 1
        bin_cent = (bin_high + bin_low) / 2
        np.savetxt(self.binning_file, np.array([bin_low, bin_high, bin_cent]).T)

        self.kwargs = dict(
            data_vec=self.data_vec,
            theory_vec=self.theory_vec,
            cov=self.cov,
            binning_file=self.binning_file,
            lmax=np.inf,
            spec_name_list=self.spec_name_list,
            spectra_order=self.spectra_order,
        )

    def test_compute_chi2(self):
        chi2, ndof = compute_chi2(**self.kwargs)
        self.assertAlmostEqual(chi2, np.sum(self.data_vec**2))
        self.assertEqual(ndof, len(self.data_vec))

    def test_compute_chi2_excluding_spectra(self):
        chi2, ndof = compute_chi2(**self.kwargs, excluded_spectra=["TE", "EE"])
        self.assertAlmostEqual(chi2, np.sum(self.data_vec[:150] ** 2))
        self.assertEqual(ndof, 150)

    def test_compute_chi2_selecting_spectra(self):
        chi2, ndof = compute_chi2(**self.kwargs, selected_spectra=["TE", "EE"])
        self.assertAlmostEqual(chi2, np.sum(self.data_vec[150:] ** 2))
        self.assertEqual(ndof, len(self.data_vec) - 150)

    def test_compute_chi2_excluding_arrays(self):
        chi2, ndof = compute_chi2(
            **self.kwargs, excluded_spectra=["TE", "EE"], excluded_map_set=["ar2"]
        )
        self.assertAlmostEqual(chi2, np.sum(self.data_vec[:50] ** 2))
        self.assertEqual(ndof, 50)

    def test_compute_chi2_with_multipole_cuts(self):
        chi2, ndof = compute_chi2(
            **self.kwargs,
            spectra_cuts={"ar1": {"T": [10, 25], "P": [0, 25]}},
            excluded_spectra=["TE"],
            excluded_map_set=["ar2"]
        )
        self.assertAlmostEqual(
            chi2, np.sum(self.data_vec[9:22] ** 2) + np.sum(self.data_vec[300:322] ** 2)
        )
        self.assertEqual(ndof, 13 + 22)

    def test_compute_chi2_excluding_everything(self):
        chi2, ndof = compute_chi2(**self.kwargs, excluded_spectra=self.spectra_order)
        self.assertAlmostEqual(chi2, 0.0)
        self.assertEqual(ndof, 0)
