import tempfile
import unittest

import numpy as np
from pspipe_utils import covariance as psc 


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
        chi2, ndof = psc.compute_chi2(**self.kwargs)
        self.assertAlmostEqual(chi2, np.sum(self.data_vec**2))
        self.assertEqual(ndof, len(self.data_vec))

    def test_compute_chi2_excluding_spectra(self):
        chi2, ndof = psc.compute_chi2(**self.kwargs, excluded_spectra=["TE", "EE"])
        self.assertAlmostEqual(chi2, np.sum(self.data_vec[:150] ** 2))
        self.assertEqual(ndof, 150)

    def test_compute_chi2_selecting_spectra(self):
        chi2, ndof = psc.compute_chi2(**self.kwargs, selected_spectra=["TE", "EE"])
        self.assertAlmostEqual(chi2, np.sum(self.data_vec[150:] ** 2))
        self.assertEqual(ndof, len(self.data_vec) - 150)

    def test_compute_chi2_excluding_arrays(self):
        chi2, ndof = psc.compute_chi2(
            **self.kwargs, excluded_spectra=["TE", "EE"], excluded_map_set=["ar2"]
        )
        self.assertAlmostEqual(chi2, np.sum(self.data_vec[:50] ** 2))
        self.assertEqual(ndof, 50)

    def test_compute_chi2_with_multipole_cuts(self):
        chi2, ndof = psc.compute_chi2(
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
        chi2, ndof = psc.compute_chi2(**self.kwargs, excluded_spectra=self.spectra_order)
        self.assertAlmostEqual(chi2, 0.0)
        self.assertEqual(ndof, 0)


class CovarianceUtilitiesTest(unittest.TestCase):
    
    def test_correct_analytical_cov_keep_res_diag(self):
        # these two matrices share orthogonal eigenbases so mc correction
        # gives mc
        ana_cov = np.array([[4, 0, 1], [0, 4, 0], [1, 0, 4]])
        mc_cov = np.array([[9, 0, 1], [0, 9, 0], [1, 0, 9]])

        cor_cov = psc.correct_analytical_cov_keep_res_diag(ana_cov, mc_cov, return_diag=False)
        test = np.allclose(cor_cov, mc_cov, rtol=1e-10, atol=0)
        self.assertTrue(test)

        cor_cov, res_diag = psc.correct_analytical_cov_keep_res_diag(ana_cov, mc_cov, return_diag=True)
        test1 = np.allclose(cor_cov, mc_cov, rtol=1e-10, atol=0)
        test2 = np.allclose(res_diag, np.array([8/3, 9/4, 10/5]))

        self.assertTrue(test1)
        self.assertTrue(test2)

    def test_canonize_connected_2pt(self):
        l = [i for i in 'abcdef']

        x, y = psc.canonize_connected_2pt('b', 'd', l)
        self.assertEqual((x, y), ('b', 'd'))

        x, y = psc.canonize_connected_2pt('e', 'c', l)
        self.assertEqual((x, y), ('c', 'e'))

    def test_canonize_disconnected_4pt(self):
        l = [i for i in 'abcdef']

        w, x, y, z = psc.canonize_disconnected_4pt('b', 'd', 'e', 'c', l)
        self.assertEqual((w, x, y, z), ('b', 'd', 'c', 'e'))

        w, x, y, z = psc.canonize_disconnected_4pt('f', 'e', 'b', 'a', l)
        self.assertEqual((w, x, y, z), ('a', 'b', 'e', 'f'))

        w, x, y, z = psc.canonize_disconnected_4pt('d', 'd', 'c', 'd', l)
        self.assertEqual((w, x, y, z), ('c', 'd', 'd', 'd'))

    def test_get_mock_noise_ps(self):
        out = psc.get_mock_noise_ps(10, 5, 2, -4)
        ans = np.array([40.0625,
                        40.0625,
                        40.0625,
                        8.71604938271605,
                        3.4414062499999996,
                        2.0,
                        1.4822530864197532,
                        1.260308204914619,
                        1.152587890625,
                        1.0952598689224204,
                        1.0625])
        test = np.allclose(out, ans, rtol=1e-10, atol=0)
        self.assertTrue(test)

    def test_bin_spec(self):
        bin_low = np.array([0, 2])
        bin_high = np.array([2, 5])
        specs = np.arange(25).reshape(5, 5)
        out = psc.bin_spec(specs, bin_low, bin_high)
        ans = np.array([[ 1.,  3.],
                        [ 6.,  8.],
                        [11., 13.],
                        [16., 18.],
                        [21., 23.]])
        test = np.allclose(out, ans, rtol=1e-10, atol=0)
        self.assertTrue(test)

    def test_bin_mat(self):
        bin_low = np.array([0, 2])
        bin_high = np.array([2, 5])
        mats = np.arange(75).reshape(3, 5, 5)
        out = psc.bin_mat(mats, bin_low, bin_high)
        ans = np.array([[[ 6.,  8.], [16., 18.]],
                        [[31., 33.], [41., 43.]],
                        [[56., 58.], [66., 68.]]])
        test = np.allclose(out, ans, rtol=1e-10, atol=0)
        self.assertTrue(test)

    def test_get_expected_pseudo_func(self):
        mcm = np.arange(9).reshape(3, 3)
        tf = np.array([0, 1, 4])
        ps = np.arange(1, 4)
        out = psc.get_expected_pseudo_func(mcm, tf, ps)(.5)
        ans = np.array([14, 38, 62])
        test = np.allclose(out, ans, rtol=1e-10, atol=0)
        self.assertTrue(test)

        bin_low = np.array([0, 1])
        bin_high = np.array([1, 2])
        out = psc.get_expected_pseudo_func(mcm, tf, ps, bin_low, bin_high)(.5)
        ans = np.array([26, 50])
        test = np.allclose(out, ans, rtol=1e-10, atol=0)
        self.assertTrue(test)

    def test_get_expected_cov_diag_func(self):
        mcm = np.arange(9).reshape(3, 3)
        w2 = .5
        tf = np.array([0, 1, 16])
        ps = np.arange(1, 4)
        coup = mcm
        out = psc.get_expected_cov_diag_func(mcm, w2, tf, ps, coup)(.5)
        ans = np.diag(2 * (np.array([14, 38, 62]) + np.array([14, 38, 62])[:, None])**2 * mcm)
        test = np.allclose(out, ans, rtol=1e-10, atol=0)
        self.assertTrue(test)

        bin_low = np.array([0, 1])
        bin_high = np.array([1, 2])
        pre_mcm_inv = 2*np.eye(3)
        out = psc.get_expected_cov_diag_func(mcm, w2, tf, ps, coup, bin_low, bin_high, pre_mcm_inv)(.5)
        ans = np.array([ 67840., 532224.])
        test = np.allclose(out, ans, rtol=1e-10, atol=0)
        self.assertTrue(test)

    def test_add_term_to_pseudo_cov(self):
        _temp = np.zeros((3, 3))
        S = np.arange(3)
        T = np.arange(1, 4)
        coup = np.arange(9).reshape(3, 3)
        
        ans = (S + S[:, None]) * (T + T[:, None]) * coup + coup
        psc.add_term_to_pseudo_cov(_temp, S, T, coup)

        out = coup.copy()
        psc.add_term_to_pseudo_cov(out, S, T, coup)

        test = np.allclose(out, ans, rtol=1e-10, atol=0)
        self.assertTrue(test)