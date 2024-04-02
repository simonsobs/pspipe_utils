import unittest
from itertools import combinations_with_replacement as cwr
from itertools import product

import numpy as np


class SimulationTest(unittest.TestCase):
    def test_get_foreground_dict(self):
        ell = np.arange(2, 5000)
        bandpass = {"exp_f090": [[90.0], [1.0]], "exp_f150": [[150], [1.0]]}

        fg_components = {
            "tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
            "te": ["radio", "dust"],
            "ee": ["radio", "dust"],
            "bb": ["radio", "dust"],
            "tb": ["radio", "dust"],
            "eb": [],
        }
        fg_params = {
            "a_tSZ": 3.30,
            "a_kSZ": 1.60,
            "a_p": 6.90,
            "beta_p": 2.08,
            "a_c": 4.90,
            "beta_c": 2.20,
            "a_s": 3.10,
            "a_gtt": 2.79,
            "a_gte": 0.36,
            "a_gtb": 0.36,
            "a_gee": 0.13,
            "a_gbb": 0.13,
            "a_psee": 0.05,
            "a_psbb": 0.05,
            "a_pste": 0,
            "a_pstb": 0,
            "xi": 0.1,
            "T_d": 9.60,
            "beta_s": -2.5,
            "alpha_s": 1.0,
            "T_effd": 19.6,
            "beta_d": 1.5,
            "alpha_dT": -0.6,
            "alpha_dE": -0.4,
            "alpha_p": 1.0,
        }

        from pspipe_utils.best_fits import get_foreground_dict

        fg_dict = get_foreground_dict(ell, bandpass, fg_components, fg_params)
        # Just check is the dict is correctly filled
        for k, v in fg_dict.items():
            self.assertEqual(v.size, ell.size)
        # for mode, components in fg_components.items():
        #     for component, (f1, f2) in product(components, cwr(frequencies, 2)):
        #         self.assertIn((mode, component, f1, f2), fg_dict)
