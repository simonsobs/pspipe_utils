import unittest
from itertools import combinations_with_replacement as cwr
from itertools import product

import numpy as np


class SimulationTest(unittest.TestCase):
    def test_get_foreground_dict(self):
        ell = np.arange(2, 5000)
        frequencies = [90, 150, 220]

        fg_components = {
            "tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
            "te": ["radio", "dust"],
            "ee": ["radio", "dust"],
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
            "a_gee": 0.13,
            "a_psee": 0.05,
            "a_pste": 0,
            "xi": 0.1,
            "T_d": 9.60,
        }

        from pspipe_utils.simulation import get_foreground_dict

        fg_dict = get_foreground_dict(ell, frequencies, fg_components, fg_params)
        # Just check is the dict is correctly filled
        for k, v in fg_dict.items():
            self.assertEqual(v.size, ell.size)
        # for mode, components in fg_components.items():
        #     for component, (f1, f2) in product(components, cwr(frequencies, 2)):
        #         self.assertIn((mode, component, f1, f2), fg_dict)
