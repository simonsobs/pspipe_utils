import unittest
from itertools import combinations_with_replacement as cwr
from itertools import product

import numpy as np


class SimulationTest(unittest.TestCase):
    def test_get_foreground_dict(self):
        ell = np.arange(5000)
        beam = ell[::-1] / ell[-1]
        bandpass = {
            "exp_f090": np.array([[90, 95], [0.5, 0.5]]),
            "exp_f150": np.array([[150, 155], [0.5, 0.5]]),
        }
        beams = {
            "exp_f090_s0": {"nu": np.array([90, 95]), "beams": np.array([beam, beam])},
            "exp_f150_s0": {"nu": np.array([150, 155]), "beams": np.array([beam, beam])},
        }
        beams["exp_f090_s2"] = beams["exp_f090_s0"]
        beams["exp_f150_s2"] = beams["exp_f150_s0"]
        # beams = None

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
            "alpha_tSZ": 0.0,
        }

        from pspipe_utils.best_fits import get_foreground_dict

        fg_dict = get_foreground_dict(ell, bandpass, fg_components, fg_params, beams=beams)
        # Just check is the dict is correctly filled
        for k, v in fg_dict.items():
            self.assertEqual(v.size, ell.size)
        for mode, components in fg_components.items():
            for component, (exp1, exp2) in product(components, cwr(bandpass, 2)):
                if component == "tSZ_and_CIB":
                    continue
                self.assertIn((mode, component, exp1, exp2), fg_dict)
