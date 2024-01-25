import tempfile
import unittest

import numpy as np
import sacc
from pspipe_utils.io import port2sacc


class IOTest(unittest.TestCase):
    def test_port2sacc(self):
        cov_order = [("TT", "dr6_pa4_f220xdr6_pa4_f220"), ("TT", "dr6_pa4_f220xdr6_pa6_f090")]
        passbands = {"dr6_pa4_f220": [[220.0], [1.0]], "dr6_pa6_f090": [[90.0], [1.0]]}

        size = 50
        data = np.random.rand(size * len(cov_order))
        cov = np.random.rand(len(data), len(data))
        bin_low = np.arange(2, size + 2)
        bin_high = bin_low + 1
        bin_center = (bin_high + bin_low) / 2
        binning_file = tempfile.NamedTemporaryFile().name
        np.savetxt(binning_file, np.array([bin_low, bin_high, bin_center]).T)

        sacc_file_name = tempfile.NamedTemporaryFile().name
        port2sacc(
            data_vec=data,
            cov=cov,
            cov_order=cov_order,
            binning_file=binning_file,
            lmax=np.inf,
            passbands=passbands,
            sacc_file_name=sacc_file_name,
        )

        s = sacc.Sacc().load_fits(sacc_file_name)
        self.assertTrue(s.has_covariance())
        np.testing.assert_array_equal(s.covariance.covmat, cov)
        for name, tracer in s.tracers.items():
            self.assertTrue(tracer.ell.size == 0)
            self.assertTrue(tracer.beam.size == 0)

            name = "_".join(name.split("_")[:-1])
            np.testing.assert_array_equal(tracer.nu, passbands[name][0])
            np.testing.assert_array_equal(tracer.bandpass, passbands[name][1])
