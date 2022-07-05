"""
"""
from pspy import  so_map
import numpy as np
arrays = ["pa4_f150", "pa4_f220", "pa5_f090", "pa5_f150", "pa6_f090", "pa6_f150"]

for ar in arrays:
    binary = so_map.read_map("binary_dr6_%s.fits" % ar)
    binary = binary.downgrade(8)
    binary.data[binary.data >= 0.5] = 1
    binary.data[binary.data <0.5] = 0
    binary.data = binary.data.astype(np.int8)
    binary.write_map("binary_dr6_%s_downgraded.fits" % ar)
    binary.plot()

