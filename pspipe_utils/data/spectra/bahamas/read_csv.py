import csv
import numpy as np
import pylab as plt
from scipy.interpolate import make_smoothing_spline,  make_splrep




def read_bahamas(high_AGN=False):
    l = np.arange(2, 10000)
    
    if high_AGN == True:
        file =  "BAHAMAS_tSZ_PS_Dellyy_highAGN.csv"
        nline = 51
    else:
        file =  "BAHAMAS_tSZ_PS_Dellyy_fiducial.csv"
        nline = 48

    with open(file) as csvfile:
        spamreader = csv.reader(csvfile)
        lb, Db = np.zeros((2, nline))

        for i, row in enumerate(spamreader):
            lb[i], Db[i] = row[0], row[1]
    spl = make_splrep(lb, Db, s=0.01)
    Dell = spl(l)
    Dell[Dell<0] = 0

    return l, Dell


l, Dell0 = read_bahamas(high_AGN=False)
l, Dell1 = read_bahamas(high_AGN=True)

plt.plot(l, Dell0)
plt.plot(l, Dell1)
plt.show()
