# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:25:14 2016

@author: casim
"""

from nose.tools import assert_equal
import numpy as np
import os
import tempfile
from mock import patch

import matplotlib.pyplot as plt
from pyxe.area_analysis import Mono

shape = 1000, 1000 # fake detector shape
yc, xc = 500, 500 # fake detector centre pos
rings = range(100, 500, 100)

# representative detector parameters
sample2detector = 1000 # mm
wavelength = 0.01240 * 10**-9 # m (100keV)
fit2Dparams = sample2detector, xc, yc, 0, 0, 200, 200, wavelength

work_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(work_dir, r'temp_data')

class FakeFabIO(object):
    def __init__(self, shape=(1000,1000), xc=500, yc=500,
                 rings=range(100, 500, 100)):
        y, x = np.ogrid[:shape[0], :shape[1]]  # meshgrid without linspace
        y -= yc
        x -= xc
        r = np.sqrt(x ** 2 + y ** 2)
        img = np.zeros(shape)
        for radius in rings:
            img += np.exp(-(r - radius) ** 2)

        self.data = img

@patch("fabio.open")
@patch("pyxe.area_analysis.extract_fnames")
def test_import(mfun, mfun2):
    """
    Check that the correct number of files is being loaded.
    """
    mfun.return_value = ['x.tif', 'y.tif']
    mfun2.return_value = FakeFabIO(shape, xc,  yc)
    data = Mono(folder_path, np.array([[1,2], [1,2]]), fit2Dparams,
                f_ext='.tif', npt_rad=500, npt_az=36, az_range=(-180,180))
    #plt.plot(data.q[0], data.I[0, 0])
    #print(data.q.shape, data.I.shape)
    #plt.show()