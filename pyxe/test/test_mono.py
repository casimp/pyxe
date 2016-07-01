# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:25:14 2016

@author: casim
"""

from nose.tools import assert_equal
import numpy as np
import os
import tempfile
from mock import patch, MagicMock
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pyxe.area_analysis import Mono

shape = 400, 400 # fake detector shape
yc, xc = 200, 200 # fake detector centre pos
rings = range(50, 200, 50)

# representative detector parameters
sample2detector = 1000 # mm
wavelength = 0.01240 * 10**-9 # m (100keV)
fit2Dparams = sample2detector, xc, yc, 0, 0, 200, 200, wavelength

#work_dir = os.path.dirname(os.path.abspath(__file__))
#folder_path = os.path.join(work_dir, r'temp_data')

class FakeFabIO(object):
    def __init__(self, shape=(500,500), xc=250, yc=250,
                 rings=range(50, 250, 50), strain=0):
        y, x = np.ogrid[:shape[0], :shape[1]]  # meshgrid without linspace
        y -= yc
        x -= xc
        r = np.sqrt(x ** 2 + y ** 2)
        img = np.zeros(shape)
        for radius in rings:
            radius += strain*radius
            img += np.exp(-(r - radius) ** 2)

        self.data = img

def random_strain_field(delta=0.2, max_strain=10**-3, plot=False):
    x = np.arange(-1.0, 1.0, delta)
    y = np.arange(-1.0, 1.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    # difference of Gaussians
    Z = 10 * max_strain * (Z2 - Z1)
    if plot:
        plt.contourf(X, Y, Z)
        plt.colorbar()
        plt.show()
    return X, Y, Z


# Lets create some data to test...
X, Y, strain = random_strain_field()
points = strain.size
co_ords = np.hstack((X.reshape(points, 1), Y.reshape(points, 1)))
fnames = ['{}.tif'.format(num) for num in range(points)]
images = [FakeFabIO(shape, xc,  yc, strain=e) for e in strain.flatten()]

@patch("fabio.open")
@patch("pyxe.area_analysis.extract_fnames")
def test_import(extract_fnames, fabio_open):
    """
    Check that the correct number of files is being loaded.
    """
    # Will now return or yield a different im for each call to mock
    fabio_open.side_effect = images
    extract_fnames.return_value = fnames

    data = Mono('', co_ords, fit2Dparams, f_ext='.tif', progress=False,
                npt_rad=shape[0]//2)
    return data