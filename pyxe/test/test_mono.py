# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:25:14 2016

@author: casim
"""
from collections import Counter
from nose.tools import assert_equal
import numpy as np
import os
import tempfile
from mock import patch, MagicMock
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pyxe.monochromatic import Mono


def cubic_multiplicity(h, k, l):
    hkl = Counter([h, k, l])
    m = {'h00': 6, 'hh0': 12, 'hk0': 24, 'hhh': 8, 'hkh': 24, 'hkl': 48}
    h = 'h'
    k = '0' if hkl[0] == 2 else 'k' if len(hkl) - hkl[0] >= 2 else 'h'
    l = '0' if hkl[0] >= 1 else 'h' if len(hkl) <= 2 else 'l'
    return h + k + l, m[h + k + l]


def structure_factor(h, k, l, f_1=1, f_2=1):
    pass

def strain_trans(e_xx, e_yy, e_xy, theta):
    e_xx_1 = ((e_xx + e_yy) / 2 + (e_xx - e_yy) * np.cos(2 * theta) / 2 +
              e_xy * np.sin(2 * theta))
    return e_xx_1


def miller_indices(struct='bcc'):
    simple = lambda h, k, l: True
    bcc = lambda h, k, l: (h + k + l) % 2
    fcc = lambda h, k, l: h % 2 == k % 2 == l % 2
    miller_rules = {'simple': simple, 'bcc': bcc, 'fcc': fcc}


def d0(struct='bcc', a=2.856):
    # a: lattice constant
    miller_indices(struct)
    return do, multiplicity


def lattice_to_ring_radius(detector='default', struct='bcc', a=2.856):

    d0, multiplicity = d0(struct, a)

    # find sample to detector
    # convert to radial positions
    rings = None
    return rings, multiplicity


def strained_rings(detector='default', rings=range(50, 250, 50),
                   intensity=None, e_xx=0, e_yy=0, e_xy=0):

    if detector == 'default':
        shape = (250, 250)
        xc, yc = shape[0] // 2, shape[1] // 2

    y, x = np.ogrid[:shape[0], :shape[1]]  # meshgrid without linspace
    y -= yc
    x -= xc
    r = np.sqrt(x ** 2 + y ** 2)  # simple pythagoras - radius of each pixel
    theta = np.cos(x / r)  # what angle are these pixels at

    e_xx_1 = strain_trans(e_xx, e_yy, e_xy, theta)

    img = np.zeros(shape)
    for idx, radius in enumerate(rings):
        rel_intensity = 1 if intensity is None else intensity[idx]
        radius += e_xx_1 * radius
        img += np.exp(-(r - radius) ** 2) * rel_intensity # recentre radius and apply gaussian

    return img


def strain_tensor_to_image_array(x, y, e_xx, e_yy, e_xy, a=2.856,
                                 struct='bcc', detector='default'):

    if detector == 'default':
        shape = 400, 400  # fake detector shape
        yc, xc = 200, 200  # fake detector centre pos

    rings, rel_int = lattice_to_ring_radius(detector, struct, a)
    images = np.zeros_like(x) # won't work - expand aray to account for image size
    for idx in np.ndenumerate(x):
        images[idx] = strained_rings(detector, rings, rel_int, e_xx[idx],
                                     e_yy[idx], e_xy[idx])
    return images




class FakeFabIO(object):
    def __init__(self, img):
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
#images = [FakeFabIO(shape, xc,  yc, strain=e) for e in strain.flatten()]

@patch("fabio.open")
@patch("pyxe.area_analysis.extract_fnames")
def test_import(extract_fnames, fabio_open):
    """
    Check that the correct number of files is being loaded.
    """
    # Will now return or yield a different im for each call to mock
#    fabio_open.side_effect = images
    extract_fnames.return_value = fnames

#    data = Mono('', co_ords, fit2Dparams, f_ext='.tif', progress=False,
#                npt_rad=shape[0]//2)
#    return data

if __name__ == '__main__':
    pass