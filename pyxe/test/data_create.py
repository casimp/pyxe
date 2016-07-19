from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from pyxe.fitting_functions import gaussian


def gaussian_2d(height=1, centre=(0., -0.), sigma=(0.5, 0.5), pnts=(8, 8)):

    x = np.linspace(-1, 1, pnts[0])
    y = np.linspace(-1, 1, pnts[1])
    X, Y = np.meshgrid(x, y)

    Z = (gaussian(X, *(0, np.sqrt(height), centre[0], sigma[0])) *
         gaussian(Y, *(0, np.sqrt(height), centre[1], sigma[1])))

    return X, Y, Z


def test_tensor_array(pnts=(8, 8), max_strain=1e-2, flat=True):
    """ Creates a sensible, non-symmetric strain distribution for testing"""

    X, Y, e_xx = gaussian_2d(height=max_strain, centre=(0.25, -0.25),
                             sigma=(0.5, 0.5), pnts=pnts)

    e_yy = -gaussian_2d(height=max_strain, centre=(-0.25, 0.25),
                        sigma=(0.5, 0.5), pnts=pnts)[2]

    e_xy = gaussian_2d(height=max_strain / 10, centre=(0., 0.),
                       sigma=(0.5, 0.5), pnts=pnts)[2]

    if flat:

        X, Y = X.reshape(X.size, 1), Y.reshape(X.size, 1)
        e_xx, e_yy, e_xy = e_xx.flatten(), e_yy.flatten(), e_xy.flatten()

    return X, Y, e_xx, e_yy, e_xy


def create_ring_array(detector, pnts=(8, 8), max_strain=1e-3, exclude=0.1,
                      crop=0.6, background=0):
    # Create a strain (tensor) maps, find co-ords and image 'names'
    X, Y, e_xx, e_yy, e_xy = test_tensor_array(pnts, max_strain)
    co_ords = np.hstack((X, Y))
    fnames = ['{}.tif'.format(num) for num in range(e_xx.size)]

    # Create Debye-Scherrer rings for each point
    images = []
    for idx, i in enumerate(e_xx):
        # Representative detector but the s_to_d is small so we crop
        img = detector.rings(exclude, crop, background,
                             strain_tensor=(e_xx[idx], e_yy[idx], e_xy[idx]))
        images.append(FakeFabIO(img))

    return fnames, co_ords, images, (e_xx, e_yy, e_xy)


def data_dict(X, Y, q, I):
        d = {'entry1/scan_command': [b'ss2_x ss2_y'],
             'entry1/EDXD_elements/edxd_q': q,
             'entry1/EDXD_elements/data': I,
             'entry1/EDXD_elements/ss2_x': X,
             'entry1/EDXD_elements/ss2_y': Y}
        return d


def create_intensity_array(detector, pnts=(8, 8),
                           max_strain=1e-3, background=0):

    # Create a strain (tensor) maps, find co-ords and image 'names'
    X, Y, e_xx, e_yy, e_xy = test_tensor_array(pnts, max_strain, flat=False)

    # Create Debye-Scherrer rings for each point
    intensity = np.zeros(X.shape + (detector.phi.size, detector.q.size))
    print(intensity.shape)
    for idx in np.ndindex(e_xx.shape):
        # Representative detector but the s_to_d is small so we crop
        tensor = (e_xx[idx], e_yy[idx], e_xy[idx])

        data = detector.intensity(background=background,
                                  strain_tensor=tensor)[1]['total']
        intensity[idx] = data

    return data_dict(X, Y, data[0], intensity), (e_xx, e_yy, e_xy)



# pass [] as unused detector


class FakeFabIO(object):
    def __init__(self, img):
        self.data = img
