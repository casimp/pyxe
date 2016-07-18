# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:25:14 2016

@author: casim
"""
import matplotlib.pyplot as plt

import numpy as np
from mock import patch
from xrdpb.detectors import MonoDetector
from xrdpb.conversions import e_to_w
from pyxe.fitting_functions import gaussian
from pyxe.monochromatic import Mono


def gaussian_2d(height=1, centre=(0., -0.), sigma=(0.5, 0.5), pnts=8):

    x = np.linspace(-1, 1, pnts)
    y = np.linspace(-1, 1, pnts)
    X, Y = np.meshgrid(x, y)

    Z = (gaussian(X, *(0, height, centre[0], sigma[0])) *
         gaussian(Y, *(0, height, centre[1], sigma[1])))
    return X, Y, Z


class FakeFabIO(object):
    def __init__(self, img):
        self.data = img

# Detector parameters
shape = (2000, 2000)
crop = 0.825
cropped_shape = (shape[0] - shape[0] * crop, shape[1] - shape[1] * crop)
pixel_size = 0.2
sample_detector = 300
centerX = (cropped_shape[0] - 1) / 2
centerY = (cropped_shape[1] - 1) / 2
tilt=0
tiltPlanRotation=0
fit2Dparams = (sample_detector, centerX, centerY, tilt, tiltPlanRotation,
               pixel_size*1000, pixel_size*1000, e_to_w(100))

# Lets create a detector...
mono = MonoDetector(shape, pixel_size, sample_detector,
                    energy=100, energy_sigma=0.75)

# And add material/peak positions...
mono.add_peaks('Fe')

# Create a test strain tensor map
X, Y, e_xx = gaussian_2d(height=0.2, centre=(0.25, -0.25), sigma=(0.5, 0.5))
e_yy = -gaussian_2d(height=0.2, centre=(-0.25, 0.25), sigma=(0.5, 0.5))[2]
e_xy = gaussian_2d(height=0.02, centre=(0., 0.), sigma=(0.5, 0.5))[2]

X, Y = X.reshape(X.size, 1), Y.reshape(X.size, 1)
e_xx, e_yy, e_xy = e_xx.flatten(), e_yy.flatten(), e_xy.flatten()

co_ords = np.hstack((X, Y))
fnames = ['{}.tif'.format(num) for num in range(e_xx.size)]

# Create Debye-Scherrer rings for each point.
images = []
for idx, i in enumerate(e_xx):
    # Representative detector but the s_to_d is small so we crop
    img = mono.rings(exclude_criteria=0.1, crop=crop, background=0,
                             strain_tensor=(e_xx[idx], e_yy[idx], e_xy[idx]))
    images.append(FakeFabIO(img))

q0_image = [FakeFabIO(mono.rings(exclude_criteria=0.1, crop=crop, background=0,
                      strain_tensor=(0, 0, 0)))]
q0_name = ['q0.tif']
q0_co_ord = np.array([[0,0]])

@patch("fabio.open")
@patch("pyxe.monochromatic.extract_fnames")
def test_import(extract_fnames, fabio_open):
    """
    Check that the correct number of files is being loaded.
    """
    # Will now return or yield a different im for each call to mock
    fabio_open.side_effect = images
    extract_fnames.return_value = fnames

    data = Mono('', co_ords, fit2Dparams, f_ext='.tif', progress=False,
                npt_rad=cropped_shape[0]//2)

    fabio_open.side_effect = q0_image
    extract_fnames.return_value = q0_name

    q0 = Mono('', q0_co_ord, fit2Dparams, f_ext='.tif', progress=False,
                npt_rad=cropped_shape[0]//2)
    return q0, data

if __name__ == '__main__':
    q0, data = test_import()
    data.peak_fit(3.1, 0.3)
    q0.peak_fit(3.1, 0.3)
    data.calculate_strain(q0)
    plt.figure()
    data.plot_slice(data='shear strain', phi=0)
    plt.show()
    # d = data.extract_slice(phi=0)
    # plt.figure()
    # data.plot_slice(phi=0)
    # plt.show()
    # plt.figure()
    # data.plot_slice(phi=np.pi/2)
    # plt.show()
    # plt.figure()
    # data.plot_slice(data='shear strain', phi=np.pi / 2)
    # plt.show()
