# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:25:14 2016

@author: casim
"""
import matplotlib.pyplot as plt
from nose.tools import assert_raises
import numpy as np
from mock import patch
from xrdpb.detectors import MonoDetector
from xrdpb.conversions import e_to_w
from pyxe.monochromatic import Mono
from pyxe.fitting_functions import strain_transformation
from pyxe.test.data_create import create_ring_array

# Detector parameters
shape = (2000, 2000)  # pixels
crop = 0.6  # relative
pixel_size = 0.2  # mm
sample_detector = 1000  # mm
energy = 100  # keV
energy_sigma = 1  # keV
crop_shape = ((1 - crop) * shape[0], (1 - crop) * shape[1])
fit2Dparams = (sample_detector, crop_shape[0] / 2, crop_shape[1] / 2, 0, 0,
               pixel_size*1000, pixel_size*1000)

# Lets create a detector and add peaks...
mono = MonoDetector(shape, pixel_size, sample_detector, energy, energy_sigma)
mono.add_peaks('Fe')

max = 1e-2
# Create the test data from that setup
fnames, co_ords, images, tensor = create_ring_array(mono, crop=crop, max_strain=1e-3)
e_xx, e_yy, e_xy = tensor
q0 = create_ring_array(mono, crop=crop, pnts=(1, 1), max_strain=0)
q0_fnames, q0_co_ords, q0_images, _ = q0


class TestMono(object):

    @patch("fabio.open")
    @patch("pyxe.monochromatic.extract_fnames")
    def setUp(self, extract_fnames, fabio_open):
        fabio_open.side_effect = images
        extract_fnames.return_value = fnames

        self.data = Mono('', co_ords, fit2Dparams, e_to_w(energy),
                         f_ext='.tif', progress=False,
                         npt_rad=crop_shape[0] / 2)

        fabio_open.side_effect = q0_images
        extract_fnames.return_value = q0_fnames

        self.q0 = Mono('', q0_co_ords, fit2Dparams, e_to_w(energy),
                       f_ext='.tif', progress=False, npt_rad=crop_shape[0] / 2)


    def tearDown(self):
        pass

    def test_peak_fit(self):
        self.data.peak_fit(3.1, 1.)

    def test_strain_calc(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)

    def test_extract_slice(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)
        self.data.extract_slice('strain', phi=0)
        assert_raises(AssertionError, self.data.extract_slice, 'strain err', 0)
        self.data.extract_slice('shear strain', phi=5 * np.pi)
        self.data.extract_slice('strain', phi=np.pi / 3)
        self.data.extract_slice('strain', az_idx=3)

    def test_positions(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)
        # Compare angles, same position
        for idx in [0, 7, 12, 26, 32]:
            initial = strain_transformation(self.data.phi[idx],
                                            e_xx, e_yy, e_xy)
            processed_1 = self.data.strain[..., idx]
            processed_2 = self.data.extract_slice(phi=self.data.phi[idx])
            assert np.allclose(processed_1, initial, atol=1e-5, rtol=0), idx
            assert np.allclose(processed_2, initial, atol=1e-5, rtol=0), idx


    def test_angles(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)
        # Compare angles, same position
        for p_idx in [0, 7, 12, 26, 39, 45, 52, 60]:
            initial = strain_transformation(self.data.phi, e_xx[p_idx],
                                            e_yy[p_idx], e_xy[p_idx])
            processed = self.data.strain[p_idx]
            assert np.allclose(initial, processed, atol=1e-5, rtol=0), p_idx



# def test_import(extract_fnames, fabio_open):
#     """
#     Check that the correct number of files is being loaded.
#     """
#     # Will now return or yield a different im for each call to mock
#     fabio_open.side_effect = images
#     extract_fnames.return_value = fnames
#
#     data = Mono('', co_ords, fit2Dparams, f_ext='.tif', progress=False,
#                 npt_rad=crop_shape[0] / 2)
#
#     fabio_open.side_effect = q0_images
#     extract_fnames.return_value = q0_fnames
#
#     q0 = Mono('', q0_co_ords, fit2Dparams, f_ext='.tif', progress=False,
#               npt_rad=crop_shape[0] / 2)
#     return q0, data
#
# if __name__ == '__main__':
#     q0, data = test_import()
#     data.peak_fit(3.1, 1.)
#     q0.peak_fit(3.1, 1.)
#     data.calculate_strain(q0)
#
#     # Compare positions, same angle
#     idx = 13
#     plt.plot(data.extract_slice(phi=data.phi[idx]), label='{}_phi'.format(idx))
#     plt.plot(data.strain[..., idx], label='{}_raw'.format(idx))
#     plt.plot(data.extract_slice(az_idx=idx), label='{}_raw_b'.format(idx))
#     plt.plot(strain_transformation(data.phi[idx], e_xx, e_yy, e_xy), label='{}trans_tensor'.format(idx))
#     plt.legend()
#     plt.show()
#
#     p_idx=39
#     # Compare angles, same position
#     plt.plot(data.strain[p_idx], label='posn_{}_raw'.format(p_idx))
#     plt.plot(strain_transformation(data.phi, e_xx[p_idx], e_yy[p_idx], e_xy[p_idx]),
#                                    label='posn_{}trans_tensor'.format(p_idx))
#     plt.legend()
#     plt.show()