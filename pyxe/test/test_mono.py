# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:25:14 2016

@author: casim
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from nose.tools import assert_raises
import numpy as np
from mock import patch
from pyxpb.detectors import MonoDetector
from pyxpb.conversions import e_to_w
from pyxe.monochromatic import MonoPyFAI
from pyxe.fitting_functions import strain_transformation
from pyxpb.array_create import ring_array
from pyxe.merge import ordered_merge
import matplotlib.pyplot as plt
from pyxe.peak_analysis import PeakAnalysis
from pyxe.plotting import DataViz

from itertools import product


class FakeFabIO(object):
    def __init__(self, img):
        self.data = img


def faked_data(detector, pnts=(7, 7), max_strain=1e-3, crop=0.5):
    d_ = ring_array(detector, pnts, max_strain=max_strain, crop=crop)
    x, y, ims, (e_xx, e_yy, e_xy) = d_
    fnames_ = ['{}.tif'.format(num) for num in range(e_xx.size)]
    co_ords_ = np.hstack((x.reshape(x.size, 1), y.reshape(y.size, 1)))
    images_ = []
    for idx in np.ndindex(e_xx.shape):
        images_.append(FakeFabIO(ims[idx]))
    # Need to convert nd image array to list of faked images
    # images_ = images.reshape(-1, images.shape[-2], images.shape[-1])
    # images_ = [FakeFabIO(img[0]) for img in images]
    tensor_ = (e_xx.flatten(), e_yy.flatten(), e_xy.flatten())
    return fnames_, co_ords_, images_, tensor_

# Detector parameters
shape = (2000, 2000)  # pixels
crop = 0.6  # relative
pixel_size = 0.2  # mm
sample_detector = 1000  # mm
energy = 100  # keV
energy_sigma = 1  # keV
crop_shape = ((1 - crop) * shape[0], (1 - crop) * shape[1])
fit2Dparams = (sample_detector, crop_shape[0] / 2, crop_shape[1] / 2,
               0, 0, pixel_size * 1000, pixel_size * 1000)

# Lets create a detector and add peaks...
mono = MonoDetector(shape, pixel_size, sample_detector, energy, energy_sigma)
mono.add_material('Fe')

# Create the test data from that setup
fnames, co_ords, images, tensor = faked_data(mono, (6, 6), crop=crop,
                                             max_strain=1e-3)
e_xx, e_yy, e_xy = tensor

# Create associated q0 data
q0_fnames, q0_co_ords, q0_images, _ = faked_data(mono, (1,1), crop=crop,
                                                 max_strain=0)


@patch("fabio.open")
@patch("pyxe.monochromatic.extract_fnames")
def integration(extract_fnames, fabio_open):
    fabio_open.side_effect = images
    extract_fnames.return_value = fnames
    data_ = MonoPyFAI('', co_ords, fit2Dparams, e_to_w(energy), f_ext='.tif',
                 progress=False, npt_rad=crop_shape[0] / 2)

    fabio_open.side_effect = q0_images
    extract_fnames.return_value = q0_fnames
    q0_ = MonoPyFAI('', q0_co_ords, fit2Dparams, e_to_w(energy), f_ext='.tif',
               progress=False, npt_rad=crop_shape[0] / 2)

    return data_, q0_

def peak_fit():
    data, q0 = integration()
    data.peak_fit(3.1, 1.)
    q0.peak_fit(3.1, 1.)
    return data, q0


#class Dep_TestMono(object):
#
#    data, q0 = peak_fit()
#
#    def test_basic_merge(self):
#        merged = self.data + self.data
#        assert np.array_equal(merged.phi, self.data.phi)
#        assert merged.peaks.size == 2 * self.data.peaks.size
#
#    def test_strain_calc(self):
#        self.data.calculate_strain(self.q0)
#
#    def test_basic_plot(self):
#        self.data.calculate_strain(self.q0)
#        self.data.plot_intensity()
#        self.data.plot_strain_fit()
#        plt.close()
#
#    def test_stress_calc(self):
#        self.data.calculate_strain(self.q0)
#        self.data.material_parameters(E=200*10**9, v=0.3)
#
#    def test_extract_slice(self):
#        self.data.calculate_strain(self.q0)
#        self.data.material_parameters(E=200 * 10 ** 9, v=0.3)
#        # Try a few slice extract options
#        self.data.extract_slice('strain', phi=np.pi/3)
#        assert_raises(AssertionError, self.data.extract_slice, 'strain err', 0)
#        self.data.extract_slice('shear stress', phi=5*np.pi)
#        self.data.extract_slice('peaks err', az_idx=3)
#
#    def test_positions(self):
#        self.data.calculate_strain(self.q0)
#        # Compare positions, same angle
#        for idx in [0, 7, 12, 26, 32]:
#            tensor = e_xx, e_yy, e_xy
#            initial = strain_transformation(self.data.phi[idx], *tensor)
#            processed_1 = self.data.strain[..., idx]
#            processed_2 = self.data.extract_slice(phi=self.data.phi[idx])
#            for processed in [processed_1, processed_2]:
#                max_diff = np.abs(np.max(initial - processed))
#                assert max_diff < 10**-4, (idx, max_diff)  # Brittle (linux)?!
#
#    def test_angles(self):
#        self.data.calculate_strain(self.q0)
#        # Compare angles, same position
#        for p_idx in [0, 7, 12, 26, 32]:
#            tensor = e_xx[p_idx], e_yy[p_idx], e_xy[p_idx]
#            initial = strain_transformation(self.data.phi, *tensor)
#            processed = self.data.strain[p_idx]
#            max_diff = np.abs(np.max(initial - processed))
#            assert max_diff < 10**-4, (p_idx, max_diff)  # Brittle (linux)?!
#
#    def test_ordered_merge(self):
#        self.data.calculate_strain(self.q0)
#        data2 = copy.deepcopy(self.data)
#
#        pad = 0.35
#        shift = 1.00001
#        data2.d1 += shift
#        added = np.sum(data2.d1 > (1 + pad))
#
#        merged = ordered_merge([self.data, data2], [0, 1], pad)
#        assert np.array_equal(merged.phi, self.data.phi)
#        assert merged.d1.size == added + self.data.d1.size, (
#        merged.d1.size, added, self.data.d1.size)
#
#    def test_plot_line(self):
#        self.data.calculate_strain(self.q0)
#        self.data.material_parameters(E=200 * 10 ** 9, v=0.3)
#        data2 = copy.deepcopy(self.data)
#        data2.d1 += 1.0001
#        merged = ordered_merge([self.data, data2], [0, 1])
#
#        # Try a few slice extract options
#        phi_names = ['strain', 'stress', 'shear strain', 'shear stress']
#        az_names = ['peaks', 'fwhm', 'strain', 'stress', 'peak err',
#                    'fwhm err', 'strain err']
#        phi_, az_ = [-2 * np.pi], [20]
#        pnt_, theta_ = [(-0.2, 0.2)], [0, -np.pi / 3]
#        d_ = [self.data, merged]
#
#        iterator = product(d_, phi_names, phi_, pnt_, theta_)
#        for d1, name, phi, pnt, theta in iterator:
#            d1.plot_line(name, phi=phi, pnt=pnt, theta=theta)
#            plt.close()
#
#        iterator = product(d_, az_names, phi_, pnt_, theta_)
#        for d2, name, az, pnt, theta in iterator:
#            d2.plot_line(name, az_idx=az, pnt=pnt, theta=theta)
#            plt.close()
#
#    def test_plot_slice(self):
#        self.data.calculate_strain(self.q0)
#        self.data.material_parameters(E=200 * 10 ** 9, v=0.3)
#        data2 = copy.deepcopy(self.data)
#        data2.d1 += 1.0001
#        merged = ordered_merge([self.data, data2], [0, 1])
#
#        # Try a few slice extract options
#
#        phi_names = ['strain', 'stress', 'shear strain', 'shear stress']
#        az_names = ['peaks', 'fwhm', 'strain', 'stress', 'peak err',
#                    'fwhm err', 'strain err']
#
#        phi_ = [-2.5*np.pi]
#        az_ = [20]
#        d_ = [self.data, merged]
#
#        for d1, name, phi in product(d_, phi_names, phi_):
#            d1.plot_slice(name, phi=phi)
#            plt.close()
#        for d2, name, az in product(d_, az_names, az_):
#            d2.plot_slice(name, az_idx=az)
#            plt.close()
#
#    def test_save_reload(self):
#        self.data.calculate_strain(self.q0)
#
#        data2 = copy.deepcopy(self.data)
#        data2.d1 += 1.00001
#        merged = ordered_merge([self.data, data2], [0, 1], 0.1)
#        merged.plot_slice('shear strain', phi=np.pi / 3)
#        plt.close()
#        merged.save_to_hdf5(fpath='pyxe/data/mono_test_pyxe.h5', overwrite=True)
#        merged_reload = PeakAnalysis(fpath='pyxe/data/mono_test_pyxe.h5')
#        merged_reload_b = DataViz(fpath='pyxe/data/mono_test_pyxe.h5')
#        merged_reload.plot_slice('shear strain', phi=np.pi/3)
#        plt.close()
#        merged_reload_b.plot_slice('shear strain', phi=np.pi / 3)
#        plt.close()
#
#        assert np.array_equal(merged.peaks, merged_reload.peaks), \
#            (merged.peaks.shape, merged_reload.peaks.shape)
#
#    def test_save_to_text(self):
#        self.data.calculate_strain(self.q0)
#        self.data.material_parameters(200*10**9, 0.3)
#        data2 = copy.deepcopy(self.data)
#        data2.d1 += 1.00001
#        merged = ordered_merge([self.data, data2], [0, 1], 0.1)
#        merged.plot_slice('shear strain', phi=np.pi / 3)
#        az_lst = ['fwhm', 'fwhm error', 'peaks', 'peaks error',
#                  'strain', 'strain error', 'stress']
#        phi_lst = ['strain', 'shear strain', 'stress', 'shear stress']
#        merged.save_to_txt('pyxe/data/test.csv', az_lst, az_idx=2)
#        merged.save_to_txt('pyxe/data/test.csv', phi_lst, phi=-np.pi/3)
#

#
#if __name__ == '__main__':
#    pass
    # data, q0 = integration()
    # data.calculate_strain(q0)
    # d2 = copy.deepcopy(data)
    # d2.d1 += 1.0001
    # merged = ordered_merge([data, d2], [0, 1])
