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
from pyxe.monochromatic import Mono
from pyxe.fitting_functions import strain_transformation
from pyxe.test.data_create import create_ring_array
from pyxe.merge import ordered_merge

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
mono.add_peaks('Fe')

# Create the test data from that setup
data = create_ring_array(mono, pnts=(7, 7), max_strain=1e-3, crop=crop)
fnames, co_ords, images, (e_xx, e_yy, e_xy) = data
q0 = create_ring_array(mono, crop=crop, pnts=(1, 1), max_strain=0)
q0_fnames, q0_co_ords, q0_images, _ = q0


@patch("fabio.open")
@patch("pyxe.monochromatic.extract_fnames")
def test_integration(extract_fnames, fabio_open):
    fabio_open.side_effect = images
    extract_fnames.return_value = fnames
    data_ = Mono('', co_ords, fit2Dparams, e_to_w(energy), f_ext='.tif',
                 progress=False, npt_rad=crop_shape[0] / 2)

    fabio_open.side_effect = q0_images
    extract_fnames.return_value = q0_fnames
    q0_ = Mono('', q0_co_ords, fit2Dparams, e_to_w(energy), f_ext='.tif',
               progress=False, npt_rad=crop_shape[0] / 2)

    return data_, q0_


class TestMono(object):

    data, q0 = test_integration()

    def test_peak_fit(self):
        self.data.peak_fit(3.1, 1.)

    def test_basic_merge(self):
        self.data.peak_fit(3.1, 1.)
        merged = self.data + self.data
        assert np.array_equal(merged.phi, self.data.phi)
        assert merged.peaks.size == 2 * self.data.peaks.size

    def test_strain_calc(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)

    def test_basic_plot(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)
        self.data.plot_intensity()
        self.data.plot_strain_fit()

    def test_stress_calc(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)
        self.data.define_material(E=200*10**9, v=0.3)

    def test_extract_slice(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)
        self.data.define_material(E=200 * 10 ** 9, v=0.3)
        # Try a few slice extract options
        self.data.extract_slice('strain', phi=np.pi/3)
        assert_raises(AssertionError, self.data.extract_slice, 'strain err', 0)
        self.data.extract_slice('shear stress', phi=5*np.pi)
        self.data.extract_slice('peaks err', az_idx=3)

    def test_positions(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)
        # Compare positions, same angle
        for idx in [0, 7, 12, 26, 32]:
            tensor = e_xx, e_yy, e_xy
            initial = strain_transformation(self.data.phi[idx], *tensor)
            processed_1 = self.data.strain[..., idx]
            processed_2 = self.data.extract_slice(phi=self.data.phi[idx])
            for processed in [processed_1, processed_2]:
                max_diff = np.abs(np.max(initial - processed))
                assert max_diff < 10**-4, (idx, max_diff)  # Brittle (linux)?!

    def test_angles(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)
        # Compare angles, same position
        for p_idx in [0, 7, 12, 26, 39, 45]:
            tensor = e_xx[p_idx], e_yy[p_idx], e_xy[p_idx]
            initial = strain_transformation(self.data.phi, *tensor)
            processed = self.data.strain[p_idx]
            max_diff = np.abs(np.max(initial - processed))
            assert max_diff < 10**-4, (p_idx, max_diff)  # Brittle (linux)?!

    def test_plotting(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)
        self.data.define_material(E=200 * 10 ** 9, v=0.3)
        # Try a few slice extract options
        self.data.plot_slice('strain', phi=np.pi/3)
        self.data.plot_slice('shear stress', phi=5*np.pi)
        self.data.plot_slice('peaks err', az_idx=3)

    def test_ordered_merge(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)

        data2 = copy.deepcopy(self.data)

        pad = 0.35
        shift = 1.00001
        data2.d1 += shift
        added = np.sum(data2.d1 > (1 + pad))

        merged = ordered_merge([self.data, data2], [0, 1], pad)
        assert np.array_equal(merged.phi, self.data.phi)
        assert merged.d1.size == added + self.data.d1.size, (
        merged.d1.size, added, self.data.d1.size)

    def test_merged_plot_slice(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)

        data2 = copy.deepcopy(self.data)
        shift = 1.00001
        data2.d1 += shift
        merged = ordered_merge([self.data, data2], [0, 1])
        merged.plot_slice('strain', phi=np.pi / 3)
        merged.plot_slice('shear stress', phi=5 * np.pi)
        merged.plot_slice('peaks err', az_idx=3)


    def test_plot_line(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)
        self.data.define_material(E=200 * 10 ** 9, v=0.3)
        # Try a few slice extract options
        self.data.plot_line('strain', phi=np.pi / 3)
        self.data.plot_slice('shear stress', phi=5 * np.pi, pnt=(0, 0),
                             theta=np.pi / 3)
        self.data.plot_slice('peaks err', az_idx=3, pnt=(0.2, 0.1),
                             theta=-np.pi / 3)


    def test_merged_plot_line(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)

        data2 = copy.deepcopy(self.data)
        shift = 1.00001
        data2.d1 += shift
        merged = ordered_merge([self.data, data2], [0, 1])
        merged.plot_slice('strain', phi=np.pi / 3)
        merged.plot_slice('shear stress', phi=5 * np.pi, pnt=(0, 0),
                          theta=np.pi / 3)
        merged.plot_slice('peaks err', az_idx=3, pnt=(0.2, 0.1), theta=-np.pi / 3)

if __name__ == '__main__':
    data, q0 = test_integration()
    data.peak_fit(3.1, 1.)
    q0.peak_fit(3.1, 1.)
    data.calculate_strain(q0)

    data2 = copy.deepcopy(data)

    if data2.d1[0].size % 2 == 0:
        data2.d1[0] += 1
        added = data2.d1[0].size / 2
    else:
        data2.d1 += 1 + 1 / (data2.d1[0].size - 1)
        added = data2.d1[0].size // 2

    merged = ordered_merge([data, data2], [0, 1])
    #
    # # Compare positions, same angle
    # for a_idx in [0, 7, 12, 26, 32]:
    #     i = strain_transformation(data.phi[a_idx], *(e_xx, e_yy, e_xy))
    #     p_1 = data.strain[..., a_idx]
    #     p_2 = data.extract_slice(phi=data.phi[a_idx])
    #     for p in [p_1, p_2]:
    #         abs_max = np.max(np.abs(i - p))
    #         print(abs_max)
    #         assert abs_max < 10 ** -4, (a_idx, abs_max)
