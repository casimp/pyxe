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
from pyxpb.detectors import i12_energy
from pyxpb.conversions import e_to_w
from pyxe.energy_dispersive import EDI12
from pyxe.fitting_functions import strain_transformation
from pyxe.test.data_create import create_intensity_array
from pyxe.merge import ordered_merge

# Lets create a detector and add peaks...
i12_energy.add_peaks('Fe')

# Create the test data from that setup
data_ = create_intensity_array(i12_energy, pnts=(7, 7), max_strain=1e-3)
data_dict,(e_xx, e_yy, e_xy) = data_

q0_ = create_intensity_array(i12_energy, pnts=(1, 1), max_strain=0)
q0_data_dict, _ = q0_


@patch("h5py.File")
def test_integration(h5py_file):
    h5py_file.return_value = data_dict
    data_ = EDI12('', [])
    h5py_file.return_value = q0_data_dict
    q0_ = EDI12('', [])
    return data_, q0_


class TestEnergy(object):

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
        for idx in [0, 7, 12, 17, 22]:
            tensor = e_xx, e_yy, e_xy
            initial = strain_transformation(self.data.phi[idx], *tensor)
            processed_1 = self.data.strain[..., idx]
            processed_2 = self.data.extract_slice(phi=self.data.phi[idx])
            for processed in [processed_1, processed_2]:
                max_diff = np.abs(np.max(initial - processed))
                assert max_diff < 10**-4, (idx, max_diff)

    def test_angles(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)
        # Compare angles, same position
        for p_idx in [(0, 0), (4, 4), (3, 5), (1, 2), (0, 4)]:
            tensor = e_xx[p_idx], e_yy[p_idx], e_xy[p_idx]
            initial = strain_transformation(self.data.phi, *tensor)
            processed = self.data.strain[p_idx]
            max_diff = np.abs(np.max(initial - processed))
            assert max_diff < 10**-4, (p_idx, max_diff)

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
        err = (merged.d1.size, added, self.data.d1.size)
        assert merged.d1.size == added + self.data.d1.size, err

    def test_plotting(self):
        self.data.peak_fit(3.1, 1.)
        self.q0.peak_fit(3.1, 1.)
        self.data.calculate_strain(self.q0)
        self.data.define_material(E=200 * 10 ** 9, v=0.3)
        # Try a few slice extract options
        self.data.plot_slice('strain', phi=np.pi/3)
        self.data.plot_slice('shear stress', phi=5*np.pi)
        self.data.plot_slice('peaks err', az_idx=3)


if __name__ == '__main__':
    data, q0 = test_integration()
    data.peak_fit(3.1, 1.)
    q0.peak_fit(3.1, 1.)
    data.calculate_strain(q0)
