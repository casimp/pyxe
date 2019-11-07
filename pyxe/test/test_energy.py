# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:25:14 2016

@author: casim
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import product
import matplotlib.pyplot as plt
import copy
from nose.tools import assert_raises
import numpy as np
from mock import patch
from pyxpb.detectors import i12_energy
from pyxpb.conversions import e_to_w
from pyxe.energy_dispersive import EDI12
from pyxe.fitting_functions import strain_transformation
from pyxpb.array_create import intensity_array
from pyxe.merge import ordered_merge
from pyxe.peak_analysis import PeakAnalysis
from pyxe.plotting import DataViz


def i12_dict(X, Y, q, I):
    d = {'entry1/scan_command': np.array([b'ss2_x ss2_y']),
         'entry1/EDXD_elements/edxd_q': q,
         'entry1/EDXD_elements/data': I,
         'entry1/EDXD_elements/ss2_x': X,
         'entry1/EDXD_elements/ss2_y': Y}
    return d

# Lets create a detector and add peaks...
i12 = i12_energy()
i12.add_material('Fe')

# Create the test data from that setup
data_ = intensity_array(i12, pnts=(6, 6), max_strain=1e-3)
x, y, q, I,(e_xx, e_yy, e_xy) = data_
data_dict=i12_dict(x, y, q, I)

q0_ = intensity_array(i12, pnts=(1, 1), max_strain=0)
x, y, q, I, _ = q0_

q0_data_dict=i12_dict(x, y, q, I)

@patch("h5py.File")
def integration(h5py_file):
    h5py_file.return_value = data_dict
    data_ = EDI12('', [])
    h5py_file.return_value = q0_data_dict
    q0_ = EDI12('', [])
    return data_, q0_


def peak_fit():
    data, q0 = integration()

    data.I += 1e-6
    data.peak_fit(3.1, 1.)
    q0.I += 1e-6

    # Test different fitting functions
    q0.peak_fit(3.1, 1., func='lorentzian')
    q0.peak_fit(3.1, 1., func='psuedo_voigt')
    q0.peak_fit(3.1, 1., func='gaussian')

    return data, q0


def test_no_analysis_save_load():
    data, q0 = integration()
    data.save_to_hdf5(fpath='pyxe/data/energy_test_pyxe.h5', overwrite=True)
    PeakAnalysis(fpath='pyxe/data/energy_test_pyxe.h5')


class TestEnergy(object):

    data, q0 = peak_fit()

    def test_basic_merge(self):
        merged = self.data + self.data
        assert np.array_equal(merged.phi, self.data.phi)
        assert merged.peaks.size == 2 * self.data.peaks.size

    def test_strain_calc(self):
        self.data.calculate_strain(self.q0)

    def test_basic_plot(self):
        self.data.calculate_strain(self.q0)
        self.data.plot_intensity()
        plt.close()
        self.data.plot_strain_fit()
        plt.close()

    def test_stress_calc(self):
        self.data.calculate_strain(self.q0)
        self.data.material_parameters(E=200*10**9, v=0.3)

    def test_extract_slice(self):
        self.data.calculate_strain(self.q0)
        self.data.material_parameters(E=200 * 10 ** 9, v=0.3)
        # Try a few slice extract options
        self.data.extract_slice('strain', phi=np.pi/3)
        assert_raises(AssertionError, self.data.extract_slice, 'strain err', 0)
        self.data.extract_slice('shear stress', phi=5*np.pi)
        self.data.extract_slice('peaks err', az_idx=3)

    def test_positions(self):
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
        self.data.calculate_strain(self.q0)
        # Compare angles, same position
        for p_idx in [(0, 0), (4, 4), (3, 5), (1, 2), (0, 4)]:
            tensor = e_xx[p_idx], e_yy[p_idx], e_xy[p_idx]
            initial = strain_transformation(self.data.phi, *tensor)
            processed = self.data.strain[p_idx]
            max_diff = np.abs(np.max(initial - processed))
            assert max_diff < 10**-4, (p_idx, max_diff)

    def test_ordered_merge(self):
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

    def test_plot_line(self):
        self.data.calculate_strain(self.q0)
        self.data.material_parameters(E=200 * 10 ** 9, v=0.3)
        data2 = copy.deepcopy(self.data)
        data2.d1 += 1.0001
        merged = ordered_merge([self.data, data2], [0, 1])

        # Try a few slice extract options
        phi_names = ['strain', 'stress', 'shear strain', 'shear stress']
        az_names = ['peaks', 'fwhm', 'strain', 'stress', 'peak err',
                    'fwhm err', 'strain err']
        phi_, az_ = [-2 * np.pi], [20]
        pnt_, theta_ = [(-0.2, 0.2)], [0, -np.pi / 3]
        d_ = [self.data, merged]

        iterator = product(d_, phi_names, phi_, pnt_, theta_)
        for d1, name, phi, pnt, theta in iterator:
            d1.plot_line(name, phi=phi, pnt=pnt, theta=theta)
            plt.close()

        iterator = product(d_, az_names, phi_, pnt_, theta_)
        for d2, name, az, pnt, theta in iterator:
            d2.plot_line(name, az_idx=az, pnt=pnt, theta=theta)
            plt.close()

    def test_plot_slice(self):
        self.data.calculate_strain(self.q0)
        self.data.material_parameters(E=200 * 10 ** 9, v=0.3)
        data2 = copy.deepcopy(self.data)
        data2.d1 += 1.0001
        merged = ordered_merge([self.data, data2], [0, 1])

        # Try a few slice extract options

        phi_names = ['strain', 'stress', 'shear strain', 'shear stress']
        az_names = ['peaks', 'fwhm', 'strain', 'stress', 'peak err',
                    'fwhm err', 'strain err']

        phi_ = [-2.5*np.pi]
        az_ = [20]
        d_ = [self.data, merged]

        for d1, name, phi in product(d_, phi_names, phi_):
            d1.plot_slice(name, phi=phi)
            plt.close()
        for d2, name, az in product(d_, az_names, az_):
            d2.plot_slice(name, az_idx=az)
            plt.close()
            # Except for contour levels must be increasing (silly array!)
            # except ValueError:
            #     pass

    def test_save_reload(self):
        self.data.calculate_strain(self.q0)

        data2 = copy.deepcopy(self.data)
        data2.d1 += 1.00001
        merged = ordered_merge([self.data, data2], [0, 1], 0.1)
        merged.plot_slice('shear strain', phi=np.pi / 3)
        plt.close()
        merged.save_to_hdf5(fpath='pyxe/data/energy_test_pyxe.h5', overwrite=True)
        merged_reload = PeakAnalysis(fpath='pyxe/data/energy_test_pyxe.h5')
        merged_reload_b = DataViz(fpath='pyxe/data/energy_test_pyxe.h5')
        merged_reload.plot_slice('shear strain', phi=np.pi/3)
        plt.close()
        merged_reload_b.plot_slice('shear strain', phi=np.pi / 3)
        plt.close()

        err = '{}, {}, ({}, {})'.format(merged.peaks.shape, merged_reload.peaks.shape,
                                        merged.peaks, merged_reload.peaks)
        assert np.array_equal(merged.peaks, merged_reload.peaks)

    def test_save_to_text(self):
        self.data.calculate_strain(self.q0)
        self.data.material_parameters(200*10**9, 0.3)
        data2 = copy.deepcopy(self.data)
        data2.d1 += 1.00001
        merged = ordered_merge([self.data, data2], [0, 1], 0.1)
        merged.plot_slice('shear strain', phi=np.pi / 3)
        az_lst = ['fwhm', 'fwhm error', 'peaks', 'peaks error',
                  'strain', 'strain error', 'stress']
        phi_lst = ['strain', 'shear strain', 'stress', 'shear stress']
        merged.save_to_txt('pyxe/data/test.csv', az_lst, az_idx=2)
        merged.save_to_txt('pyxe/data/test.csv', phi_lst, phi=-np.pi/3)

    def test_crop_rotate_centre(self):
        data2 = copy.deepcopy(self.data)
        data2.centre((0.25, -0.25))
        data2.flipaxis(0)
        data2.flipaxis(1)
        data2.swapaxes(0, 1)

        data2.crop1d(0, None, 2)
        data2.plot_slice('strain', phi=0)
        plt.close()


    def test_pawley_plot(self):

        self.data.add_material('Fe')
        self.data.plot_intensity(pawley=True)
        plt.close()

    def test_pawley_fit(self):
        data2 = copy.deepcopy(self.data)
        # Slow so reduce number of points
        data2.crop1d(0, None, 3)
        data2.add_material('Fe')
        data2.pawley_fit()
        data2.calculate_strain(a0=data2.peaks[0,0])

    def test_fwhm_est(self):
        self.data.add_material('Fe')
        self.data.estimate_fwhm(pnt=(0,0),q0s=[3.01, 4.4, 5.37, 6.2, 6.9], k=0)
        self.data.estimate_fwhm(pnt=(0,0),q0s=[3.01, 4.4, 5.37, 6.2, 6.9], k=2)
        self.data.estimate_fwhm(pnt=(0,0),q0s=[3.01, 4.4, 5.37, 6.2, 6.9], k=1)

    def test_define_background(self):
        self.data.add_material('Fe')
        self.data.define_background(seg=13, k=1)
        plt.close()

    def test_define_temperature(self):
        # 2d temp distrib
        self.data.define_temperature(T=[10, 20, 30, 40], d1=[-1, -1, 1, 1],
                                     d2=[-1, 1, -1, 1], plot=True)
        plt.close()

        # 1d temp distrib in d1
        self.data.define_temperature(T=[10, 20, 30, 40], d1=[-1, -0.8, 0, 1],
                             bounds_error=False,
                             fill_value='extrapolate', plot=True)
        plt.close()
        # 1d temp distrib in d2
        self.data.define_temperature(T=[10, 20, 30, 40], d2=[-1, -0.8, 0, 1],
                                     bounds_error=False,
                                     fill_value='extrapolate', plot=True)
        plt.close()

#if __name__ == '__main__':
#
#    h = TestEnergy()
#    h.data.calculate_strain(h.q0)
#    h.data.plot_slice('strain', phi=0)
#    data, q0 = integration()
#    data.peak_fit(3.1, 1.)
#    q0.peak_fit(3.1, 1.)
#    data.calculate_strain(q0)
#    data.material_parameters(200*10**9, 0.3, G=None)
#    data.plot_intensity()
#    data.plot_strain_fit()
#    shift = 1.00001
#    data2 = copy.deepcopy(data)
#    data2.d1 += shift
#    merged = ordered_merge([data, data2], [0, 1], 0)
#    merged.plot_slice('shear strain', phi=0)
#    #plt.show()
#    merged.save_to_hdf5(overwrite=True)
#    merged_reload = PeakAnalysis('_pyxe.h5')
#    #print(merged_reload.analysis_state.decode())
#    merged_reload.plot_slice('shear strain', phi=0)
#    plt.show()
