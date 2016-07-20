# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casimp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata, interp2d

from pyxe.command_parsing import analysis_check
from pyxe.fitting_functions import strain_transformation, shear_transformation
from pyxe.plotting_tools import plot_complex, meshgrid_res, line_extract, az90
from pyxe.command_parsing import complex_check, text_cleaning


class DataViz(object):
    """

    """

    def __init__(self, fpath):
        """
        # fpath:      Data is either the filepath to an analyzed pyxe NeXus
                      file or a pyxe data object (inc. merged object)
        """
        self.fpath = fpath
        with h5py.File(fpath, 'r') as f:
            data = f['entry1/pyxe_analysis']
            self.n_dims = data['n_dims']
            self.d1, self.d2, self.d3 = data['d1'], data['d2'], data['d3']
            self.q, self.I, self.phi = data['q'], data['I'], data['phi']
            self.peaks, self.peaks_err = data['strain'], data['strain_err']
            self.fwhm, self.fwhm_err = data['fwhm'], data['fwhm_err']
            self.strain, self.strain_err = data['strain'], data['strain_err']
            self.strain_tensor = data['strain_tensor']
            self.stress_state = data['stress_state']
            self.E, self.v, self.G = data['E'], data['v'], data['G']
            self.analysis_state = data['analysis_state']

    def plot_intensity(self, pnt=None, az_idx=0, figsize=(7, 5), ax=False):
        """
        Plots q v intensity.

        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty.
        # point:      Define point (index) from which to extract q v I plot.
                      First point in array chosen if not (default) specified.
        """
        if not ax:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)

        pnt = (0,) * len(self.I[..., 0, 0].shape) if pnt is None else pnt

        ax.plot(self.q[az_idx], self.I[pnt][az_idx], 'k-')
        ax.set_xlabel('q (rad)')
        ax.set_ylabel('Intensity')
        return ax

    @analysis_check('strain fit')
    def plot_strain_fit(self, pnt=None, figsize=(7, 5), ax=False):
        """
            Plots fitted in-plane strain field for given data point.

            # point:      Define point (index) from which to plot fitted in-plane
                          strain field.
            # q_idx:      0 based indexing - 0 (default) to 23 - detector 23 empty.
            # figsize:    Figure dimensions
            """
        pnt = (0,) * (self.strain.ndim - 1) if pnt is None else pnt

        if not ax:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)

        p = self.strain_tensor[pnt]
        ax.plot(self.phi, self.strain[pnt], 'k*')
        phi_2 = np.linspace(self.phi[0], self.phi[-1], 1000)
        ax.plot(phi_2, strain_transformation(phi_2, *p), 'k-')
        ax.set_xlabel(r'$\phi$ (rad)', size=14)
        ax.set_ylabel(r'$\epsilon$', size=14)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))
        # Include mohr!

    def extract_line(self, data='strain', phi=None, az_idx=None,
                     pnt=None, theta=0, z_idx=None, res=0.1):
        data = self.extract_slice(data, phi, az_idx, z_idx)
        if self.n_dims == 1:
            return self.d1, data
        else:
            ## or merged??
            x, y, d = line_extract(self.d1, self.d2, pnt, theta, res)
            print(x.shape, y.shape, d.shape)
            co_ords = (self.d1.flatten(), self.d2.flatten())
            line = griddata(co_ords, data.flatten(), (x, y))
            print(line.shape)
            return x, y, d, line

    def extract_slice(self, data='strain', phi=None, az_idx=None, z_idx=None):
        complex_check(data, self.analysis_state, phi, az_idx)
        command = text_cleaning(data)
        az_command = 'phi' if phi is not None else 'az_idx'

        if az_command == 'az_idx':
            if 'stress' not in command:
                data_command = {'peaks': self.peaks,
                                'peaks error': self.peaks_err,
                                'fwhm': self.fwhm,
                                'fwhm error': self.fwhm_err,
                                'strain': self.strain,
                                'strain error': self.strain_err}

                data = data_command[command][..., az_idx]
            else:
                d = self.strain if 'err' not in command else self.strain_err
                e_xx, e_yy = d[..., az_idx],  d[..., az90(self.phi, az_idx)]
                data = self.stress_eqn(e_xx, e_yy, self.E, self.v)

        else:
            tensor = self.strain_tensor
            tensor = tensor[..., 0], tensor[..., 1], tensor[..., 2]
            shear = True if 'shear' in command else False
            stress = True if 'stress' in command else False

            if shear:
                e_xy = shear_transformation(phi, *tensor)
                data = self.G * e_xy if stress else e_xy

            elif stress:
                e_xx = strain_transformation(phi, *tensor)
                e_yy = strain_transformation(phi + np.pi / 2, *tensor)
                data = self.stress_eqn(e_xx, e_yy, self.E, self.v)

            else:
                data = strain_transformation(phi, *tensor)
        data = data[z_idx] if z_idx is not None else data
        print('data', data.shape)
        return data

    def plot_line(self, data='strain', phi=None, az_idx=None, z_idx=None,
                  pnt=(0, 0), theta=0, res=0.1, pos_value='d', ax=False):

        x, y, d, line = self.extract_line(data, phi, az_idx, pnt,
                                          theta, z_idx, res)
        position = {'x': x, 'y': y, 'd': d}

        if not ax:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(1, 1, 1)
        ax.plot(position[pos_value], line, 'k-')
        ax.set_xlabel('Position ({})'.format('mm'))
        ax.set_ylabel(data)
        return ax

    def plot_slice(self, data='strain', phi=None, az_idx=None, z_idx=None,
                     plot_func=None, **kwargs):
        data = self.extract_slice(data, phi, az_idx, z_idx)
        plot_func = plot_complex if plot_func is None else plot_func
        if data.ndim == 1:
            d1_, d2_ = meshgrid_res(self.d1, self.d2, spatial_resolution=0.1)
            z = griddata((self.d1.flatten(), self.d2.flatten()),
                         data.flatten(), (d1_, d2_))
        else:
            d1_, d2_, z = self.d1, self.d2, data

        ax_ = plot_func(self.d1, self.d2, d1_, d2_, z, **kwargs)

        return ax_
