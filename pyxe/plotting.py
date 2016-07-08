# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casimp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata

from pyxe.fitting_functions import strain_transformation, shear_transformation
from pyxe.plotting_tools import plot_complex, meshgrid_res, plot_line
from pyxe.command_parsing import complex_check, text_cleaning


def plot_intensity(q, I, az_idx=0, pnt=(), figsize=(7, 5), ax=False):
    """
    Plots q v intensity.

    # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty.
    # point:      Define point (index) from which to extract q v I plot.
                  First point in array chosen if not (default) specified.
    """
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    if pnt == ():
        pnt = (0,) * len(I[..., 0, 0].shape)

    ax.plot(q[az_idx], I[pnt][az_idx], 'k-')
    ax.set_xlabel('q (rad)')
    ax.set_ylabel('Intensity')
    return ax


def plot_fitted(phi, strain, strain_tensor, pnt=(), q_idx=0, figsize=(7, 5), ax=False):
    """
    Plots fitted in-plane strain field for given data point.

    # point:      Define point (index) from which to plot fitted in-plane
                  strain field.
    # q_idx:      0 based indexing - 0 (default) to 23 - detector 23 empty.
    # figsize:    Figure dimensions
    """
    pnt = (0,) * (strain.ndim - 1) if pnt == () else pnt

    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

    p = strain_tensor[pnt]
    # Data from edi12 has extra, unused detector (filled with nan)
    ax.plot(phi, strain[pnt], 'k*')
    phi_2 = np.linspace(phi[0], phi[-1], 1000)
    ax.plot(phi_2, strain_transformation(phi_2, *p), 'k-')
    ax.set_xlabel(r'$\phi$ (rad)', size=14)
    ax.set_ylabel(r'$\epsilon$', size=14)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))


class DataViz(object):
    """

    """

    def __init__(self, pyxe_object):
        self.peaks = pyxe_object.peaks
        self.peaks_err = pyxe_object.peaks_err
        self.fwhm = pyxe_object.fwhm
        self.fwhm_err = pyxe_object.fwhm_err
        self.analysis_state = pyxe_object.analysis_state

    def extract_line(self, data='strain', phi=None, az_idx=None,
                     pnt=None, theta=0, z_idx=0, npnts=200):
        data = self.extract_slice(data, phi, az_idx, z_idx)

    @complex_check
    def extract_slice(self, data='strain', phi=None, az_idx=None, z_idx=None):
        command = text_cleaning(data)
        az_command = 'phi' if phi is not None else 'az_idx'

        if az_command == 'az_idx':

            data_command = {'peaks': self.peaks,
                            'peaks error': self.peaks_err,
                            'fwhm': self.peaks,
                            'fwhm error': self.peaks_err,
                            'strain': self.peaks,
                            'strain error': self.peaks_err}

            data = data_command[command][..., az_idx]
        else:
            tensor = self.strain_tensor
            shear = True if 'shear' in command else False
            stress = True if 'stress' in command else False

            if shear:
                e_xy = shear_transformation(tensor, phi)
                data = self.G * e_xy if stress else e_xy

            elif stress:
                e_xx = strain_transformation(tensor, phi)
                e_yy = strain_transformation(tensor, phi + np.pi / 2)
                data = self.stress_eqn(e_xx, e_yy)

            else:
                data = strain_transformation(tensor, phi)

        return data[z_idx] if z_idx is not None else data

    def plot_line(self, data='strain', phi=None, az_idx=None, z_idx=0,
                  pnt=(0, 0), theta=0, npnts=100):

        dims, data = self.extract_line(data)

    def plot_slice(self, data='strain', phi=None, az_idx=None, z_idx=0,
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
