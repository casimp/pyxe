from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import shutil

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

from pyxe.fitting_tools import array_fit
from pyxe.fitting_functions import strain_transformation
from pyxe.analysis_tools import full_ring_fit, pyxe_to_nxs



def analysis_state_check(required_state):
    def dec_check(func):
        def wrapper(*args, **kwargs):
            state_dict = {1:'peak_fit', 2:'calculate_strain',
                          3: 'define_material_properties'}
            s_ids = range(args[0].analysis_state + 1, required_state + 1)
            c = '\n'.join([state_dict[i] for i in s_ids])
            error = '\nPlease run the following commands first:\n{}'.format(c)
            try:
                assert args[0].analysis_state >= required_state
                return func(*args, **kwargs)
            except AssertionError:
                print(error)

        return wrapper
    return dec_check


def plot_fitted(phi, strain, strain_tensor, pnt=(), figsize=(7, 5), ax=False):
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
        ax = fig.add_subplot(111)

    p = strain_tensor[pnt]
    ax.plot(phi, strain[pnt], 'k*')
    phi_2 = np.linspace(phi[0], phi[-1], 1000)
    ax.plot(phi_2, strain_transformation(phi_2, *p), 'k-')
    ax.set_xlabel(r'$\phi$ (rad)', size=14)
    ax.set_ylabel(r'$\epsilon$', size=14)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))


class PeakAnalysis(object):
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

        self.analysis_state = 0

    def peak_fit(self, q0_approx, window_width, func='gaussian',
                 err_lim=10**-4, progress=True):

        peak_window = [q0_approx - window_width/2, q0_approx + window_width/2]
        self.q0_approx = q0_approx

        # Iterate across q0 values and fit peaks for all detectors
        array_shape = self.I.shape[:-1]
        data = [np.nan * np.ones(array_shape) for _ in range(4)]
        self.peaks, self.peaks_err, self.fwhm, self.fwhm_err = data

        print('\n%s acquisition points\n' % self.I[..., 0, 0].size)

        fit = array_fit(self.q, self.I, peak_window, func, err_lim, progress)
        self.peaks, self.peaks_err, self.fwhm, self.fwhm_err = fit

        self.analysis_state = 1

    @analysis_state_check(1)
    def calculate_strain(self, q0, tensor_fit=True):
        """
        Ideally pass in a pyxe data object containing
        """
        if isinstance(q0, PeakAnalysis):
            assert q0.phi == self.phi
            q0 = q0.peaks.mean(axis=tuple(range(0, q0.n_dims)))
        self.q0 = q0
        self.strain = (self.q0 / self.peaks) - 1
        self.strain_err = (self.q0 / self.peaks_err) - 1
        if tensor_fit:
            self.strain_tensor = full_ring_fit(self.peaks, self.phi)

        self.analysis_state = 2

    @analysis_state_check(2)
    def define_material(self, E, v, G, stress_state='plane_strain'):
        self.analysis_state = 3

    # @analysis_state_check(1)
    # def plot_fitted(self, pnt=(), figsize=(7, 5), ax=False):
    #     plot_fitted(self.phi, self.strain, self.strain_tensor, pnt=pnt,
    #                 figsize=figsize, ax=ax)


    def save_to_nxs(self, fpath=None, overwrite=False):
        """
        Saves all data back into an expanded .nxs file. Contains all original
        data plus q0, peak locations and strain.

        # fpath:      Abs. path for new file - default is to save to parent
                      directory (*/folder/folder_pyxe.nxs)
        # overwrite:  Overwrite file if it already exists (True/[False])
        """
        if fpath is None:
            fpath = '%s_pyxe.nxs' % os.path.splitext(self.fpath)[0]

        pyxe_to_nxs(fpath, self, overwrite)
