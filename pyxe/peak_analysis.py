from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py
import numpy as np
import os

from pyxe.command_parsing import analysis_check
from pyxe.fitting_tools import array_fit
from pyxe.analysis_tools import full_ring_fit, pyxe_to_nxs
from pyxe.plotting import DataViz
from pyxe.merge import basic_merge


class PeakAnalysis(DataViz):
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
        # Reset strain to None after peak fitting...
        self.strain, self.strain_err, self.strain_tensor = None, None, None
        self.analysis_state = 'peaks'

    @analysis_check('peaks')
    def calculate_strain(self, q0, tensor_fit=True):
        """
        Ideally pass in a pyxe data object containing
        """
        if isinstance(q0, PeakAnalysis):
            assert np.array_equal(q0.phi, self.phi)
            q0 = q0.peaks.mean(axis=tuple(range(0, q0.n_dims)))
        self.q0 = q0
        self.strain = (self.q0 / self.peaks) - 1
        self.strain_err = (self.q0 / self.peaks_err) - 1
        if tensor_fit:
            self.strain_tensor = full_ring_fit(self.strain, self.phi)
            self.analysis_state = 'strain fit'
        else:
            self.analysis_state = 'strain'

    @analysis_check('strain')
    def define_material(self, E, v, G=None, stress_state='plane_strain'):
        G = E / (2 * (1-v)) if G is None else G
        self.E, self.v, self.G, self.stress_state = E, v, G, stress_state
        self.analysis_state = self.analysis_state.replace('strain', 'stress')

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

    def __add__(self, other):
        return basic_merge([self, other])

Reload = PeakAnalysis
