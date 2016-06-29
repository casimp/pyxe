from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import shutil

import h5py
import numpy as np
from scipy.optimize import curve_fit

from pyxe.fitting_tools import array_fit
from pyxe.fitting_functions import strain_transformation
#from pyxe.strain_tools import StrainTools
#from pyxe.plotting import StrainPlotting
from pyxe.analysis_tools import dimension_fill, full_ring_fit


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
        array_shape = self.I.shape[:-1] + (1,)
        data = [np.nan * np.ones(array_shape) in range(4)]
        self.peaks, self.peaks_err, self.fwhm, self.fwhm_err = data

        print('\nFile: %s - %s acquisition points\n' %
              (self.name, self.I[..., 0, 0].size))

        fit = array_fit(self.q, self.I, peak_window, func, err_lim, progress)
        self.peaks, self.peaks_err, self.fwhm, self.fwhm_err = fit

        # Import peak plotting and peak extraction
        #self._extract_peaks = None
        #self.plot_peaks = PeakPlotting(self)
        self.analysis_state = 1

    # Must calculate peaks first
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

        # Import strain plotting and strain extraction
        #self._extract_strain = None
        #self.plot_strain = StrainPlotting(self)
        self.analysis_state = 2

    # Must calculate strain first
    def define_material(self, E, v, G, stress_state='plane_strain'):
        #self._extract_stress = None
        #self.plot_stress = StressPlotting(self)
        self.analysis_state = 3


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
