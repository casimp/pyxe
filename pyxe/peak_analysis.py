from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval

from pyxe.command_parsing import analysis_check
from pyxe.fitting_tools import array_fit, array_fit_pawley
from pyxe.analysis_tools import full_ring_fit, pyxe_to_hdf5, data_extract
from pyxe.plotting import DataViz
from pyxe.merge import basic_merge
from pyxe.fitting_functions import plane_strain, plane_stress


class PeakAnalysis(DataViz):
    def __init__(self, fpath):
        """ Analysis class for the calculation of strain from q/a values.

        The analysis is based around two peak fitting methods - single peak
        and Pawley refinement. The Pawley refinement requires a more complete
        description of the constituent material/phases and returns a lattice
        parameter, whereas the single peak analysis returns a particular
        lattice spacing, d0, or rather the reciprocal equivalent, q0.

        The strain analysis follows naturally from this via a simple strain
        calculation (e = delta_a / a0), with the variation in strain wrt.
        azimuthal position allowing for the computation of the full strain
        tensor (e_xx, e_yy, e_xy).

        It is then possible to finalise the analytical procedure and calculate
        stress, which relies on the material system being in a plane strain
        or plane stress state. The in-plane stress state can then be
        determined.

        Args:
            fpath (str): Path to an analyzed (integrated) pyxe hdf5 file
        """
        self.fpath = fpath

        with h5py.File(fpath, 'r') as f:
            self.ndim, self.d1, self.d2, self.d3 = data_extract(f, 'dims')
            self.q, self.I, self.phi = data_extract(f, 'raw')
            self.peaks, self.peaks_err = data_extract(f, 'peaks')
            self.fwhm, self.fwhm_err = data_extract(f, 'fwhm')
            self.strain, self.strain_err = data_extract(f, 'strain')
            self.strain_tensor = data_extract(f, 'tensor')[0]
            self.E, self.v, self.G = data_extract(f, 'material')
            self.stress_state, self.analysis_state = data_extract(f, 'state')
            if self.stress_state is None:
                self.stress_eqn = None
            else:
                p_strain = self.stress_state == 'plane strain'
                self.stress_eqn = plane_strain if p_strain else plane_stress

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
        """ Calculate strain based on q0/a0 value(s).

        Strain can be calculated with respect to a single value of
        q0 or a0 but it is recommended to pass in a analyzed pyxe
        data object containing q0/a0 measurements. Strain can then be
        assessed wrt. azimuthal position, which reduces error in the case
        of an area detector (see Kolunsky et al. ??)

        There is an option to then compute the full strain tensor. This fits
        the stress/strain transformation equations to the data and stores
        e_xx, e_yy, e_xy.

        Args:
            q0 (float, object): q0/a0 float or pyxe data object
            tensor_fit (bool): Calculate the full strain tensor
        """
        if isinstance(q0, PeakAnalysis):
            assert np.array_equal(q0.phi, self.phi)
            q0 = q0.peaks.mean(axis=tuple(range(0, q0.ndim)))
        self.q0 = q0
        self.strain = (self.q0 / self.peaks) - 1
        self.strain_err = (self.q0 / self.peaks_err) - 1
        if tensor_fit:
            self.strain_tensor = full_ring_fit(self.strain, self.phi)
            self.analysis_state = 'strain fit'
        else:
            self.analysis_state = 'strain'

    @analysis_check('strain')
    def material_parameters(self, E, v, G=None, stress_state='plane strain'):
        G = E / (2 * (1-v)) if G is None else G
        self.E, self.v, self.G, self.stress_state = E, v, G, stress_state
        eqn = plane_strain if stress_state == 'plane strain' else plane_stress
        self.stress_eqn = eqn
        self.analysis_state = self.analysis_state.replace('strain', 'stress')

    def save_to_hdf5(self, fpath=None, overwrite=False):
        """ Save data back to hdf5 file format.

        Saves analyzed information and the detector setup. Data is discarded
        relative to the NeXus data format (where applicable).

        Args:
            fpath (str): Abs. path for new file - default is to save to
                         parent directory (*/folder/folder_pyxe.h5)
            overwrite (bool): Overwrite file if it already exists
        """
        if fpath is None:
            if self.fpath[-8:] == '_pyxe.h5':
                fpath = self.fpath
            else:
                fpath = '%s_pyxe.h5' % os.path.splitext(self.fpath)[0]

        pyxe_to_hdf5(fpath, self, overwrite)

    def __add__(self, other):
        return basic_merge([self, other])

Reload = PeakAnalysis
