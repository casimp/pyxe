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

    def add_material(self, material, weight=1, background=True):
        """ Add material or phase to setup (needed for Pawley fit).

        Finds Bragg peaks and estimates relative intensities based on material
        and scanning setup.

        Args:
            material (str): Element symbol (compound formula)
            weight (float): Relative peak weight (useful for mixtures/phases)
            background (bool): Fit background profile based on material/peaks
        """
        self.detector.add_material(material, weight=weight)
        if background:
            self.define_background()

    def define_background(self, seg=50, k=12, pnt=None, fwhm=None, plot=True,
                          az_idx=0, auto=True, x=None, y=None):
        """ Background profile fitting - segment data and auto find points.

        Fits a Chebyshev polynomial (of order k) to automatically acquired
        q, I points. These points are selected wrt. the materials present
        (defined via add_materials), with the Bragg peaks being avoided.

        If auto is False, q v I points can be manually specified.

        Args:
            seg (int): Number of points to split data into/extract
            k (int): Order for Cheyshev polynomial
            pnt (tuple): Point co_ords or None for averaged data
            fwhm (float): Peak fwhm (to exclude more data around peaks)
            plot (bool): True/False
            az_idx (int): Azimuthal slice to plot
            auto (bool): Automatically extract q v I data
            x (ndarray): If not auto maunally defined q values
            y (ndarray): If not auto maunally defined I values
        """
        # Automatically averages over all points unless point is specified
        if pnt is None:
            I = np.mean(self.I, tuple(range(self.I.ndim - 2)))
        else:
            I = self.I[pnt]

        if auto:
            az = self.q.shape[0]
            x, y = np.zeros((az, seg)), np.zeros((az, seg))

            for a in range(az):
                split_q = np.array_split(self.q[a], seg)
                split_I = np.array_split(I[a], seg)

                for idx, q in enumerate(split_q):
                    q_min, q_max = np.min(q), np.max(q)
                    for mat in self.detector.materials:
                        q0 = self.detector.q0[mat]
                        sig = self.detector.sigma[mat]
                        sig = sig if fwhm is None else fwhm
                        clash = np.any(np.logical_and(q0 > q_min - 2 * sig,
                                                      q0 < q_max + 2 * sig))
                        x[a, idx] = np.nan if clash else np.mean(split_q[idx])
                        y[a, idx] = np.nan if clash else np.mean(split_I[idx])

        self.detector.define_background(x, y, k)
        f = self.detector.background

        if plot:
            plt.plot(self.q[az_idx], I[az_idx], 'k')
            plt.plot(self.q[az_idx], chebval(self.q[az_idx], f[az_idx]), 'r-')
            plt.plot(x[az_idx], y[az_idx], 'r+')

    def peak_fit(self, q0_approx, window_width, func='gaussian',
                 err_lim=1e-4, progress=True):
        """ Single peak fitting to all points/azimuthal slices.

        Fits a Gaussian/Lorentzian/Psuedo-Voigt curve to a peak in a
        defined window. An error limit (wrt strain) can be defined - any
        points that fail to meet this criterion will be replaced with nan.

        Args:
            q0_approx (float): Approximate q0 value
            window_width (float): Curve fitting window
            func (str): Curve fitting func (gaussian/lorentzian/psuedo-voigt)
            err_lim (float): Error limit (default 1e-4)
            progress (bool): Output progress bar
        """
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

    def pawley_fit(self, err_lim=1e-4, q_lim=[2, None], progress=True):
        """ Basic Pawley refinement of full diffraction profile.

        Fits the full diffraction profile according to a Pawley type
        methodology. The detector-sample setup must be properly defined
        (inc. materials/phases via add_materials). This is computational
        expensive - it is recommended that you first call
        plot_intensity(pawley=True) to test goodness of fit.

        The detector/sample setup is used to provide an initial estimate
        of lattice parameter (a), fwhm wrt q and peak intensity. Peak intensity
        is allowed to vary freely. An error limit (wrt strain) can be defined -
        any points that fail to meet this criterion will be replaced with nan.

        Args:
            err_lim (float): Error limit (default 1e-4)
            q_lim (list): [min, max] q to interrogate
            progress (bool): Output progress bar
        """
        # Iterate across q0 values and fit peaks for all detectors
        array_shape = self.I.shape[:-1]
        data = [np.nan * np.ones(array_shape) for _ in range(4)]
        self.peaks, self.peaks_err, self.fwhm, self.fwhm_err = data

        print('\n%s acquisition points\n' % self.I[..., 0, 0].size)

        fit = array_fit_pawley(self.q, self.I, self.detector, err_lim,
                               q_lim, progress)
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
