# -*- coding: utf-8 -*-
""" Analysis module for the interrogation of integrated diffraction profiles.

The foundation of the pyXe package, allowing for single peak fitting and, more
recently, a Pawley type refinement of the complete diffracted profile. The
peak fitting is completed at each acquisiion point for every azimuthal slice.
Inherits from the DataViz class for plotting functionality.

The fitted peaks (or calculated lattice parameter) can be compared against
a stress free equivalent to allow for the calculation of strain. It is
recommended that rather than pass in a single value of q0/a0, an analyzed,
stress free, pyxe object is supplied. This essentially allows q0/a0 to vary
wrt. azimuthal position, which helps account for errors in beam centering.

The variation in strain wrt. azimuthal position can be used to calculate the
full strain tensor. This is done according to the traditional stress/strain
transformation equations. The strain tensor should improve the accuracy of
calculated strain and allow for the computation of strain at any arbitrary
azmimuthal angle (not just at slice positions). In addition to this, the
strain tensor provides information about the shear strain.

It is also possible to supply material parameters (and stress state), which
will allow for the calculation of stress/shear stress.

The pyxe data objects can be saved an reloaded at any point through the
analysis process.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval
from scipy.interpolate import interp1d, interp2d

from pyxe.command_parsing import analysis_check
from pyxe.fitting_tools import (array_fit, array_fit_pawley, full_ring_fit,
    single_pawley, fwhm_single, q0_valid_range)
from pyxe.data_io import pyxe_to_hdf5, data_extract, detector_extract
from pyxe.plotting import DataViz
from pyxe.merge import basic_merge
from pyxe.fitting_functions import plane_strain, plane_stress



class PeakAnalysis(DataViz):
    def __init__(self, fpath):
        """ Analysis class for the calculation of peaks and strain.

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
            self.detector = detector_extract(f)
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
            self.define_background(plot=False)
            
    def define_temperature(self, T, x=None, y=None, kind='linear', 
                           bounds_error=True, fill_value=np.nan, plot=True):
        """ Fits a function to temperature data - calculates T across all pos.
        
        See documentation fro interpd1d/interp2d
        
        Not yet able to take 3D data - is this needed? This is to be used in 
        conjunciton with other stuff.
        """
        if x is None:
            f = interp1d(y, T, kind=kind, bounds_error=bounds_error, 
                         fill_value=fill_value)
            self.T = f(self.d2)
            if plot:
                self.plot_temperature(1)
                
        elif y is None:
            f = interp1d(x, T, kind=kind, bounds_error=bounds_error, 
                         fill_value=fill_value)
            self.T = f(self.d1)
            if plot:
                self.plot_temperature(1)
            
        else:
            f = interp2d(x, y, T, kind=kind, bounds_error=bounds_error, 
                         fill_value=fill_value)
            self.T = f(self.d1, self.d2)
            if plot:
                self.plot_temperature(2)
        

        
    
    def plot_temperature(self, order=1):
        
        if order == 1:
            plt.plot(self.d1, self.T) 
        else:
            plt.contourf(self.T)

    def define_background(self, seg=50, k=1, pnt=None, fwhm=None, plot=True,
                          az_idx=0, auto=True, x=None, y=None):
        """ Background profile fitting - segment data and auto find points.

        Fits a Chebyshev polynomial (of order k) to automatically acquired
        q, I points. These points are selected wrt. the materials present
        (defined via add_materials), with the Bragg peaks being avoided.

        If auto is False, q v I points can be manually specified.

        Args:
            seg (int): Number of points to split data into/extract
            k (int): Order for Chebyshev polynomial
            pnt (tuple): Point co_ords or None for averaged data
            fwhm (float): Peak fwhm (to exclude more data around peaks)
            plot (bool): True/False
            az_idx (int): Azimuthal slice to plot
            auto (bool): Automatically extract q v I data
            x (ndarray): If not auto maunally defined q values
            y (ndarray): If not auto maunally defined I values
        """
        # Automatically averages over all points unless point is specified

        assert not self.detector.materials == {}, 'Define material(s)'

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
                        fw = self.detector.fwhm[mat] if fwhm is None else fwhm
                        clash = np.any(np.logical_and(q0 > q_min - 2 * fw,
                                                      q0 < q_max + 2 * fw))

                        x[a, idx] = np.nan if clash else np.mean(split_q[idx])
                        y[a, idx] = np.nan if clash else np.mean(split_I[idx])

        # print(x, y)
        self.detector.define_background(x, y, k)
        f = self.detector._back

        if plot:
            plt.plot(self.q[az_idx], I[az_idx], 'k')
            plt.plot(self.q[az_idx], chebval(self.q[az_idx], f[az_idx]), 'r-')
            plt.plot(x[az_idx], y[az_idx], 'r+')
            plt.ylim([0, np.nanmax(y) * 1.5])
            plt.xlabel(r'$\mathregular{q (A^{-1})}$')
            plt.ylabel(r'Intensity')

    def estimate_fwhm(self, pnt, az_idx=0, k=None, single=False,
                      window=None, store=True):
        """ FWHM estimation and fitting.

        Re-estimation of the polynomial parameters used for the FWHM fit.
        These values are used in the complete Pawley fit and improving the
        initial estimate may aid convergence. The order of the polynomial may
        be reduced from 2 to 1. This has been observed to help in situations
        in which the FWHM distribution is linear and not quadratic.

        Args:
            az_idx (int): Azimuthal slice to plot
            pnt (tuple): Point co_ords or None for averaged data
            k (int): Order for FWHM polynomial
            single(bool): Show the FWHM values calc. from single peak fit
            window (tuple): Window used for single peak fit
            store (bool): Store new estimation of parameters
        """
        # Automatically average over all points unless point is specified
        #ndim = self.I.ndim - 2
        #I = np.sum(self.I, tuple(range(ndim))) if pnt is None else self.I[pnt]
        I = self.I[pnt]
        q, I = self.q[az_idx], I[az_idx]
        detector, back = self.detector, self.detector._back[az_idx]
        nmat, nf = len(detector.materials), len(detector._fwhm)

        # Initial fitting using detector based estimated (k=2)
        k = nf if k is None else k
        error = "k must be less or equal to than current polynomial order."
        assert nf - 1 >= k, error

        # Calculate the polynomials from Pawley fit
        q0_all = np.concatenate([detector.q0[i] for i in detector.materials])
        estp0 = single_pawley(detector, q, I, back)
        estp1 = None
        if k < nf - 1:
            f0 = estp0[0][nmat:nmat+nf]
            f = np.polyfit(q0_all, np.polyval(f0, q0_all), k)
            estp1 = single_pawley(detector, q, I, back, f)

        # Evaluate and plot fwhm from polynomials
        q0_all = q0_all[np.argsort(q0_all)]
        fwhm0 = np.polyval(detector._fwhm, q0_all) #** 0.5
        plt.plot(q0_all, fwhm0, 'r--', label='Est0 (k={})'.format(nf - 1))
        for idx, (est, c) in enumerate(zip([estp0, estp1], ['r', 'k'])):
            if est is not None:
                i = range(nmat, nmat + nf - idx)
                coeff, var_mat = est
                fwhm = np.polyval(coeff[i], q0_all) # ** 0.5
                #print(fwhm)
                q0_min, q0_max = q0_valid_range(detector, [q.min(), q.max()])
                q0_range = np.linspace(q0_min, q0_max, 100)
                perr = np.sqrt(np.diag(var_mat))
                err = np.sum(np.polyval(perr[i], q0_range) ** 0.5) / 100
                label = 'Est{} (k={}, e={:.0e})'.format(idx, nf - 1 - idx, err)
                plt.plot(q0_all, fwhm, '.-', color=c, label=label)

        # Store new default parameters
        detector._fwhm = list(coeff[i]) if store else detector._fwhm

        # Attempt to find FWHM for each peak individually (for comparison)
        if single:
            q0_v, fw_v = fwhm_single(detector, q, I, window)
            #print('fw_v', fw_v)
            plt.plot(q0_v, fw_v, 'o', color='0.75', label='Single')
        plt.legend(numpoints=1, loc=2)
        plt.xlabel(r'$\mathregular{q (A^{-1})}$')
        plt.ylabel(r'$\mathregular{FWHM (A^{-1})}$')

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

    def pawley_fit(self, err_lim=1e-4, q_lim=[2, None], progress=True, func='gaussian'):
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
                               q_lim, progress, func=func)
        self.peaks, self.peaks_err, self.fwhm, self.fwhm_err = fit
        # Reset strain to None after peak fitting...
        self.strain, self.strain_err, self.strain_tensor = None, None, None
        self.analysis_state = 'peaks'

    @analysis_check('peaks')
    def calculate_strain(self, q0=None, a0=None, tensor_fit=True, f=None, variables=None):
        """ Calculate strain based on q0/a0 value(s).

        Strain can be calculated with respect to a single value of
        q0 or a0 but it is recommended to pass in a analyzed pyxe
        data object containing q0/a0 measurements. Strain can then be
        assessed wrt. azimuthal position, which reduces error in the case
        of an area detector (see Korsunsky et al. ??)

        There is an option to then compute the full strain tensor. This fits
        the stress/strain transformation equations to the data and stores
        e_xx, e_yy, e_xy.

        Args:
            q0 (float, object): q0 float or pyxe data object
            a0 (float, object): a0 float or pyxe data object
            tensor_fit (bool): Calculate the full strain tensor
            f (func): interp1d or interpd2d function
            variables (list): list of variables ('d1', 'd2', 'T' etc) for func
        """
        assert (q0, a0) is not (None, None), 'Specify either q0 or a0'
        q0 = a0 if a0 is not None else q0

        if f is not None:
            vals = []
            for v in variables:
                vals.append(self.__dict__[v])
            # This should just tell us the distribution of q0/a0 wrt. 
            # variables not wrt. phi
            q0_mean = f(*vals)
            print('q0_mean', np.shape(q0_mean))
            #q0_mean.respae()

        if isinstance(q0, PeakAnalysis):
            assert np.array_equal(q0.phi, self.phi)
            q0 = np.nanmean(q0.peaks, axis=tuple(range(0, q0.peaks.ndim - 1)))
                

        if q0 is None:
            
            if f is not None:
                a0_mean = q0_mean.reshape(q0_mean.shape + (1,))
                a0 = np.array(a0).reshape(1, -1)
                a0 /= np.nanmean(a0) ## Consider it in relation to e eqn..?
                a0 = a0_mean * a0
            self.a0 = a0
            self.strain = (self.peaks - self.a0) / self.a0
            self.strain_err = self.peaks_err / self.a0
        else:
            
            if f is not None:
                q0_mean = q0_mean.reshape(q0_mean.shape + (1,))
                print('q0_mean', np.shape(q0_mean))
                q0 = np.array(q0).reshape(1, -1)
                print('q0', np.shape(q0))
                q0 /= np.nanmean(q0) ## Consider it in relation to e eqn..?
                print('q0', np.shape(q0))
                q0 = q0_mean * q0 
                print('q0', np.shape(q0))
                
            self.q0 = q0
            self.strain = (self.q0 / self.peaks) - 1
            self.strain_err = 1 - self.q0 / (self.q0 + self.peaks_err) ## ???
            #### self.strain_err = (self.q0 / self.peaks_err) - 1

        if tensor_fit:
            self.strain_tensor = full_ring_fit(self.strain, self.phi)
            self.analysis_state = 'strain fit'
        else:
            self.analysis_state = 'strain'


    @analysis_check('strain')
    def material_parameters(self, E, v, G=None, stress_state='plane strain'):
        """ Apply material parameters and stress state to system.

        The parameters are used for the calcaultion of stress - currently
        no option to apply different parameters for different materials in
        multi-material systems.

        Args:
            E (float): Young's modulus (MPa)
            v (float): Poisson's ratio
            G (float): Shear modulus else estimated from E/v if None
            stress_state (str): plane stress or plane strain stress state
        """

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
