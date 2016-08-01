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
from numpy.polynomial.chebyshev import chebfit, chebval
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from matplotlib.pyplot import cm

from pyxe.command_parsing import analysis_check
from pyxe.fitting_functions import strain_transformation, shear_transformation
from pyxe.plotting_tools import plot_complex, meshgrid_res, line_extract, az90
from pyxe.command_parsing import complex_check, text_cleaning, name_convert
from pyxe.analysis_tools import data_extract
from pyxe.fitting_functions import plane_stress, plane_strain
from pyxe.fitting_tools import pawley_hkl, extract_parameters


class DataViz(object):

    def __init__(self, fpath):
        """
        # fpath:      Data is either the filepath to an analyzed pyxe NeXus
                      file or a pyxe data object (inc. merged object)
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

    def flipaxis(self, axis):
        """
        Flip axis (positive to negative).
        """
        axes = {0: self.d1, 1: self.d2, 2: self.d3}
        axes[axis] *= -1

    def swapaxes(self, axis1, axis2):
        """
        Interchange two axes of an array. This effectively rotates the data.
        """
        axes = {0: self.d1, 1: self.d2, 2: self.d3}
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]

    def centre(self, coordinate):
        """
        Centre array on point (tuple).
        """
        axes = {0: self.d1, 1: self.d2, 2: self.d3}
        for idx, i in enumerate(coordinate):
            axes[idx] -= i

    def plot_intensity(self, pnt=None, az_idx=0, figsize=(9, 6), ax=False,
                       pawley=True):
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
        q, I = self.q[az_idx], self.I[pnt][az_idx]


        if pawley:
            detector = self.detector
            background = chebval(q, detector.background[az_idx])
            q_lim = [np.min(q), np.max(q)]
            p0 = extract_parameters(detector, q_lim, np.nanmax(I))
            pawley = pawley_hkl(detector, background)
            coeff, var_mat = curve_fit(pawley, q, I, p0=p0)
            I_pawley = pawley(q, *coeff)

            ax.plot(q, I, 'o', markeredgecolor='0.3', markersize=4, markerfacecolor='none', label=r'$\mathregular{Y_{obs}}$')
            ax.plot(q, I_pawley, '-', color='r', linewidth=0.75, label=r'$\mathregular{Y_{calc}}$')

            I_diff = I - I_pawley
            max_diff = np.max(I_diff)


            ylim = ax.get_ylim()
            ymin = -ylim[1]/10

            materials = self.detector.materials

            for idx, mat in enumerate(materials):
                offset = (1 + idx) * ymin / 2
                for q0 in self.detector.q0[mat]:
                    ax.plot([q0, q0], [offset + ymin/8, offset - ymin/8], '-', color='g', linewidth=2)

                ax.errorbar(0, 0, yerr=1, fmt='none', capsize=0,
                            ecolor='g', elinewidth=1.5, label=r'Bragg ({})'.format(mat))

            ax.plot(q, I - I_pawley + (idx + 2) * ymin/2 - max_diff, '-', color='b', linewidth=0.75,
                    label=r'$\mathregular{Y_{diff}}$')

            ylocs = ax.yaxis.get_majorticklocs()
            yticks = ax.yaxis.get_major_ticks()
            for idx, yloc in enumerate(ylocs):
                if yloc < 0:
                    yticks[idx].set_visible(False)



            ax.legend(numpoints=1)





        else:
            ax.plot(q, I, 'k-', linewidth=0.5)
        ax.set_xlabel('q (A$^{-1}$)')
        ax.set_ylabel('Intensity')




        return ax

    @analysis_check('strain fit')
    def plot_strain_fit(self, pnt=None, figsize=(10,5)):
        """
        Plots fitted in-plane strain field for given data point.

        # pnt:      Define point (index) from which to plot fitted in-plane
                      strain field.
        # figsize:    Figure dimensions
        """
        pnt = (0,) * (self.strain.ndim - 1) if pnt is None else pnt

        fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=figsize)

        p = self.strain_tensor[pnt]
        ax_1.plot(self.phi, self.strain[pnt], 'k*')
        phi_2 = np.linspace(self.phi[0], self.phi[-1], 1000)
        ax_1.plot(phi_2, strain_transformation(phi_2, *p), 'k-', linewidth=0.5)
        ax_1.set_xlabel(r'$\phi$ (rad)', size=14)
        ax_1.set_ylabel(r'$\epsilon$', size=14)
        ax_1.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))

        ax_2.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))
        ax_2.set_xlabel(r'$\epsilon$', size=14)
        ax_2.set_ylabel(r'$\gamma}$', size=14)

        e_xx, e_yy, e_xy = self.strain_tensor[pnt]
        mean = (e_xx + e_yy) / 2
        e_1 = mean + (e_xy**2 + ((e_xx - e_yy) / 2)**2)**0.5
        e_2 = e_xx + e_yy - e_1
        radius = (e_1 - e_2) / 2

        for x, text in zip([self.phi[0], self.phi[0] + np.pi/2],
                           [r'$\epsilon_{xx}$', r'$\epsilon_{yy}$']):
            ax_1.axvline(x, ymax=0.93, linewidth=0.5, ls='--', color='k')
            y = ax_1.get_ylim()[1] * 0.96
            ax_1.text(x, y, text, ha='center', va='bottom')

        circ = plt.Circle((mean, 0), radius=radius, color='k', fill=False)
        ax_2.add_patch(circ)
        for x, y, text in zip([e_1, e_2, e_xx, e_yy],
                              [0, 0, e_xy, -e_xy],
                              [r'$\epsilon_{1}$', r'$\epsilon_{2}$',
                               r'$(\epsilon_{xx}$, $\epsilon_{xy})$',
                               r'$(\epsilon_{yy}$, $\epsilon_{yx})$']):
            ax_2.plot(x, y, 'k.')
            ax_2.annotate('  %s' % text, xy=(x, y),  xytext=(x, y), ha='left')
        ax_2.plot([e_xx, e_yy], [e_xy, -e_xy], 'k--', linewidth=0.5)
        fig.tight_layout()

    def extract_line(self, data='strain', phi=None, az_idx=None,
                     pnt=None, theta=0, z_idx=None, res=0.1):
        data = self.extract_slice(data, phi, az_idx, z_idx)
        if self.ndim == 1:
            return self.d1, data
        else:
            x, y, d = line_extract(self.d1, self.d2, pnt, theta, res)
            print(x.shape, y.shape, d.shape)
            co_ords = (self.d1.flatten(), self.d2.flatten())
            line = griddata(co_ords, data.flatten(), (x, y))
            print(line.shape)
            return x, y, d, line

    def extract_slice(self, data='strain', phi=None, az_idx=None, z_idx=None):
        """ Extract 2D data slice wrt. azimuthal slice index or angle (phi).

        The extracted data is defined using the data variable (str), which
        must be one of the following: peaks, peaks error, fwhm, fwhm error,
        strain, strain error, shear strain, stress, shear stress.

        Certain combinations of data type and azimuthal index/phi will not
        work (e.g. can't extract peaks wrt. phi only wrt. az. index).

        Note: must define EITHER phi or az_idx

        Args:
            data (str): Data type to extract (see above)
            phi (float): Azimuthal angle in rad
            az_idx (int): Azimuthal slice index
            z_idx (int): Index of slice height in 3D array
        """
        complex_check(data, self.analysis_state, phi, az_idx)
        command = text_cleaning(data)
        az_command = 'phi' if phi is not None else 'az_idx'

        if az_command == 'az_idx':
            az_idx = int(az_idx)
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
                e_xx, e_yy = d[..., az_idx], d[..., az90(self.phi, az_idx)]
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
                   res=0.051, plot_func=None, **kwargs):
        """ Plot 2D data slice wrt. azimuthal slice index or angle (phi).

        The extracted data is defined using the data variable (str), which
        must be one of the following: peaks, peaks error, fwhm, fwhm error,
        strain, strain error, shear strain, stress, shear stress.

        Certain combinations of data type and azimuthal index/phi will not
        work (e.g. can't plot peaks wrt. phi only wrt. az. index):

        Note: must define EITHER phi or az_idx

        Args:
            data (str): Data type to extract (see above)
            phi (float): Azimuthal angle in rad
            az_idx (int): Azimuthal slice index
            z_idx (int): Index of slice height in 3D array
            res (float): Resolution of re-gridded/plotted data points in mm
            plot_func (func): User defined plotting function
            kwargs (list): Passed to plot_func.
        """
        data = self.extract_slice(data, phi, az_idx, z_idx)
        finite = np.isfinite(data)
        d1, d2 = meshgrid_res(self.d1[finite], self.d2[finite],
                              spatial_resolution=res)
        z = griddata((self.d1[finite], self.d2[finite]), data[finite], (d1, d2))
        plot_func = plot_complex if plot_func is None else plot_func
        ax_ = plot_func(self.d1[finite], self.d2[finite], d1, d2, z, **kwargs)

        return ax_

    def save_to_txt(self, fname, data, phi=None, az_idx=None, perp=True):
        n_lst = [d for d in ['d1', 'd2', 'd3'] if getattr(self, d) is not None]
        d_lst = [getattr(self, d) for d in n_lst]

        for d in data:
            print(d)
            name = name_convert(d, phi, az_idx)
            d_lst.append(self.extract_slice(d, phi=phi, az_idx=az_idx))
            n_lst.append(name)

            if perp:
                a90 = az90(self.phi, az_idx) if az_idx is not None else az_idx
                p90 = phi + np.pi/2 if phi is not None else phi
                name = name_convert(d, p90, a90, perp)
                d_lst.append(self.extract_slice(d, phi=p90, az_idx=a90))
                n_lst.append(name)
        data = np.hstack([d.reshape(d.size, 1) for d in d_lst])
        headers = ','.join(n_lst)
        np.savetxt(fname, data, delimiter=',', header=headers)
