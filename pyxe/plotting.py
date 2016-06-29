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

from pyxe.fitting_functions import cos_, strain_transformation
from pyxe.plotting_tools import plot_complex, meshgrid_res, plot_line


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
        pnt = (0,) * len(self.I[..., 0, 0].shape)

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


class PeakPlotting(object):
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks
    and associated strain.
    """

    def __init__(self, pyxe_object):
        self.peaks = pyxe_object.peaks
        self.peaks_err = pyxe_object.peaks_err
        self.fwhm = pyxe_object.fwhm
        self.fwhm_err = pyxe_object.fwhm_err

    def plot_intensity(self, az_idx=0, pnt=(), figsize=(7, 5), ax=False):
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
            pnt = (0,) * len(self.I[..., 0, 0].shape)

        q = self.q[az_idx]
        I = self.I[pnt][az_idx]
        ax.plot(q, I, 'k-')
        ax.set_xlabel('q (rad)')
        ax.set_ylabel('Intensity')
        return ax

    def plot_peak_map(self, phi=0, az_idx=None, q_idx=0, z_idx=0, err=False,
                      fwhm=False, res=0.1, plotting=plot_complex, **kwargs):
        """
        Plot a 2D heat map of the strain field (or shear/err variants of this).
        Returns plot axis object, which allows for plot customization (e.g.
        axis naming) or the overlaying of additional data.

        Must define **either** an azimuthal angle, phi, or azimuthal (cake)
        index. The azimuthal angle leverages the fitted strain profiles,
        the az_idx plots that specific azimuthal slice. Note that for the
        EDXD detector in I12, az_idx == detector_idx.

        # phi:        Azimuthal angle in radians.
        # az_idx:     Index for azimuthal slice/cake - in EDXD (I12) = detector
        # q_idx:      Specify lattice parameter/peak to display.
        # z_idx:      Slice height (index) for 3D data
        # shear:      Plot shear strain (True/False)
        # err:        Plot strain error (True/False)
        # res:        Resolution in points per unit length (of raw data)
        """
        slice_extract = self.extract_fwhm_slice if fwhm else self.extract_peak_slice
        x = slice_extract(phi, az_idx, q_idx, z_idx, err)
        [d1, d2], data = x

        if data.ndim == 1:
            d1_, d2_ = meshgrid_res(d1, d2, spatial_resolution=res)
            z = griddata((d1.flatten(), d2.flatten()),
                         data.flatten(), (d1_, d2_))
        else:
            d1_, d2_, z = d1, d2, data

        ax_ = plotting(d1, d2, d1_, d2_, z, **kwargs)

        return ax_

    @plot_line
    def plot_peak_line(self, phi=0, az_idx=None, q_idx=0, z_idx=0, pnt=(0, 0),
                       line_angle=0, npnts=100, axis='scalar', method='linear',
                       err=False, fwhm=False):
        """
        Plots a line profile through a 1D/2D strain field - extract_line
        method must be run first. *Not yet implemented in 3D.*

        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty.
        # q0_index:   Specify lattice parameter/peak to display.
        # axis:       Plot strain against the 'd1', 'd2' or 'scalar' (default)
                      position. 'scalar' co-ordinates re-zeroed/centred against
                      point specified in the extract_line command.
        """
        line_extract = self.extract_fwhm_line if fwhm else self.extract_peak_line
        dims, data = line_extract(phi, az_idx, q_idx, z_idx, err,
                                  pnt, line_angle, npnts, method)

        return pnt, axis, dims, data



class StrainPlotting(object):
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
    def __init__(self, pyxe_object):
        self.strain = pyxe_object.strain
        self.strain_err = pyxe_object.strain_err
        self.strain_tensor = pyxe_object.strain_tensor


    def plot_intensity(self, az_idx=0, pnt=(), figsize=(7, 5), ax=False):
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
            pnt = (0, ) * len(self.I[..., 0, 0].shape)        

        q = self.q[az_idx]
        I = self.I[pnt][az_idx]
        ax.plot(q, I, 'k-')
        ax.set_xlabel('q (rad)')
        ax.set_ylabel('Intensity')
        return ax

    def plot_fitted(self, pnt=(), q_idx=0, figsize=(7, 5), ax=False):
        """
        Plots fitted in-plane strain field for given data point. 
        
        # point:      Define point (index) from which to plot fitted in-plane
                      strain field.
        # q_idx:      0 based indexing - 0 (default) to 23 - detector 23 empty.
        # figsize:    Figure dimensions
        """
        pnt = (0, ) * len(self.strain[..., 0, 0].shape) if pnt == () else pnt
        
        if not ax:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)

        p = self.strain_param[pnt][q_idx]
        theta = self.phi
        # Data from edi12 has extra, unused detector (filled with nan) 
        ax.plot(theta, self.strain[pnt][..., q_idx], 'k*')
        theta_2 = np.linspace(self.phi[0], self.phi[-1], 1000)
        ax.plot(theta_2, cos_(theta_2, *p), 'k-')
        ax.set_xlabel(r'$\phi$ (rad)', size=14)
        ax.set_ylabel(r'$\epsilon$', size=14)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))
        
        return ax
            
    def plot_mohrs(self, pnt=(), q_idx=0, angle=0, figsize=(7, 5), ax=False):
        """
        Use fitted in-plane styrain tensor to plot Mohr's circle. 
        
        # pnt:        Define point (index) from which to plot fitted in-plane
                      strain field.
        # q_idx:      0 based indexing - 0 (default) to 23 - detector 23 empty.
        # angle:      Angle to highlight on circle (inc + 90deg).
        # figsize:    Figure dimensions
        """
        if not ax:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)
        ax.axis('equal')
        pnt = (0, ) * len(self.strain[..., 0, 0].shape) if pnt == () else pnt
        p = self.strain_param[pnt][q_idx] 
        theta = p[1] + angle

        e_yy, e_xx = cos_(angle, *p), cos_(angle + np.pi/2, *p)
        e_1, e_2 = (p[2] + abs(p[0])), (p[2] - abs(p[0]))
        e_xy = -np.sin(2 * theta) * ((p[2] + p[0]) - (p[2] - p[0]))/2
        
        r, mean = (e_1 - e_2) / 2, (e_1 + e_2) / 2

        circle = plt.Circle((mean, 0), radius=r, color='k', fill=False)
        ax.add_patch(circle)
        
        ax.set_xlim([mean - abs(2 * r), mean + abs(2 * r)])
        ax.plot([e_1, e_2], [0, 0], 'ko', markersize=3)
        
        ax.plot(e_xx, e_xy, 'ko', label=r'$(\epsilon_{yy}$, $-\epsilon_{xy})$')
        ax.plot(e_yy, -e_xy, 'wo', label=r'$(\epsilon_{xx}$, $\epsilon_{xy})$')
        
        ax.legend(numpoints=1, frameon=False, handletextpad=0.2)
        ax.plot([e_xx, e_yy], [e_xy, -e_xy], 'k-.')
        
        offset = (e_1 - e_2)/30        
        ax.annotate('%s' % r'$\epsilon_{1}$', xy=(e_1 + offset, 0), 
                    textcoords='offset points', xytext=(e_1, 0), size=14)
        ax.annotate('%s' % r'$\epsilon_{2}$', xy=(e_2 + offset, 0), 
                    textcoords='offset points', xytext=(e_2, 0), size=14)
        ax.set_xlabel(r'$\epsilon$', size=14)
        ax.set_ylabel(r'$\gamma$', size=14)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))
        return ax

    def plot_peak_map(self, phi=0, az_idx=None, q_idx=0, z_idx=0, err=False,
                      fwhm=False, res=0.1, plotting=plot_complex, **kwargs):
        """
        Plot a 2D heat map of the strain field (or shear/err variants of this).
        Returns plot axis object, which allows for plot customization (e.g. 
        axis naming) or the overlaying of additional data.
        
        Must define **either** an azimuthal angle, phi, or azimuthal (cake)
        index. The azimuthal angle leverages the fitted strain profiles, 
        the az_idx plots that specific azimuthal slice. Note that for the
        EDXD detector in I12, az_idx == detector_idx.
        
        # phi:        Azimuthal angle in radians. 
        # az_idx:     Index for azimuthal slice/cake - in EDXD (I12) = detector
        # q_idx:      Specify lattice parameter/peak to display.  
        # z_idx:      Slice height (index) for 3D data
        # shear:      Plot shear strain (True/False)
        # err:        Plot strain error (True/False)
        # res:        Resolution in points per unit length (of raw data) 
        """
        slice_extract = self.extract_fwhm_slice if fwhm else self.extract_peak_slice
        x = slice_extract(phi, az_idx, q_idx, z_idx, err)
        [d1, d2], data = x
                    
        if data.ndim == 1:
            d1_, d2_ = meshgrid_res(d1, d2, spatial_resolution=res)
            z = griddata((d1.flatten(), d2.flatten()),
                         data.flatten(), (d1_, d2_))
        else:
            d1_, d2_, z = d1, d2, data
            
        ax_ = plotting(d1, d2, d1_, d2_, z, **kwargs)

        return ax_
            
    def plot_strain_map(self, phi=0, az_idx=None, q_idx=0, z_idx=0, err=False,
                        shear=False, res=0.1, plotting=plot_complex, **kwargs):
        """
        Plot a 2D heat map of the strain field (or shear/err variants of this).
        Returns plot axis object, which allows for plot customization (e.g. 
        axis naming) or the overlaying of additional data.
        
        Must define **either** an azimuthal angle, phi, or azimuthal (cake)
        index. The azimuthal angle leverages the fitted strain profiles, 
        the az_idx plots that specific azimuthal slice. Note that for the
        EDXD detector in I12, az_idx == detector_idx.
        
        # phi:        Azimuthal angle in radians. 
        # az_idx:     Index for azimuthal slice/cake - in EDXD (I12) = detector
        # q_idx:      Specify lattice parameter/peak to display.  
        # z_idx:      Slice height (index) for 3D data
        # shear:      Plot shear strain (True/False)
        # err:        Plot strain error (True/False)
        # res:        Resolution in points per unit length (of raw data) 
        """
        x = self.extract_strain_slice(phi, az_idx, q_idx, z_idx, err, shear)
        [d1, d2], data = x
                    
        if data.ndim == 1:
            d1_, d2_ = meshgrid_res(d1, d2, spatial_resolution=res)
            z = griddata((d1.flatten(), d2.flatten()),
                         data.flatten(), (d1_, d2_))
        else:
            d1_, d2_, z = d1, d2, data
            
        ax_ = plotting(d1, d2, d1_, d2_, z, **kwargs)

        return ax_
    
    def plot_stress_map(self, phi=0, az_idx=None, q_idx=0, z_idx=0, err=False,
                        shear=False, res=0.1, plotting=plot_complex, **kwargs):
        """
        Plot a 2D heat map of the stress field (or shear/err variants of this).
        Returns plot axis object, which allows for plot customization (e.g. 
        axis naming) or the overlaying of additional data.
        
        Must define **either** an azimuthal angle, phi, or azimuthal (cake)
        index. The azimuthal angle leverages the fitted strain profiles, 
        the az_idx plots that specific azimuthal slice. Note that for the
        EDXD detector in I12, az_idx == detector_idx.
        
        # phi:        Azimuthal angle in radians. 
        # az_idx:     Index for azimuthal slice.
        # q_idx:      Specify lattice parameter/peak to display.  
        # z_idx:      Slice height (index) for 3D data
        # shear:      Plot shear stress (True/False)
        # err:        Plot stress error (True/False)
        # res:        Resolution in points per unit length (of raw data) 
        """
        x = self.extract_stress_slice(phi, az_idx, q_idx, z_idx, err, shear)
        [d1, d2], data = x
                    
        if data.ndim == 1:
            d1_, d2_ = meshgrid_res(d1, d2, spatial_resolution=res)
            z = griddata((d1.flatten(), d2.flatten()),
                         data.flatten(), (d1_, d2_))
        else:
            d1_, d2_, z = d1, d2, data
            
        ax_ = plotting(d1, d2, d1_, d2_, z, **kwargs)

        return ax_

    @plot_line
    def plot_peak_line(self, phi=0, az_idx=None, q_idx=0, z_idx=0, pnt=(0, 0),
                       line_angle=0, npnts=100, axis='scalar', method='linear',
                       err=False, fwhm=False):
        """
        Plots a line profile through a 1D/2D strain field - extract_line 
        method must be run first. *Not yet implemented in 3D.*
                
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty.       
        # q0_index:   Specify lattice parameter/peak to display.   
        # axis:       Plot strain against the 'd1', 'd2' or 'scalar' (default)
                      position. 'scalar' co-ordinates re-zeroed/centred against 
                      point specified in the extract_line command.
        """
        line_extract = self.extract_fwhm_line if fwhm else self.extract_peak_line
        dims, data = line_extract(phi, az_idx, q_idx, z_idx, err, 
                                  pnt, line_angle, npnts, method)
                                    
        return pnt, axis, dims, data 

    @plot_line
    def plot_strain_line(self, phi=0, az_idx=None, q_idx=0, z_idx=0, err=False,
                         shear=False, pnt=(0, 0), line_angle=0, npnts=100,
                         method='linear', axis='scalar'):
        """
        Plots a line profile through a 1D/2D strain field - extract_line 
        method must be run first. *Not yet implemented in 3D.*
                
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty.       
        # q0_index:   Specify lattice parameter/peak to display.   
        # axis:       Plot strain against the 'd1', 'd2' or 'scalar' (default)
                      position. 'scalar' co-ordinates re-zeroed/centred against 
                      point specified in the extract_line command.
        """
 
        dims, data = self.extract_strain_line(phi, az_idx, q_idx, z_idx, err, 
                                              shear, pnt, line_angle, npnts,
                                              method)
        return pnt, axis, dims, data
            
    @plot_line            
    def plot_stress_line(self, phi=0, az_idx=None, q_idx=0, z_idx=0,
                         pnt=(0, 0), line_angle=0, npnts=100, axis='scalar',
                         method='linear', shear=False, err=False):
        """
        Plots a line profile through a 1D/2D strain field - extract_line 
        method must be run first. *Not yet implemented in 3D.*
                
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty.       
        # q0_index:   Specify lattice parameter/peak to display.   
        # axis:       Plot strain against the 'd1', 'd2' or 'scalar' (default)
                      position. 'scalar' co-ordinates re-zeroed/centred against 
                      point specified in the extract_line command.
        """
 
        dims, data = self.extract_stress_line(phi, az_idx, q_idx, z_idx, err, 
                                              shear, pnt, line_angle, npnts,
                                              method)
        return pnt, axis, dims, data
