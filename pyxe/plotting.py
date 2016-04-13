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
from pyxe.fitting_functions import cos_
from pyxe.plotting_tools import plot_complex, meshgrid_res, mohrs_dec, plot_line


class StrainPlotting(object):
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
    def __init__(self):
        pass

        
    def plot_intensity(self, az_idx=0, pnt=(), figsize=(7, 5)):
        """
        Plots q v intensity.
        
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty. 
        # point:      Define point (index) from which to extract q v I plot.
                      First point in array chosen if not (default) specified.
        """
        if pnt == ():
            pnt = (0, ) * len(self.I[..., 0, 0].shape)        

        q = self.q[az_idx]
        I = self.I[pnt][az_idx]
        plt.figure(figsize = figsize)
        plt.plot(q, I, 'k-')
        plt.xlabel('q (rad)')
        plt.ylabel('Intensity')

            
    def plot_fitted(self, pnt=(), q_idx=0, figsize=(7, 5)):
        """
        Plots fitted in-plane strain field for given data point. 
        
        # point:      Define point (index) from which to plot fitted in-plane
                      strain field.
        # q_idx:      0 based indexing - 0 (default) to 23 - detector 23 empty.
        # figsize:    Figure dimensions
        """
        pnt = (0, ) * len(self.strain[..., 0, 0].shape) if pnt == () else pnt

        plt.figure(figsize = figsize)
        p = self.strain_param[pnt][q_idx]
        theta = self.phi
        # Data from edi12 has extra, unused detector (filled with nan) 
        try:
            plt.plot(theta, self.strain[pnt][..., q_idx], 'k*')
        except ValueError:
            plt.plot(theta, self.strain[pnt][..., q_idx][:-1], 'k*')
        theta_2 = np.linspace(self.phi[0], self.phi[-1], 1000)
        plt.plot(theta_2, cos_(theta_2, *p), 'k-')
        plt.xlabel('Azimuthal angle (rad)')
        plt.ylabel('Strain')
            
    @mohrs_dec        
    def plot_mohrs(self, pnt =(), q_idx=0, angle=-np.pi, figsize=(7, 5)):
        """
        Use fitted in-plane styrain tensor to plot Mohr's circle. 
        
        # pnt:        Define point (index) from which to plot fitted in-plane
                      strain field.
        # q_idx:      0 based indexing - 0 (default) to 23 - detector 23 empty.
        # angle:      Angle to highlight on circle (inc + 90deg).
        # figsize:    Figure dimensions
        """
        pnt = (0, ) * len(self.strain[..., 0, 0].shape) if pnt == () else pnt
        p = self.strain_param[pnt][q_idx] 
        theta = p[1] + angle

        e_xx, e_yy = cos_(angle, *p), cos_(angle + np.pi/2, *p)
        e_1, e_2 = (p[2] + abs(p[0])), (p[2] - abs(p[0]))
        e_xy = -np.sin(2 * theta) * ((p[2] + p[0]) - (p[2] - p[0]))/2
        
        return e_xx, e_yy, e_xy, e_1, e_2

    
    def plot_peak_map(self, phi=0, az_idx=None, q_idx=0, z_idx=0, err=False, 
                        fwhm=False, res=0.1, plotting=plot_complex, **kwargs):
        """
        Plot a 2D heat map of the strain field (or shear/error variants of this).
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
        x = self.extract_peak_slice(phi, az_idx, q_idx, z_idx, err, fwhm)
        [d1, d2], data = x
                    
        if data.ndim == 1:
            D1, D2 = meshgrid_res(d1, d2, spatial_resolution = res)
            Z = griddata((d1.flatten(),d2.flatten()), data.flatten(), (D1, D2))
        else:
            D1, D2, Z = d1, d2, data
            
        ax_ = plotting(d1, d2, D1, D2, Z, **kwargs)

        return ax_
            
            
    def plot_strain_map(self, phi=0, az_idx=None, q_idx=0, z_idx=0, err=False, 
                        shear=False, res=0.1, plotting=plot_complex, **kwargs):
        """
        Plot a 2D heat map of the strain field (or shear/error variants of this).
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
            D1, D2 = meshgrid_res(d1, d2, spatial_resolution = res)
            Z = griddata((d1.flatten(),d2.flatten()), data.flatten(), (D1, D2))
        else:
            D1, D2, Z = d1, d2, data
            
        ax_ = plotting(d1, d2, D1, D2, Z, **kwargs)

        return ax_
    
    
    def plot_stress_map(self, phi=0, az_idx=None, q_idx=0, z_idx=0, err=False, 
                        shear=False, res=0.1, plotting=plot_complex, **kwargs):
        """
        Plot a 2D heat map of the stress field (or shear/error variants of this).
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
            D1, D2 = meshgrid_res(d1, d2, spatial_resolution = res)
            Z = griddata((d1.flatten(),d2.flatten()), data.flatten(), (D1, D2))
        else:
            D1, D2, Z = d1, d2, data
            
        ax_ = plotting(d1, d2, D1, D2, Z, **kwargs)

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
 
        dims, data = self.extract_peak_line(phi, az_idx, q_idx, z_idx, err, 
                                    fwhm, pnt, line_angle, npnts, method)
                                    
        return pnt, axis, dims, data 

            
    @plot_line
    def plot_strain_line(self, phi=0, az_idx=None, q_idx=0, z_idx=0, err=False,
                         shear = False, pnt=(0, 0), line_angle=0, npnts=100, 
                         method = 'linear', axis='scalar'):
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
                                    shear, pnt, line_angle, npnts, method)
        
        return pnt, axis, dims, data 
            
    @plot_line            
    def plot_stress_line(self, phi=0, az_idx=None, q_idx=0, z_idx=0, pnt=(0, 0),
                         line_angle=0, npnts=100, axis='scalar', method='linear',
                         shear = False, err=False):
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
                                    shear, pnt, line_angle, npnts, method)
        
        return pnt, axis, dims, data 
            