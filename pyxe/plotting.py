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
from pyxe.plotting_tools import plot_complex, meshgrid_res, line_extract


class StrainPlotting(object):
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
    def __init__(self):
        pass

        
    def plot_intensity(self, detector = 0, point = (), figsize = (7, 5)):
        """
        Plots q v intensity. *Not implemented for merged files.*
        
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty. 
        # point:      Define point (index) from which to extract q v I plot.
                      First point in array chosen if not (default) specified.
        """
        if point == ():
            point = (0, ) * len(self.strain[..., 0, 0].shape)        

        q = self.q[detector]
        I = self.I[point][detector]
        plt.figure(figsize = figsize)
        plt.plot(q, I, 'k-')
        plt.xlabel('q (rad)')
        plt.ylabel('Intensity')

            
    def plot_fitted(self, pnt=(), q_idx=0, figsize=(7, 5)):
        """
        Plots fitted in-plane strain field for given data point. 
        *Not implemented for merged files.*
        
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
            
        p = self.strain_param[pnt][q_idx] ### test change
        R = p[0]
        theta = p[1] + angle

        e_xx, e_yy = cos_(angle, *p), cos_(angle + np.pi/2, *p)
        e_1, e_2 = (p[2] + abs(p[0])), (p[2] - abs(p[0]))
        e_xy = -np.sin(2 * theta) * ((p[2] + p[0]) - (p[2] - p[0]))/2

        fig = plt.figure(figsize = figsize)
        plt.axis('equal')
        ax = fig.add_subplot(1, 1, 1)
        circ = plt.Circle((p[2], 0), radius=R, color='k', fill = False)
        
        ax.add_patch(circ)
        
        plt.xlim([p[2] - abs(2 * R), p[2] + abs(2 * R)])
        plt.plot([e_1, e_2], [0, 0], 'ko', markersize = 3)
        
        plt.plot(e_xx, e_xy, 'ko',label=r'$(\epsilon_{xx}$, $\epsilon_{xy})$')
        plt.plot(e_yy,-e_xy, 'wo',label=r'$(\epsilon_{yy}$, $-\epsilon_{xy})$')
        
        plt.legend(numpoints=1, frameon = False, handletextpad = 0.2)
        plt.plot([e_xx, e_yy], [e_xy, -e_xy], 'k-.')
        ax.annotate('  %s' % r'$\epsilon_{1}$', xy=(e_1, 0), 
                    textcoords='offset points', xytext=(e_1, 0))
        ax.annotate('  %s' % r'$\epsilon_{2}$', xy=(e_2, 0), 
                    textcoords='offset points', xytext=(e_2, 0))
        plt.xlabel('Normal strain')
        plt.ylabel('Shear strain')

    
    def plot_peak_map(self, phi=0, az_idx=None, q_idx=0, z_idx=0, err=False, 
                        FWHM=False, res=0.1, plotting=plot_complex, **kwargs):
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
        x = self.extract_peak_slice(phi, az_idx, q_idx, z_idx, err, FWHM)
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

    def plot_peak_line(self, phi=0, az_idx=None, q_idx=0, z_idx=0, pnt=(0, 0),
                         line_angle=0, npnts=100, axis='scalar', method='linear',
                         err=False, FWHM=False):
        """
        Plots a line profile through a 1D/2D strain field - extract_line 
        method must be run first. *Not yet implemented in 3D.*
                
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty.       
        # q0_index:   Specify lattice parameter/peak to display.   
        # axis:       Plot strain against the 'd1', 'd2' or 'scalar' (default)
                      position. 'scalar' co-ordinates re-zeroed/centred against 
                      point specified in the extract_line command.
        """
 
        dims, data = self.extract_stress_line(phi, az_idx, q_idx, z_idx, pnt, 
                                      line_angle, npnts, method, err, FWHM)
        if len(self.dims) == 1:
            plt.plot(dims, data, '-*')
        
        else:
            d1, d2 = dims
                                    
            if d1[0] > d1[-1]:
                d1, d2, data = d1[::-1], d2[::-1], data[::-1]
                
            zero = ((pnt[0] - d1[0])**2 + (pnt[1] - d2[0])**2)**0.5
            scalar_ext = ((d1 - d1[0])**2 + (d2 - d2[0])**2)**0.5 - zero

            x = {'d1' : d1, 'd2' : d2, 'scalar' : scalar_ext}
            plt.plot(x[axis], data, '-x')
            



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
        if len(self.dims) == 1:
            plt.plot(dims, data, '-*')
        
        else:
            d1, d2 = dims
                                    
            if d1[0] > d1[-1]:
                d1, d2, data = d1[::-1], d2[::-1], data[::-1]
                
            zero = ((pnt[0] - d1[0])**2 + (pnt[1] - d2[0])**2)**0.5
            scalar_ext = ((d1 - d1[0])**2 + (d2 - d2[0])**2)**0.5 - zero

            x = {'d1' : d1, 'd2' : d2, 'scalar' : scalar_ext}
            plt.plot(x[axis], data, '-x')
            
            
    def plot_stress_line(self, phi=0, az_idx=None, q_idx=0, z_idx=0, pnt=(0, 0),
                         line_angle=0, npnts=100, axis='scalar', method = 'linear',
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
 
        dims, data = self.extract_stress_line(phi, az_idx, q_idx, z_idx, pnt, 
                                      line_angle, npnts, method, err, shear)
        if len(self.dims) == 1:
            plt.plot(dims, data, '-*')
        
        else:
            d1, d2 = dims
                                    
            if d1[0] > d1[-1]:
                d1, d2, data = d1[::-1], d2[::-1], data[::-1]
                
            zero = ((pnt[0] - d1[0])**2 + (pnt[1] - d2[0])**2)**0.5
            scalar_ext = ((d1 - d1[0])**2 + (d2 - d2[0])**2)**0.5 - zero

            x = {'d1' : d1, 'd2' : d2, 'scalar' : scalar_ext}
            plt.plot(x[axis], data, '-x')

        
    def plot_line(self, phi = 0, detector = [], q_idx = 0, 
                  line_angle = 0, pnt = (0, 0), npnts = 100, axis = 'scalar', 
                  method = 'linear', stress = False, shear = False):
        """
        Plots a line profile through a 1D/2D strain field - extract_line 
        method must be run first. *Not yet implemented in 3D.*
                
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty.       
        # q0_index:   Specify lattice parameter/peak to display.   
        # axis:       Plot strain against the 'd1', 'd2' or 'scalar' (default)
                      position. 'scalar' co-ordinates re-zeroed/centred against 
                      point specified in the extract_line command.
        """
        error = 'Line plotting only implemented for 1D or 2D data sets'
        assert len(self.dims) <= 2, error      
        
        if len(self.dims) == 1:
            d1, data = self.extract_line(phi, detector, q_idx, pnt, 
                                    line_angle, npnts, method, stress, shear)
            plt.plot(d1, data, '-*')
        
        else:
            d1, d2, data = self.extract_line(phi, detector, q_idx, pnt, 
                                    line_angle, npnts, method, stress, shear)
                                    
            if d1[0] > d1[-1]:
                d1, d2, data = d1[::-1], d2[::-1], data[::-1]
                
            zero = ((pnt[0] - d1[0])**2 + (pnt[1] - d2[0])**2)**0.5
            scalar_ext = ((d1 - d1[0])**2 + (d2 - d2[0])**2)**0.5 - zero

            x = {'d1' : d1, 'd2' : d2, 'scalar' : scalar_ext}
            plt.plot(x[axis], data, '-x')


    def plot_detector(self, detector = 0, q_idx = 0, stress = False, err=False,  
                      res = 0.1, lvls = 11, figsize = (10, 10), line = False,                    
                      pnt = (0,0), line_angle = 0, line_props = 'w-', 
                      plotting = plot_complex, ax=False, cbar =True, **kwargs):
        """
        Plot a 2D heat map of the strain field. *Not yet implemented in 3D.*
        
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty. 
        # q0_index:   Specify lattice parameter/peak to display.  
        # res:        Resolution in points per unit length (of raw data) 
                      - only implemented for merged data
        # cmap:       The colormap (default - 'RdBu_r') to use for plotting
        # lvls:       Number of contours to overlay on map. Can also explicitly 
                      define levels.
        # figsize:    Tuple containing the fig size (x, y) - default (10, 10).
                      Constrained by axis being equal.
        
        Additional functionality allows for the overlaying of a line on top of 
        the map - to be used in conjunction with the line plotting.
        
        # line:       Plot line (default = False)
        # line_props: Define line properties (default = 'w-')
        # mark:       Mark properties for centre point of line (default = None)
        """
        error = 'Plotting method only compatible with 2D data sets'
        assert len(self.dims) == 2, error
                                     
        d1, d2 = [self.co_ords[x] for x in self.dims] 
        
        if err:
            data = 1 - self.strain_err[..., detector, q_idx]
        else:
            data = self.extract_slice(detector = detector, q_idx = q_idx, 
                                      stress = stress)
        
        if data.ndim != 2:
            D1, D2 = meshgrid_res(d1, d2, res)
            Z = griddata((d1.flatten(), d2.flatten()),data.flatten(), (D1, D2))
        else:
            D1, D2, Z = d1, d2, data
            
        ax_ = plotting(d1, d2, D1, D2, Z, lvls = lvls, figsize = figsize, 
                       ax = ax, cbar = cbar, **kwargs)

        if line == True:
            plt.figure()
            line = line_extract(D1, D2, pnt, line_angle, 100)   
            ax.plot(line[0], line[1], line_props, linewidth = 2)
        
        return ax_



    def plot_angle(self, phi = 0, q_idx = 0, stress = False, shear = False,   
                   res = 0.1, lvls = 11, figsize = (10, 10), 
                   line = False, pnt = (0,0), line_angle=0, line_props = 'w-', 
                   plotting = plot_complex, ax = False, cbar = True, **kwargs):
        """
        Plot a 2D heat map of the strain field. *Not yet implemented in 3D.*
        
        # angle:      Angle in radians - default (0). 
        # q0_index:   Specify lattice parameter/peak to display.  
        # res:        Resolution in points per unit length (of raw data) 
                      - only implemented for merged data
        # cmap:       The colormap (default - 'RdBu_r') to use for plotting
        # lvls:       Number of contours to overlay on map. Can also explicitly 
                      define levels.
        # figsize:    Tuple containing the fig size (x, y) - default (10, 10).
                      Constrained by axis being equal.
        
        Additional functionality allows for the overlaying of a line on top of 
        the map - to be used in conjunction with the line plotting.
        
        # line:       Plot line (default = False)
        # line_props: Define line properties (default = 'w-')
        # mark:       Mark properties for centre point of line (default = None)
        """

        error = 'Plotting method only compatible with 2D data sets'
        assert len(self.dims) == 2, error
                                     
        d1, d2 = [self.co_ords[x] for x in self.dims]   
        data = self.extract_slice(phi, q_idx=q_idx, stress=stress, shear=shear)
                    
        if data.ndim != 2:
            D1, D2 = meshgrid_res(d1, d2, spatial_resolution = res)
            Z = griddata((d1.flatten(),d2.flatten()), data.flatten(), (D1, D2))
        else:
            D1, D2, Z = d1, d2, data
            
        ax_ = plotting(d1, d2, D1, D2, Z, lvls = lvls, figsize = figsize, 
                       ax = ax, cbar = cbar, **kwargs)
        
        if line == True:
            plt.figure()
            line = line_extract(D1, D2, pnt, line_angle, 100)   
            ax.plot(line[0], line[1], line_props, linewidth = 2)

        return ax_

