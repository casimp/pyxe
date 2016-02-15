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
from edi12.peak_fitting import cos_
from edi12.plotting import plot_complex


class XRD_plotting():
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
    def __init__(self):
        pass

        
    def plot_intensity(self, detector = 0, point = ()):
        """
        Plots q v intensity. *Not implemented for merged files.*
        
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty. 
        # point:      Define point (index) from which to extract q v I plot.
                      First point in array chosen if not (default) specified.
        """
        try:
            group = self.f['entry1']['EDXD_elements']
            q = group['edxd_q'][detector]
            if point == ():
                point = (0, ) * len(self.dims)
            assert len(self.dims) == len(point), 'must define point with correct '\
                                     'number of dimensions.'
            I = group['data'][point][detector]

            plt.plot(q, I)
        except (NameError, AttributeError):
            print("Can't plot spectrum on merged data.")
            
            
    def plot_fitted(self, point = (), q_idx = 0, figsize = (7, 5)):
        """
        Plots fitted in-plane strain field for given data point. 
        *Not implemented for merged files.*
        
        # point:      Define point (index) from which to plot fitted in-plane
                      strain field.
        # q_idx:      0 based indexing - 0 (default) to 23 - detector 23 empty.
        # figsize:    Figure dimensions
        """
        if point == ():
            point = (0, ) * len(self.dims)

        assert len(self.dims) == len(point), 'must define point with correct '\
                                             'number of dimensions.'
        
        plt.figure(figsize = figsize)
        p = self.strain_param[point][q_idx]
        theta = np.linspace(0, np.pi, 23)
        plt.plot(theta, self.strain[point][..., q_idx][:-1], 'k*')
        theta_2 = np.linspace(0, np.pi, 1000)
        plt.plot(theta_2, cos_(theta_2, *p), 'k-')
        plt.xlabel('Detector angle (rad)')
        plt.ylabel('Strain')
            
            
    def plot_mohrs(self, point = (0), q_idx = 0, angle = 0, figsize = (7, 5)):
        """
        Use fitted in-plane styrain tensor to plot Mohr's circle. 
        *Not implemented for merged files.*
        
        # point:      Define point (index) from which to plot fitted in-plane
                      strain field.
        # q_idx:      0 based indexing - 0 (default) to 23 - detector 23 empty.
        # angle:      Angle to highlight on circle (inc + 90deg).
        # figsize:    Figure dimensions
        """
        p = self.strain_param[point][q_idx][0] ### test change
        R = p[0]
        theta = p[1] + angle

        e_xx, e_yy = cos_(angle, *p), cos_(angle + np.pi/2, *p)
        e_1, e_2 = (p[2] + abs(p[0])), (p[2] - abs(p[0]))
        tau_xy = -np.sin(2 * theta) * ((p[2] + p[0]) - (p[2] - p[0]))/2

        fig = plt.figure(figsize = figsize)
        plt.axis('equal')
        ax = fig.add_subplot(1, 1, 1)
        circ = plt.Circle((p[2], 0), radius=R, color='k', fill = False)
        
        ax.add_patch(circ)
        
        plt.xlim([p[2] - abs(2 * R), p[2] + abs(2 * R)])
        plt.plot([e_1, e_2], [0, 0], 'ko', markersize = 3)
        
        plt.plot(e_xx, tau_xy, 'ko', label = r'$(\epsilon_{xx}$, $\tau_{xy})$')
        plt.plot(e_yy, -tau_xy, 'wo', label = r'$(\epsilon_{yy}$, $-\tau_{xy})$')
        
        plt.legend(numpoints=1, frameon = False, handletextpad = 0.2)
        plt.plot([e_xx, e_yy], [tau_xy, -tau_xy], 'k-.')
        ax.annotate('  %s' % r'$\epsilon_{1}$', xy=(e_1, 0), textcoords='offset points')
        ax.annotate('  %s' % r'$\epsilon_{2}$', xy=(e_2, 0), textcoords='offset points')


    def plot_line(self, detector = 0, q0_index = 0, axis = 'scalar', 
                  pnt = (0, 0), angle = 0, npnts = 100, method = 'linear'):
        """
        Plots a line profile through a 2D strain field - extract_line method
        must be run first. *Not yet implemented in 3D.*
                
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty.       
        # q0_index:   Specify lattice parameter/peak to display.   
        # axis:       Plot strain against the 'x', 'y' or 'scalar' (default)
                      position. 'scalar' co-ordinates re-zeroed/centred against 
                      point specified in the extract_line command.
        """
        A = self.extract_line()        
        plt.plot(A[0], A[2][..., 0, 0], '-*')



    def plot_detector(self, detector = 0, q_idx = 0, cmap = 'RdBu_r', res = 10, 
                 lvls = 11, figsize = (10, 10), plotting = plot_complex, 
                 line = False, line_props = 'w--', mark = None):
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
        assert len(self.dims) == 2, 'plotting method only compatible with \
                                     2d data sets'
                                     
        d1, d2 = [self.co_ords[x] for x in self.dims]           
        
        if self.strain[..., 0, 0].ndim != 2:
            # Data point spacing based on resolution
            d1_points = np.ceil(res * (np.max(d1) - np.min(d1)))
            d2_points = np.ceil(res * (np.max(d2) - np.min(d2)))
            # Data point location
            d1_ = np.linspace(np.min(d1), np.max(d1), d1_points)
            d2_ = np.linspace(np.min(d2), np.max(d2), d2_points)
            D1, D2 = np.meshgrid(d1_, d2_)
            Z = griddata((d1.flatten(), d2.flatten()), 
                          self.strain[..., detector, q_idx].flatten(), (D1, D2))
        else:
            D1, D2 = d1, d2
            Z = self.strain[..., detector, q_idx]
        f, ax = plotting(d1, d2, D1, D2, Z, cmap, lvls, figsize)

############ UNFINISHED ###############

        if line == True:
            A = self.extract_line(pnt = (0, 0), angle = 0, npnts = 100, method = 'linear')     
            ax.plot(A[0], A[1], line_props, linewidth = 2)
            plt.figure()
            self.plot_line(detector, q_idx, axis = 'scalar', pnt = (0, 0), angle = 0, npnts = 100, method = 'linear')
            #if mark != None:
             #   ax.plot(self.line_centre[0], self.line_centre[1], mark)



    def plot_angle(self, angle = 0, shear = False, q_idx = 0, cmap = 'RdBu_r',  
                 res = 10, lvls = 11, figsize = (10, 10), plotting = plot_complex, 
                 line = False, line_props = 'w--', mark = None):
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
        

        assert len(self.dims) == 2, 'plotting method only compatible with \
                                     2d data sets'
                                     
        d1, d2 = [self.co_ords[x] for x in self.dims]   
        strain_field = np.nan * self.strain[..., 0, 0]
        for idx in np.ndindex(strain_field.shape):
            p = self.strain_param[idx][0]
            if shear == True:
                e_1, e_2 = (p[2] + p[0]), (p[2] - p[0])    
                theta = p[1]
                tau_xy = -np.sin(2 * theta + angle) * (e_1 - e_2)/2
                strain_field[idx] = tau_xy
            else:
                strain_field[idx] = cos_(angle, *p)
        
        # Testing whether it is merged (and therefore flattened) data
        if self.strain[..., 0, 0].ndim != 2:
            # Data point spacing based on resolution
            d1_points = np.ceil(res * (np.max(d1) - np.min(d1)))
            d2_points = np.ceil(res * (np.max(d2) - np.min(d2)))
            # Data point location
            d1_ = np.linspace(np.min(d1), np.max(d1), d1_points)
            d2_ = np.linspace(np.min(d2), np.max(d2), d2_points)
            D1, D2 = np.meshgrid(d1_, d2_)
            Z = griddata((d1.flatten(), d2.flatten()), 
                         strain_field.flatten(), (D1, D2))
        else:
            D1, D2 = d1, d2
            Z = strain_field
            
        f, ax = plotting(d1, d2, D1, D2, Z, cmap, lvls, figsize)



