# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casim
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata, interp1d

from edi12.plotting import plot_complex, line_extract


class XRD_plotting():
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
    def __init__(self):
        pass

        
    def plot_intensity(self, detector = 0, point = None):
        """
        Plots q v intensity. *Not implemented for merged files.*
        
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty. 
        # point:      Define point (index) from which to extract q v I plot.
                      First point in array chosen if None (default) specified.
        """
        try:
            group = self.f['entry1']['EDXD_elements']
            q = group['edxd_q'][detector]
            if self.ss2_z == None:            
                I = group['data'][0, 0, detector]
                
            else:
                I = group['data'][0, 0, 0, detector]
                
            plt.plot(q, I)
        except (NameError, AttributeError):
            print("Can't plot spectrum on merged data.")
            
    def plot_fitted(self, point = (0), q_idx = 0, figsize = (7, 5)):
        plt.figure(figsize = figsize)
        p = self.strain_param[point][q_idx]
        theta = np.linspace(0, np.pi, 23)
        plt.plot(theta, self.strain[point][..., q_idx][:-1], 'k*')
        theta_2 = np.linspace(0, np.pi, 1000)
        plt.plot(theta_2, cos_(theta_2, *p), 'k-')
        plt.xlabel('Detector angle')
        plt.ylabel('Strain')
            
            
    def plot_mohrs(self, point = (0), q_idx = 0, angle = 0, figsize = (7, 5)):
        """
        Mohrs circle for each point.
        """
        p = self.strain_param[point][q_idx]
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


    def plot_line(self, detector = 0, q0_index = 0, axis = 'scalar'):
        """
        Plots a line profile through a 2D strain field - extract_line method
        must be run first. *Not yet implemented in 3D.*
                
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty.       
        # q0_index:   Specify lattice parameter/peak to display.   
        # axis:       Plot strain against the 'x', 'y' or 'scalar' (default)
                      position. 'scalar' co-ordinates re-zeroed/centred against 
                      point specified in the extract_line command.
        """
        try:
            if axis == 'x':
                position = self.x_ext
            elif axis == 'y':
                position = self.y_ext
            else:
                position = self.scalar_ext
            plt.plot(position, self.strain_ext[:, detector, q0_index])
        except NameError:
            print('Line profiles have not been extracted. '
                  'Run extract_line method.')


    def plot_map(self, detector = 0, q_idx = 0, cmap = 'RdBu_r', res = 10, 
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
        if self.ss2_x.ndim != 2:
            x_points = np.ceil(res * (np.max(self.ss2_x) - np.min(self.ss2_x)))
            y_points = np.ceil(res * (np.max(self.ss2_y) - np.min(self.ss2_y)))
            x = np.linspace(np.min(self.ss2_x), np.max(self.ss2_x), x_points)
            y = np.linspace(np.min(self.ss2_y), np.max(self.ss2_y), y_points)
            X, Y = np.meshgrid(x, y)
            Z = griddata((self.ss2_x.flatten(), self.ss2_y.flatten()), 
                          self.strain[..., detector, q_idx].flatten(), (X, Y))
        else:
            X, Y = self.ss2_x, self.ss2_y
            x, y = X, Y
            Z = self.strain[..., detector, q_idx]
        f, ax = plotting(self.ss2_x, self.ss2_y, X, Y, Z, cmap, lvls, figsize)
        if line == True:
            try:            
                ax.plot(self.x_ext, self.y_ext, line_props, linewidth = 2)
                if mark != None:
                    ax.plot(self.line_centre[0], self.line_centre[1], mark)
            except AttributeError:
                print('Run line_extract method before plotting line.')


    def plot_angle(self, angle = 0, detector = 0, q_idx = 0, cmap = 'RdBu_r',  
                 res = 10, lvls = 11, figsize = (10, 10), plotting = plot_complex, 
                 line = False, line_props = 'w--', mark = None):
        strain_field = np.nan * self.ss2_x
        for idx in np.ndindex(strain_field.shape):
            p = self.strain_param[idx][0]
            strain_field[idx] = cos_(angle, *p)
            
        if self.ss2_x.ndim != 2:
            x_points = np.ceil(res * (np.max(self.ss2_x) - np.min(self.ss2_x)))
            y_points = np.ceil(res * (np.max(self.ss2_y) - np.min(self.ss2_y)))
            x = np.linspace(np.min(self.ss2_x), np.max(self.ss2_x), x_points)
            y = np.linspace(np.min(self.ss2_y), np.max(self.ss2_y), y_points)
            X, Y = np.meshgrid(x, y)
            Z = griddata((self.ss2_x.flatten(), self.ss2_y.flatten()), 
                          strain_field.flatten(), (X, Y))
        else:
            X, Y = self.ss2_x, self.ss2_y
            Z = strain_field
            
        #print(np.shape(x), np.shape(X), np.shape(y), np.shape(Y), np.shape(Z))
        f, ax = plotting(self.ss2_x, self.ss2_y, X, Y, Z, cmap, lvls, figsize)


    def plot_shear(self, angle = 0, q_idx = 0, cmap = 'RdBu_r',  res = 10, 
                   lvls = 11, figsize = (10, 10), plotting = plot_complex, 
                   line = False, line_props = 'w--', mark = None):
        
        strain_field = np.nan * self.ss2_x
        
        for idx in np.ndindex(strain_field.shape):
            p = self.strain_param[idx][0]

            e_1, e_2 = (p[2] + p[0]), (p[2] - p[0])              
                    
            theta = p[1]
            tau_xy = -np.sin(2 * theta + angle) * (e_1 - e_2)/2
            strain_field[idx] = tau_xy
            
        if self.ss2_x.ndim != 2:
            x_points = np.ceil(res * (np.max(self.ss2_x) - np.min(self.ss2_x)))
            y_points = np.ceil(res * (np.max(self.ss2_y) - np.min(self.ss2_y)))
            x = np.linspace(np.min(self.ss2_x), np.max(self.ss2_x), x_points)
            y = np.linspace(np.min(self.ss2_y), np.max(self.ss2_y), y_points)
            X, Y = np.meshgrid(x, y)
            Z = griddata((self.ss2_x.flatten(), self.ss2_y.flatten()), 
                          strain_field.flatten(), (X, Y))
        else:
            X, Y = self.ss2_x, self.ss2_y
            Z = strain_field
            
        #print(np.shape(x), np.shape(X), np.shape(y), np.shape(Y), np.shape(Z))
        f, ax = plotting(self.ss2_x, self.ss2_y, X, Y, Z, cmap, lvls, figsize)

