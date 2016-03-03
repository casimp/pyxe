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
            error = 'Must define point with correct number of dimensions.'
            assert len(self.dims) == len(point), error
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
            point = (0, ) * len(self.strain[..., 0, 0].shape)

        #assert len(self.dims) == len(point), 'must define point with correct '\
        #                                     'number of dimensions.'
        
        plt.figure(figsize = figsize)
        p = self.strain_param[point][q_idx]
        theta = np.linspace(0, np.pi, 23)
        plt.plot(theta, self.strain[point][..., q_idx][:-1], 'k*')
        theta_2 = np.linspace(0, np.pi, 1000)
        plt.plot(theta_2, cos_(theta_2, *p), 'k-')
        plt.xlabel('Detector angle (rad)')
        plt.ylabel('Strain')
            
            
    def plot_mohrs(self, point = (), q_idx = 0, angle = 0, figsize = (7, 5)):
        """
        Use fitted in-plane styrain tensor to plot Mohr's circle. 
        *Not implemented for merged files.*
        
        # point:      Define point (index) from which to plot fitted in-plane
                      strain field.
        # q_idx:      0 based indexing - 0 (default) to 23 - detector 23 empty.
        # angle:      Angle to highlight on circle (inc + 90deg).
        # figsize:    Figure dimensions
        """
        if point == ():
            point = (0, ) * len(self.strain[..., 0, 0].shape)
            
        p = self.strain_param[point][q_idx] ### test change
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
        
        plt.plot(e_xx, e_xy, 'ko', label = r'$(\epsilon_{xx}$, $\epsilon_{xy})$')
        plt.plot(e_yy, -e_xy, 'wo', label = r'$(\epsilon_{yy}$, $-\epsilon_{xy})$')
        
        plt.legend(numpoints=1, frameon = False, handletextpad = 0.2)
        plt.plot([e_xx, e_yy], [e_xy, -e_xy], 'k-.')
        ax.annotate('  %s' % r'$\epsilon_{1}$', xy=(e_1, 0), 
                    textcoords='offset points', xytext=(e_1, 0))
        ax.annotate('  %s' % r'$\epsilon_{2}$', xy=(e_2, 0), 
                    textcoords='offset points', xytext=(e_2, 0))
        plt.xlabel('Normal strain')
        plt.ylabel('Shear strain')


    def plot_line(self, az_angles = [0, np.pi/2], detectors = [], q_idx = 0, 
                  line_angle = 0, pnt = (0, 0), npts = 100, axis = 'scalar', 
                  method = 'linear', data_type = 'strain', shear = False, 
                  E = 200 * 10**9, v = 0.3, G = 79 * 10 **9):
        """
        Plots a line profile through a 2D strain field - extract_line method
        must be run first. *Not yet implemented in 3D.*
                
        # detector:   0 based indexing - 0 (default) to 23 - detector 23 empty.       
        # q0_index:   Specify lattice parameter/peak to display.   
        # axis:       Plot strain against the 'd1', 'd2' or 'scalar' (default)
                      position. 'scalar' co-ordinates re-zeroed/centred against 
                      point specified in the extract_line command.
        """
        error = 'Line plotting only implemented for 1D or 2D data sets'
        assert len(self.dims) <= 2, error      
        
        if len(self.dims) == 1:
            d1 = self.co_ords[self.dims[0]]
            if detectors != []:            
                for i in detectors:
                    plt.plot(d1, self.strain[..., i, q_idx], '-*')
            else:
                d1, data = self.extract_line_angle(az_angles)
                for idx, angle in enumerate(az_angles):
                    plt.plot(d1, data[:, idx], '-*')
        
        else:
            if detectors == []:
                line_method = self.extract_line_angle
                ext = az_angles
            else:
                line_method = self.extract_line_detector
                ext = detectors
                
            d1, d2, data = line_method(ext, q_idx = q_idx, line_angle = line_angle, 
                            pnt = pnt, npts = npts, method = method, shear = shear,
                            data_type = data_type, E = E, v = v, G = G)  
                                    
            d1_min, d2_min = np.min(d1), np.min(d2)            
            zero = ((pnt[0] - d1_min)**2 + (pnt[1] - d2_min)**2)**0.5
            scalar_ext = ((d1 - d1_min)**2 + (d2 - d2_min)**2)**0.5 - zero
            
            
            x = {'d1' : d1, 'd2' : d2, 'scalar' : scalar_ext}
            plt.plot(x[axis], data, '-*')


    def plot_detector(self, detector = 0, q_idx = 0, cmap = 'RdBu_r', res = 0.1, 
                      lvls = 11, figsize = (10, 10), plotting = plot_complex, 
                      line = False, pnt = (0,0), line_angle = 0, npts = 100, 
                      method = 'linear', line_props = 'w--', mark = None, 
                      data_type = 'strain', E = 200 * 10**9, v = 0.3):
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
        
        if data_type == 'strain':
            data = self.strain[..., detector, q_idx]
        elif data_type == 'stress':
            data = self.extract_stress(E = E, v = v, detector = detector)[0]
        
        if data.ndim != 2:
            D1, D2 = meshgrid_res(d1, d2, res)
            Z = griddata((d1.flatten(), d2.flatten()),data.flatten(), (D1, D2))
        else:
            D1, D2, Z = d1, d2, data
            
        f, ax = plotting(d1, d2, D1, D2, Z, cmap, lvls, figsize)

        if line == True:
            plt.figure()
            line = line_extract(D1, D2, pnt, line_angle, npts)   
            ax.plot(line[0], line[1], line_props, linewidth = 2)
            self.plot_line(detectors = [detector], q_idx = q_idx, 
                          line_angle = line_angle, pnt = pnt, npts = npts, 
                          method = method, data_type = data_type, E = E, v = v)



    def plot_angle(self, angle = 0, shear = False, q_idx = 0, cmap = 'RdBu_r',  
              res = 0.1, lvls = 11, figsize = (10, 10), plotting = plot_complex, 
              line = False, pnt = (0,0), line_angle = 0, npts = 100, 
              method = 'linear', line_props = 'w--', mark = None, 
              data_type = 'strain', E = 200 * 10 **9, v = 0.3, G = 79 * 10**9):
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
        
        data = np.nan * self.strain[..., 0, 0]
        for idx in np.ndindex(data.shape):
            p = self.strain_param[idx][0]
            e_xx, e_yy = cos_(angle, *p), cos_(angle + np.pi/2, *p)
            e_xy = -np.sin(2 * (p[1] + angle)) * p[0]     
            if data_type == 'strain':
                data[idx] = e_xx if not shear else e_xy
            elif data_type == 'stress':
                sigma_xx = E * ((1 -v) * e_xx + v*e_yy) / ((1 + v) * (1 - 2*v))
                data[idx] = sigma_xx if not shear else e_xy * G
                    
        if data.ndim != 2:
            D1, D2 = meshgrid_res(d1, d2, spatial_resolution = res)
            Z = griddata((d1.flatten(),d2.flatten()), data.flatten(), (D1, D2))
        else:
            D1, D2, Z = d1, d2, data
            
        f, ax = plotting(d1, d2, D1, D2, Z, cmap, lvls, figsize)
        
        if line == True:
            plt.figure()
            line = line_extract(D1, D2, pnt, line_angle, npts)   
            ax.plot(line[0], line[1], line_props, linewidth = 2)
            self.plot_line(az_angles = [angle], q_idx = q_idx, shear = shear,
                          line_angle = line_angle, pnt = pnt, npts = npts, 
                          method = method, data_type = data_type, E = E, v = v, G = G)



