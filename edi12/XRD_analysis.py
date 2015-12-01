# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casim
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py

import matplotlib.pyplot as plt 
from scipy.interpolate import griddata, interp1d
from edi12.fitting_optimization import *
from edi12.peak_fitting import *
from edi12.merge_tools import *
import shutil
from edi12.plotting import plot_complex, line_extract
from edi12.peak_fitting import cos_

class XRD_scrape():
    """
    Basic scraping constructor class for XRD tools and analysis.
    """
    def __init__(self, file):
        self.filename = file
        self.f = h5py.File(file, 'r') 
        group = self.f['entry1']['EDXD_elements']           
        
        try:
            self.slit_size = self.f['entry1/before_scan/s4/s4_xs'][0]
        except KeyError:
            pass
        self.ss2_x = group['ss2_x'][:]
        self.ss2_y = group['ss2_y'][:]
        try:
            self.ss2_z = group['ss2_z'][:]
        except KeyError:
            self.ss2_z = None


class XRD_tools(XRD_scrape):
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
    def __init__(self, file):
        super(XRD_tools, self).__init__(file)
        group = self.f['entry1']['EDXD_elements']
        
        try:        
            self.q0 = group['q0'][:]
            self.peak_windows = group['peak_windows'][:]
            self.peaks = group['peaks'][:]
            self.peaks_err = group['peaks_err'][:]
            self.strain = group['strain'][:]
            self.strain_err = group['strain_err'][:]
            self.strain_param = group['strain_param'][:]
        except KeyError:
            print('Invalid .nxs file - try XRD_analysis tool.')

        
    def __enter__(self):
        return self

        
    def recentre(self, centre):
        """
        Shifts centre point to user defined location. Not reflected in .nxs
        file unless saved. Accept offset for both 2D and 3D data sets (x, y,z).
        """
        co_ords = [self.ss2_x, self.ss2_y, self.ss2_z]
        
        for dimension, offset in enumerate(co_ords):
            co_ords[dimension] += offset
            
            
    def extract_line(self, pnt = (0, 0), angle = 0, npnts = 100, 
                     method = 'linear'):
        """
        Extracts line profile through 2D strain field.
        
        # pnt:        Centre point for data extraction  
        # angle:      Angle at which to extract data
        # npnts:      Number of points (default = 100) to extract along line
        # method:     Interpolation mehod (default = 'linear')
        """
        x_ext, y_ext = line_extract(self.ss2_x, self.ss2_y, pnt, angle, npnts)
        data_shape = self.strain.shape
        
        strain_ext = np.nan * np.ones(((len(x_ext),) + data_shape[-2:]))
        for detector, q_idx in np.ndindex(self.strain.shape[-2:]):
            not_nan = ~np.isnan(self.strain[..., detector, q_idx])
            try:
                strain_line = griddata((self.ss2_x[not_nan], self.ss2_y[not_nan]), 
                                   self.strain[..., detector, q_idx][not_nan], 
                                   (x_ext, y_ext), method = method)
            except ValueError:
                pass
            strain_ext[:, detector, q_idx] = strain_line
        x_min, y_min = np.min(x_ext), np.min(y_ext)
        zero = ((pnt[0] - x_min)**2 + (pnt[1] - y_min)**2)**0.5
        
        self.scalar_ext = ((x_ext - x_min)**2 + (y_ext - y_min)**2)**0.5 - zero
        self.x_ext = x_ext 
        self.y_ext = y_ext
        self.strain_ext = strain_ext  
        self.line_centre = pnt
                
        return x_ext, y_ext, strain_ext


    def strain_angle(self, angle = 45, method = 'linear'):
        """
        Extracts strain field at angle (default = 45)

        # method:     Interpolation mehod (default = 'linear')
        """
        sin_angles = np.sin(np.linspace(0, np.pi, 23))    
        
        interp_data = np.nan * np.ones(self.strain[..., 0, :].shape)
        for position in np.ndindex(self.strain[..., 0, :].shape):
            strain_field = self.strain[position[:-1]][..., :-1, position[-1]]
            f = interp1d(sin_angles, strain_field, method)
            interp_data[position] = f(np.sin(angle))
        
        self.angle = angle
        self.strain_theta = interp_data

        
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

        
        
        
        
        
        #plt.show()
        #the = np.arctan(2 * tau_xy/ (e_xx - e_yy))/2
        #eva = 0.5 * (e_xx + e_yy) - (0.5 *(e_xx - e_yy) * np.cos(2 * the) + tau_xy * np.sin(2 * the))
        #print(p[1])#, e_xx, e_yy)
        #print(0.5 * (e_xx + e_yy), 0.5 *(e_xx - e_yy) * np.sin(2 * the), tau_xy * np.cos(2 * the))


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
            
        print(np.shape(x), np.shape(X), np.shape(y), np.shape(Y), np.shape(Z))
        f, ax = plotting(self.ss2_x, self.ss2_y, X, Y, Z, cmap, lvls, figsize)



                     
    def strain_to_text(self, fname, q0_index = [0], detectors = [0, 11], 
                       str_theta = False):
        """
        Saves key strain to text file. Not yet implemented in 3D.
        
        # fname:      File name/location
        # q0_index:   Specify lattice parameter/peak to save data from. 
        # detectors:  Define detectors from which to strain strain. 0 based 
                      indexing - 0 (default) to 23 - detector 23 empty.
        # str_theta:  Option to save strain data extracted at angle (default = 
                      False). Must run strain_theta method first.
        """                
        for q in q0_index:
            data_array = (self.ss2_x.flatten(), self.ss2_y.flatten())
            try:
                data_array += (self.ss2_z.flatten(), )
            except AttributeError:
                pass
            for detector in detectors:
                data_array += (self.strain[..., detector, q].flatten(), )
            if strain_theta == True:
                try:
                    data_array += (self.strain_theta.flatten(), )
                except AttributeError:
                    print('Strain profiles have not been extracted at angle.' 
                          ' Run strain_angle method.')
            np.savetxt(fname, np.vstack(data_array).T)
            
    
    def strain_to_text2(self, fname, q0_index = 0, angles = [0, np.pi/2], 
                       e_xy = True):
        """
        Saves key strain to text file. Not yet implemented in 3D.
        
        # fname:      File name/location
        # q0_index:   Specify lattice parameter/peak to save data from. 
        # detectors:  Define detectors from which to strain strain. 0 based 
                      indexing - 0 (default) to 23 - detector 23 empty.
        # str_theta:  Option to save strain data extracted at angle (default = 
                      False). Must run strain_theta method first.
        """                

        data_array = (self.ss2_x.flatten(), self.ss2_y.flatten())
        try:
            data_array += (self.ss2_z.flatten(), )
        except AttributeError:
            pass
        for angle in angles:
            strain_field = np.nan * self.ss2_x
        
            for idx in np.ndindex(strain_field.shape):
                p = self.strain_param[idx][0]
                strain_field[idx] = cos_(angle, *p)
            
            data_array += (strain_field.flatten(), )
        
        if e_xy == True:
            strain_field = np.nan * self.ss2_x
        
            for idx in np.ndindex(strain_field.shape):
                p = self.strain_param[idx][q0_index]
    
                e_1, e_2 = (p[2] + p[0]), (p[2] - p[0])              
                        
                theta = p[1] + angles[0]
                tau_xy = -np.sin(2 * theta ) * (e_1 - e_2)/2
                strain_field[idx] = tau_xy
            
            data_array += (strain_field.flatten(), )
        np.savetxt(fname, np.vstack(data_array).T)
            
    
    def save_to_nxs(self, fname):
        """
        Saves all data back into an expanded .nxs file. Contains all original 
        data plus q0, peak locations and strain.
        
        # fname:      File name/location
        
        ** Potentially needs revising - only useful for merged data **
        """

        with h5py.File(fname, 'w') as f:
            data_ids = ('peaks', 'peaks_err', 'strain', 'strain_err', 
                        'strain_param', 'ss2_x', 'ss2_y', 'q0', 'peak_windows',
                        'theta', 'strain_theta')
            data_array = (self.peaks, self.peaks_err, self.strain, 
                          self.strain_err, self.strain_param, self.ss2_x,
                           self.ss2_y, self.q0, self.peak_windows)
                          
            if self.ss2_z != None:
                data_ids += ('ss2_z', )
                data_array += (self.ss2_z, )
                
            for data_id, data in zip(data_ids, data_array):     
                base_tree = 'entry1/EDXD_elements/%s'
                f.create_dataset(base_tree % data_id, data = data)   
                
    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()
        

class XRD_analysis(XRD_tools):
    """
    Takes an un-processed .nxs file from the I12 EDXD detector and fits curves
    to all specified peaks for each detector. Calculates strain and details
    associated error. 
    """
   
    def __init__(self, file, q0, window, func = gaussian):
        """
        Extract and manipulate all pertinent data from the .nxs file. Takes 
        either one or multiple (list) q0s.
        """
        super(XRD_tools, self).__init__(file)
        group = self.f['entry1']['EDXD_elements']
        q, I = group['edxd_q'], group['data']
        
        # Convert int or float to list
        if type(q0) == int or type(q0) == float or type(q0) == np.float64:
            q0 = [q0]
        self.q0 = q0
        self.peak_windows = [[q - window/2, q + window/2] for q in q0]
        
        # Accept detector specific q0 2d-array
        if len(np.shape(q0)) == 2:
            q0_av = np.nanmean(q0, 0)
            self.peak_windows = [[q - window/2, q + window/2] for q in q0_av]
 
        # Iterate across q0 values and fit peaks for all detectors
        array_shape = I.shape[:-1] + (np.shape(q0)[-1],)
        self.peaks = np.nan * np.ones(array_shape)
        self.peaks_err = np.nan * np.ones(array_shape)
        for idx, window in enumerate(self.peak_windows):
            fit_data = array_fit(q, I, window, func)
            self.peaks[..., idx], self.peaks_err[..., idx] = fit_data
        self.strain = (self.q0 - self.peaks)/ self.q0
        self.strain_err = (self.q0 - self.peaks_err)/ self.q0
        
        self.strain_theta = None
        self.theta = None
        self.strain_fit()

    def strain_fit(self):
        """
        Fits a sin function to the 
        """
        data_shape = self.strain.shape
        self.strain_param = np.nan * np.ones(data_shape[:-2] + \
                            (data_shape[-1], ) + (3, ))
        for idx in np.ndindex(data_shape[:-2] + (data_shape[-1],)):
            data = self.strain[idx[:-1]][:-1][..., idx[-1]]
            not_nan = ~np.isnan(data)
            angle = np.linspace(0, np.pi, 23)
            p0 = [np.nanmean(data), 3*np.nanstd(data)/(2**0.5), 0]
            try:
                a, b = curve_fit(cos_, angle[not_nan], data[not_nan], p0)
                self.strain_param[idx] = a
            except (TypeError, RuntimeError):
                print('bab')

    def save_to_nxs(self, fname = None):
        """
        Saves all data back into an expanded .nxs file. Contains all original 
        data plus q0, peak locations and strain.
        
        # fname:      File name/location - default is to save to parent 
                      directory (*_md.nxs) 
        """
        if fname == None:        
            new_file = '%s_md.nxs' % self.filename[:-4]
        else:
            new_file = fname
        
        shutil.copy(self.filename, new_file)
        with h5py.File(new_file, 'r+') as f:
            data_ids = ('q0','peak_windows', 'peaks', 'peaks_err', 
                        'strain', 'strain_err', 'strain_param')
            data_array = (self.q0, self.peak_windows, self.peaks, 
                          self.peaks_err, self.strain, self.strain_err,
                          self.strain_param)
            
            for data_id, data in zip(data_ids, data_array):
                base_tree = 'entry1/EDXD_elements/%s'
                f.create_dataset(base_tree % data_id, data = data)

        
class XRD_merge(XRD_tools):
    """
    Tool to merge mutliple XRD data sets - inherits tools for XRD_tools.
    """
    def __init__(self, data, name, order = 'slit', padding = 0.1):
        """
        Merge data, specifying mering method/order
        
        # data:       Tuple or list containing data objects analysed with the 
                      XRD_analysis tool.
        # name:       Experiment name/ID.
        # order:      Merging method/order. Specify 'simple' merge (keeps all 
                      data) or by 'slit' size (default)/ user defined order.
                      Slit/user defined order allows for the supression/removal
                      of overlapping data. User defined should be a list of
                      the same length as the data tuple.
        """
        self.data = np.array(data)
        self.q0 = self.data[0].q0
        self.peak_windows = self.data[0].peak_windows
        
        
        if order == 'slit':
            priority = [data.slit_size for data in self.data]
        elif order == 'simple':
            priority = [0 for data in self.data]
        else:
            priority = order

        print(priority)
        priority_set, inds = np.unique(priority, return_inverse=True)    
        data_mask = [self.data[inds == 0],  [None] * len(self.data[inds == 0])]
        
        for idx, _ in enumerate(priority_set[1:]):
            idx += 1
            generate_mask_from = self.data[inds < idx]
            data_for_masking = self.data[inds == idx]
            
            x_lim = find_limits([i.ss2_x for i in generate_mask_from])
            y_lim = find_limits([i.ss2_y for i in generate_mask_from])
            
            if generate_mask_from[0].ss2_z == None:
                limits = [x_lim, y_lim]
            else:
                z_lim = find_limits([i.ss2_z for i in generate_mask_from])
                limits = [x_lim, y_lim, z_lim]

            data_mask[0] = np.append(data_mask[0], data_for_masking)
            data_mask[1] += [mask_generator(data, limits, padding) 
                             for data in data_for_masking]

        self.strain, self.strain_err, self.peaks, self.peaks_err, self.ss2_x, \
        self.ss2_y, self.ss2_z = masked_merge(data_mask[0], data_mask[1])
        self.strain_fit()
        
    def strain_fit(self):
        """
        Fits a sin function to the 
        """
        data_shape = self.strain.shape
        self.strain_param = np.nan * np.ones(data_shape[:-2] + \
                            (data_shape[-1], ) + (3, ))
        for idx in np.ndindex(data_shape[:-2] + (data_shape[-1],)):
            data = self.strain[idx[:-1]][:-1][..., idx[-1]]
            not_nan = ~np.isnan(data)
            angle = np.linspace(0, np.pi, 23)
            p0 = [np.nanmean(data), 3*np.nanstd(data)/(2**0.5), 0]
            try:
                a, b = curve_fit(cos_, angle[not_nan], data[not_nan], p0)
                self.strain_param[idx] = a
            except (TypeError, RuntimeError):
                print('bab')
        
    def __exit__(self, exc_type, exc_value, traceback):
        pass
 