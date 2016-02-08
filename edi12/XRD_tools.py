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
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata, interp1d

from edi12.peak_fitting import *
from edi12.plotting import plot_complex, line_extract
from edi12.peak_fitting import cos_
from edi12.XRD_scrape import XRD_scrape
from edi12.XRD_plotting import XRD_plotting

### temporary

def coord_arrange(dims, data):
    
    co_ords = []
    for dim in dims:    
        if dim == 'ss2_x':
            co_ords += data[0]
        if dim == 'ss2_y':
            co_ords += data[1]
        if dim == 'ss2_z':
            co_ords += data[2]
            
    return co_ords

class XRD_tools(XRD_scrape, XRD_plotting):
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
    def __init__(self, file):
        super(XRD_tools, self).__init__(file)
        #super(XRD_plotting, self).__init__()
        group = self.f['entry1']['EDXD_elements']
        try:        
            self.strain = group['strain'][:]
        except KeyError as e:
            e.args += ('Invalid .nxs file - no strain data found.',
                       'Run XRD_analysis tool.')
            raise
        self.dims = group['dims'][:]
        #### Temporary
        self.co_ords = coord_arrange(self.dims, [self.ss2_x, self.ss2_y, self.ss2_z])
        self.q0 = group['q0'][:]
        self.peak_windows = group['peak_windows'][:]
        self.peaks = group['peaks'][:]
        self.peaks_err = group['peaks_err'][:]    
        self.strain_err = group['strain_err'][:]
        self.strain_param = group['strain_param'][:]

        
        
    def __enter__(self):
        return self

        
    def recentre(self, centre):
        """
        Shifts centre point to user defined location. Not reflected in .nxs
        file unless saved. Accept offset for both 2D and 3D data sets (x, y,z).
        """
        pass # this won't work        
        #co_ords = [self.ss2_x, self.ss2_y, self.ss2_z]
        
        #for dimension, offset in enumerate(co_ords):
        #    co_ords[dimension] += offset
            
            
    def extract_line(self, pnt = (0, 0), angle = 0, npnts = 100, 
                     method = 'linear'):
        """
        Extracts line profile through 2D strain field.
        
        # pnt:        Centre point for data extraction  
        # angle:      Angle at which to extract data
        # npnts:      Number of points (default = 100) to extract along line
        # method:     Interpolation mehod (default = 'linear')
        """

        assert len(self.dims) == 2, 'extract_line method only compatible with \
                                     2d data sets'

        dim_1, dim_2 = self.co_ords
        dim_1_ext, dim_2_ext = line_extract(dim_1, dim_2, pnt, angle, npnts)
        data_shape = self.strain.shape
        strain_ext = np.nan * np.ones(((len(dim_1),) + data_shape[-2:]))
        
        for detector, q_idx in np.ndindex(self.strain.shape[-2:]):
            not_nan = ~np.isnan(self.strain[..., detector, q_idx])
            try:
                strain_line = griddata((self.dim_1[not_nan], self.dim_2[not_nan]), 
                                   self.strain[..., detector, q_idx][not_nan], 
                                   (dim_1_ext, dim_2_ext), method = method)
            except ValueError:
                pass
            strain_ext[:, detector, q_idx] = strain_line
        dim_1_min, dim_2_min = np.min(dim_1_ext), np.min(dim_2_ext)
        zero = ((pnt[0] - dim_1_min)**2 + (pnt[1] - dim_2_min)**2)**0.5
        
        self.scalar_ext = ((dim_1_ext - dim_1_min)**2 + 
                           (dim_2_ext - dim_2_min)**2)**0.5 - zero
        self.dim_1_ext = dim_1_ext 
        self.dim_2_ext = dim_2_ext
        self.strain_ext = strain_ext  
        self.line_centre = pnt
                
        return dim_1_ext, dim_2_ext, strain_ext
        


                             
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
            data_ids = ('dims', 'slit_size', 'peaks', 'peaks_err', 'strain', 
                        'strain_err', 'strain_param', 'ss2_x', 'ss2_y',  
                        'ss2_z', 'q0', 'peak_windows', 'theta', 'strain_theta')
            data_array = (self.dims, self.slit_size, self.peaks, 
                          self.peaks_err, self.strain, self.strain_err, 
                          self.strain_param, self.ss2_x, self.ss2_y, 
                          self.ss2_z, self.q0, self.peak_windows)
                
            for data_id, data in zip(data_ids, data_array):     
                base_tree = 'entry1/EDXD_elements/%s'
                f.create_dataset(base_tree % data_id, data = data)   
                
    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()
        
