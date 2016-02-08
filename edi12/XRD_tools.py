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
from scipy.interpolate import griddata

from edi12.peak_fitting import *
from edi12.plotting import line_extract
from edi12.peak_fitting import cos_
from edi12.XRD_scrape import XRD_scrape
from edi12.XRD_plotting import XRD_plotting


class XRD_tools(XRD_scrape, XRD_plotting):
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
    def __init__(self, file):
        super(XRD_tools, self).__init__(file)
        group = self.f['entry1']['EDXD_elements']
        try:        
            self.strain = group['strain'][:]
        except KeyError as e:
            e.args += ('Invalid .nxs file - no strain data found.',
                       'Run XRD_analysis tool.')
            raise
        self.dims = group['dims'][:]

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
        file unless saved.Accept offset for both 2D and 3D data sets (x, y, z).
        """
        pass
        #co_ords = [self.ss2_x, self.ss2_y, self.ss2_z]
        
        #for co_ord, offset in zip(co_ords):
        #    co_ord += offset
            
            
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

        dim_1, dim_2 = [self.co_ords[x] for x in self.dims]        
        dim_1_ext, dim_2_ext = line_extract(dim_1, dim_2, pnt, angle, npnts)
        data_shape = self.strain.shape
        strain_ext = np.nan * np.ones(((len(dim_1_ext),) + data_shape[-2:]))
        print(strain_ext)
        for detector, q_idx in np.ndindex(self.strain.shape[-2:]):
            not_nan = ~np.isnan(self.strain[..., detector, q_idx])
            try:
                strain_line = griddata((dim_1[not_nan], dim_2[not_nan]), 
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
            
    
    def strain_to_text(self, fname, q0_index = 0, angles = [0, np.pi/2], 
                       e_xy = [0], detectors = None):
        """
        Saves key strain to text file. Not yet implemented in 3D.
        
        # fname:      File name/location to save data to.
        # q0_index:   Specify lattice parameter/peak to save data from. 
        # angles:     Define angles (in rad) from which to calculate strain. 
                      Default - [0, pi/2].
        # detectors:  Define detectors to take strain from (rather than 
                      calculating from full detector array).
        # e_xy:       Option to save shear strain data extracted at angles 
                      (default = [0]).
        """                

        data_array = (self.ss2_x.flatten(), self.ss2_y.flatten())
        try:
            data_array += (self.ss2_z.flatten(), )
        except AttributeError:
            pass
        
        if detectors == None:
            for angle in angles:
                strain_field = np.nan * self.ss2_x
            
                for idx in np.ndindex(strain_field.shape):
                    p = self.strain_param[idx][0]
                    strain_field[idx] = cos_(angle, *p)
                    data_array += (strain_field.flatten(), )
        else:
            for detector in detectors:
                data_array += (self.strain[..., detector, q].flatten(), )
        
        if e_xy != False:
            
            for angle in e_xy:
                strain_field = np.nan * self.ss2_x
        
                for idx in np.ndindex(strain_field.shape):
                    p = self.strain_param[idx][q0_index]
        
                    e_1, e_2 = (p[2] + p[0]), (p[2] - p[0])              
                            
                    theta = p[1] + angle
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
        
