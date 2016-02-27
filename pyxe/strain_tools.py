# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casimp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#import h5py
import numpy as np
from scipy.interpolate import griddata

from pyxe.plotting import line_extract
from pyxe.fitting_functions import cos_


class StrainTools(object):
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
        
    def __enter__(self):
        return self

        
    def recentre(self, centre):
        """
        Shifts centre point to user defined location. Not reflected in .nxs
        file unless saved.Accept offset for both 2D and 3D data sets (x, y, z).
        Re-centring completed in the order in which data was acquired.
        """
        co_ords = [self.co_ords[x] for x in self.dims]
        
        for co_ord, offset in zip(co_ords, centre):
            co_ord += offset
        
           
    def extract_line_detector(self, detectors = [0, 11], q_idx = 0, pnt = (0,0),
                              line_angle = 0, npts = 100, method = 'linear', 
                              data_type = 'strain', shear = False, save = False, 
                              E = 200 * 10 **9, v = 0.3, G = 79 * 10 **9):
        """
        Extracts line profile through 2D strain field.
        
        # detectors:  Define detectors to take strain from (rather than 
                      calculating from full detector array).
        # q_idx:      Specify lattice parameter/peak to save data from. 
        # line_angle: Angle across array to extract strain from
        # pnt:        Centre point for data extraction  
        # npts:       Number of points (default = 100) to extract along line
        # method:     Interpolation mehod (default = 'linear')
        # save:       Save - replace False with filename to save
        """
        error = 'Extract_line method only compatible with 2d data sets.'
        assert len(self.dims) == 2, error
        
        d1, d2 = [self.co_ords[x] for x in self.dims]        
        d1_e, d2_e = line_extract(d1, d2, pnt, line_angle, npts)
        
        detectors = [detectors] if isinstance(detectors, 
                    (int, float, np.float64)) else detectors
                    
        data_ext = np.nan * np.ones((len(d1_e), len(detectors)))
        
        for idx, detector in enumerate(detectors):
            
            if data_type == 'strain':
                data = self.strain[..., detector, q_idx]
            elif data_type == 'stress':
                data = self.extract_stress(E = E, v = v, detector = detector)[0]                
            not_nan = ~np.isnan(data)
            try:
                data_line = griddata((d1[not_nan], d2[not_nan]), data[not_nan], 
                                      (d1_e, d2_e), method = method)
            except ValueError:
                pass
            data_ext[:, idx] = data_line
        
        if save != False:
            fname = save if isinstance(save, str) else self.filename[:-4] + '.txt'
            np.savetxt(fname, (d1_e, d2_e, data_ext), delimiter = ',')
        
        return d1_e, d2_e, data_ext

    def extract_line_angle(self, az_angles = [0, np.pi/2], q_idx = 0,  pnt = (0,0),
                           line_angle = 0, npts = 100, method = 'linear', 
                           data_type = 'strain', shear = False, save = False, 
                           E = 200 * 10 **9, v = 0.3, G = 79 * 10 **9):
        """
        Extracts line profile through 2D strain field.
        
        # az_angles:  Define angle (in rad) from which to calculate strain. 
                      Default - 0.
        # q_idx:      Specify lattice parameter/peak to save data from. 
        # line_angle: Angle across array to extract strain from
        # pnt:        Centre point for data extraction  
        # npts:       Number of points (default = 100) to extract along line
        # method:     Interpolation mehod (default = 'linear')
        # save:       Save - replace False with filename to save
        # e_xy:       Extract data from shear map rather than strain map.
        """
        error = 'Extract_line method only compatible with 2d data sets.'
        assert len(self.dims) == 2, error
        
        d1, d2 = [self.co_ords[x] for x in self.dims]        
        d1_e, d2_e = line_extract(d1, d2, pnt, line_angle, npts)

        data_ext = np.nan * np.ones((len(d1_e), len(az_angles)))
        
        for angle_idx, angle in enumerate(az_angles):
            
            data = np.nan * self.strain[..., 0, 0]
            for idx in np.ndindex(data.shape):
                p = self.strain_param[idx][0]
                e_xx, e_yy = cos_(angle, *p), cos_(angle + np.pi/2, *p)
                e_xy = -np.sin(2 * (p[1] + angle)) * p[0]     
                if data_type == 'strain':
                    data[idx] = e_xx if not shear else e_xy
                elif data_type == 'stress':
                    sigma_xx = E * ((1-v)*e_xx + v*e_yy) / ((1+v)*(1-2*v))
                    data[idx] = sigma_xx if not shear else e_xy * G

            not_nan = ~np.isnan(data)
            try:
                data_line = griddata((d1[not_nan], d2[not_nan]), data[not_nan], 
                                      (d1_e, d2_e), method = method)
            except ValueError:
                pass
            data_ext[:, angle_idx] = data_line
        
        if save != False:
            fname = save if isinstance(save, str) else self.filename[:-4] + '.txt'
            np.savetxt(fname, (d1_e, d2_e, data_ext), delimiter = ',')
        
        return d1_e, d2_e, data_ext                         
                         
                         
    def extract_stress(self, angle = 0, detector = [], q_idx = 0, 
                       E = 200*10**9, v = 0.3, G = 79 * 10**9, save = False):
        """
        Uses a plane strain assumption, with the strain the unmeasured plane
        approximating to zero. Incorrect when this is not the case.
        """
        if detector != []:
            det1 = detector
            det2 = det1 + 11 if (det1 + 11) < 22 else (det1 - 11)
            e_xx = self.strain[..., det1, q_idx]
            e_yy = self.strain[..., det2, q_idx]
        else:
            angles = [angle, angle + np.pi/2]
            e_xx = np.nan * self.strain[..., 0, 0]
            e_yy = np.nan * self.strain[..., 0, 0]
            for angle, strain_field in zip(angles, (e_xx, e_yy)):
                for idx in np.ndindex(strain_field.shape):
                    p = self.strain_param[idx][0]
                    strain_field[idx] = cos_(angle, *p)
        
        sigma_xx = E * ((1 - v) * e_xx + v * e_yy)/ ((1 + v) * (1 - 2 * v))
        sigma_yy = E * ((1 - v) * e_yy + v * e_xx)/ ((1 + v) * (1 - 2 * v))
            
        return sigma_xx, sigma_yy          
    
    def strain_to_text(self, fname, q_idx = 0, angles = [0, np.pi/2], 
                       e_xy = [0], detectors = []):
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
        data_array = ()
        for i in self.dims:
            data_array += (self.co_ords[i], )

        if detectors == []:
            for angle in angles:
                strain_field = np.nan * self.strain[..., 0, 0]
 
                for idx in np.ndindex(strain_field.shape):
                    p = self.strain_param[idx][0]
                    strain_field[idx] = cos_(angle, *p)
                    data_array += (strain_field.flatten(), )
        else:
            for detector in detectors:
                data_array += (self.strain[..., detector, q_idx].flatten(), )
        
        if e_xy != False:
            
            for angle in e_xy:
                strain_field = np.nan * self.strain[..., 0, 0]
        
                for idx in np.ndindex(strain_field.shape):
                    p = self.strain_param[idx][q_idx]
                    e_xy = -np.sin(2 * (p[1] + angle) ) * p[0]
                    strain_field[idx] = e_xy
                
                data_array += (strain_field.flatten(), )
                
        np.savetxt(fname, np.vstack(data_array).T)


    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()
        
