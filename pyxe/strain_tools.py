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

def az90(phi, idx):
    
    for i in [-np.pi/2, np.pi/2]:
        if phi[idx] < -np.pi:
            find_ind = np.isclose(phi, np.pi - phi[idx] + i)
        else:
            find_ind = np.isclose(phi, phi[idx] + i)
        if np.sum(find_ind) == 1:
            return np.argmax(find_ind)
    raise ValueError('No cake segment found perpendicular to given index.', 
                     'Number of cake segments must be divisable by 4.')


class StrainTools(object):
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
        
    def __enter__(self):
        return self

    def define_matprops(self, E = 200*10**9, v = 0.3, G = None, 
                        state = 'plane strain'):
        """
        Define material properties and sample stress state such that stress 
        can be calculated. Default values are for a nominal steel in a plane 
        strain stress state.
        
        # E:          Young's modulus (MPa)
        # v:          Poissons ratio
        # G:          Shear modulus - if not specified use E / (2 * (1 + v))
        # state:      Stress state assumption used for stress calculation
                      - 'plane strain' (default) or 'plane stress'.
        """
        
        self.E = E
        self.v = v
        self.G = E / (2 * (1 + v)) if G == None else G
        
        if state != 'plane strain':   
            self.sig_eqn = lambda e_xx, e_yy: (E/(1 - v**2)) * (e_xx + v*e_yy)
        else:
            self.sig_eqn = lambda e_xx, e_yy: E * ((1 - v) * e_xx + v * e_yy)/\
                                                   ((1 + v) * (1 - 2 * v))
        
    def recentre(self, centre, reverse = []):
        """
        Shifts centre point to user defined location. Not reflected in .nxs
        file unless saved.Accept offset for both 2D and 3D data sets (x, y, z).
        Re-centring completed in the order in which data was acquired.
        
        # centre:     New centre point
        # reverse:    List of dimensions to reverse
        """
        co_ords = [self.co_ords[x] for x in self.dims]
        
        for co_ord, offset in zip(co_ords, centre):
            co_ord -= offset
            
        reverse = [reverse] if isinstance(reverse, int) else reverse
        for i in reverse:
            self.co_ords[self.dims[i]] = -self.co_ords[self.dims[i]]
            
    def rotate(self, order = [1, 0]):
        """
        Switches order of axes, which has the same effect as rotating the 
        strain data. Order must be a list of a length equal to the number of 
        dimensions of the data. 
        
        # order:      New order for dimensions
        """
        self.dims = [self.dims[i] for i in order]
        
        
    def mask(self, patch, radius):
        """
        Pass in matplotlib patch with which to mask area. 
        Note that in 3D the patch is applied according to first 2 dims and
        applied through stack.
        
        UNTESTED!!!!
        
        # patch:      Matplotlib patch object
        # radius:     Extend or contract mask from object edge. 
        """
        pos = zip(*[self.co_ords[i] for i in self.dims[:2]])
        isin = [patch.contains_point((x, y), radius = radius) for x, y in pos]
        self.strain_param[np.array(isin)] = np.nan
        self.strain[np.array(isin)] = np.nan
        
           
    def extract_line_detector(self, detectors = [0, 11], q_idx = 0, pnt = (0,0),
                              line_angle = 0, npts = 100, method = 'linear', 
                              stress = False, shear = False, save = False):
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
            
            if stress:
                data = self.extract_slice(detector=detector, q_idx=q_idx, stress = True)
            else:
                data = self.strain[..., detector, q_idx]
             
            not_nan = ~np.isnan(data)
            try:
                data_line = griddata((d1[not_nan], d2[not_nan]), data[not_nan], 
                                      (d1_e, d2_e), method = method)
            except ValueError:
                pass
            data_ext[:, idx] = data_line
        
        if save:
            fname = save if isinstance(save, str) else self.filename[:-4] + '.txt'
            np.savetxt(fname, (d1_e, d2_e, data_ext), delimiter = ',')
        
        return d1_e, d2_e, data_ext

    def extract_line_angle(self, az_angles = [0, np.pi/2], q_idx = 0,  pnt = (0,0),
                           line_angle = 0, npts = 100, method = 'linear', 
                           stress = False, shear = False, save = False):
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
        error = 'Extract_line method only compatible with 1D/2D data sets.'
        assert len(self.dims) <= 2, error
        
        
        if len(self.dims) == 1:
            d1 = self.co_ords[self.dims[0]]
            data_ext = np.nan * np.ones((len(d1), len(az_angles)))

        else:
            d1, d2 = [self.co_ords[x] for x in self.dims]        
            d1_e, d2_e = line_extract(d1, d2, pnt, line_angle, npts)
            data_ext = np.nan * np.ones((len(d1_e), len(az_angles)))
            
        
        for angle_idx, angle in enumerate(az_angles):
            
            data = np.nan * self.strain[..., 0, 0]
            for idx in np.ndindex(data.shape):
                p = self.strain_param[idx][q_idx]
                e_xx, e_yy = cos_(angle, *p), cos_(angle + np.pi/2, *p)
                e_xy = -np.sin(2 * (p[1] + angle)) * p[0]    
                
                if stress:
                    sigma_xx = self.sig_eqn(e_xx, e_yy)
                    data[idx] = sigma_xx if not shear else e_xy * self.G
                else:
                    data[idx] = e_xx if not shear else e_xy
                    
            not_nan = ~np.isnan(data)
            if len(self.dims) == 2:
                try:
                    data = griddata((d1[not_nan], d2[not_nan]), data[not_nan], 
                                          (d1_e, d2_e), method = method)
                    
                except ValueError:
                    pass
            data_ext[:, angle_idx] = data
                
        if save:
            fname = save if isinstance(save, str) else self.filename[:-4] + '.txt'
            np.savetxt(fname, (d1_e, d2_e, data_ext), delimiter = ',')
        
        if len(self.dims) == 1:
            return d1[not_nan], data_ext[not_nan]
        else:
            return d1_e, d2_e, data_ext                      
                         
                         
    def extract_slice(self, phi = 0, detector = [], q_idx = 0,
                          stress = False, shear = False):
            """
    
            """                  
            if detector != []:
                error = "Can't calculate shear from single detector/cake slice"
                assert shear == False, error
                e_xx = self.strain[..., detector, q_idx]
                if stress:
                    e_yy = self.strain[..., az90(self.phi, detector), q_idx]
            else:
                angles = [phi, phi + np.pi/2, phi]
                e_xx = np.nan * self.strain[..., 0, 0]
                e_yy = np.nan * self.strain[..., 0, 0]
                e_xy = np.nan * self.strain[..., 0, 0]
                strains = (e_xx, e_yy, e_xy)
                for e_idx, (angle, strain) in enumerate(zip(angles, strains)):
                    for idx in np.ndindex(strain.shape):
                        p = self.strain_param[idx][0]
                        if e_idx == 2:
                            strain[idx] = -np.sin(2 * (p[1] + angle) ) * p[0]
                        else:
                            strain[idx] = cos_(angle, *p)
            
            if stress:
                data = self.sig_eqn(e_xx, e_yy) if not shear else e_xy * self.G
            else:
                data = e_xx if not shear else e_xy
            
            return data       
    
    
    def angle_to_text(self, fname, angles = [0, np.pi/2], q_idx = 0, 
                      strain = True, shear = True, stress = False):
        """
        Saves key strain to text file. Not yet implemented in 3D.
        
        # fname:      File name/location to save data to.
        # q_idx:      Specify lattice parameter/peak to save data from. 
        # angles:     Define angles (in rad) from which to calculate strain. 
                      Default - [0, pi/2].
        # detectors:  Define detectors to take strain from (rather than 
                      calculating from full detector array).
        # shear:      Option to save shear strain data extracted at angles 
                      (default = [0]).
        """                
        
        data_array = ()
        for i in self.dims:
            data_array += (self.co_ords[i], )
            
        data_cols = (strain + stress) * (1 + shear)
        data_shape = (data_cols,) + self.strain[...,0, 0].shape
        sig_idx = 2 if shear else 1
        
        for angle in angles:
            data = np.nan * np.ones(data_shape)
            for idx in np.ndindex(data.shape[1:]):
                p = self.strain_param[idx][q_idx]
                e_xx, e_yy = cos_(angle, *p), cos_(angle + np.pi/2, *p)  
                e_xy = -np.sin(2 * (p[1] + angle) ) * p[0]                  
                
                if strain:
                    data[0, idx] = e_xx
                    if shear:
                        data[1, idx] = e_xy
                if stress:
                    data[sig_idx, idx] = self.sig_eqn(e_xx, e_yy)
                    if shear:
                        data[sig_idx + 1, idx] = e_xy * self.G
                        
            data_array += (data.reshape(data.shape[0], data[0].size), )

        np.savetxt(fname, np.vstack(data_array).T, delimiter=',')


    def cake_to_text(self, fname, detectors = [0], q_idx = 0, 
                      strain = True, stress = False):
        """
        Saves key strain to text file. Not yet implemented in 3D.
        
        # fname:      File name/location to save data to.
        # q_idx:      Specify lattice parameter/peak to save data from. 
        # detectors:  Define detectors to take strain from (rather than 
                      calculating from full detector array).
        """                
        data_array = ()
        for i in self.dims:
            data_array += (self.co_ords[i], )

        if strain:
            for detector in detectors:
                data_array += (self.strain[..., detector, q_idx].flatten(), )
        
        if stress:
            pass
                
        np.savetxt(fname, np.vstack(data_array).T)
               
                
    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()
        
