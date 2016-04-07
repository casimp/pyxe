# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:39:33 2016

@author: casim
"""
import numpy as np

def line_dec(func):
    positions, data, pnt, line_angle, method = func()
    
    not_nan = ~np.isnan(data)
    if len(self.dims) == 2:
        try:
            data = griddata((d1[not_nan], d2[not_nan]), data[not_nan], 
                                  (d1_e, d2_e), method = method)
    
    if len(self.dims) == 1:
        d1 = self.co_ords[self.dims[0]]
        data_ext = np.nan * np.ones((len(d1), len(az_angles)))

    else:
        d1, d2 = [self.co_ords[x] for x in self.dims]        
        d1_e, d2_e = line_extract(d1, d2, pnt, line_angle, npts)
        data_ext = np.nan * np.ones((len(d1_e), len(az_angles)))

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

def az90(phi, idx):
    
    find_ind = np.isclose(phi, phi[idx] + np.pi/2)]
    for i in [-np.pi/2, np.pi/2]:
        if phi[idx] < -np.pi:
            find_ind = np.isclose(phi, np.pi - phi[idx] + i)]
        else:
            find_ind = np.isclose(phi, phi[idx] + i)]
        if np.sum(find_ind) == 1:
            return ind
    raise ValueError('No cake segment found perpendicular to given index. 
                     'Number of cake segments must be divisable by 4.')
    
class Classy(object):
    
    def __init__(self):
        pass
    
    def extract_slice(self, phi = 0, detector = [], 
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
            angles = [phi, phi + np.pi/2]
            e_xx = np.nan * self.strain[..., 0, 0]
            e_yy = np.nan * self.strain[..., 0, 0]
            for angle, strain_field in zip(angles, (e_xx, e_yy)):
                for idx in np.ndindex(strain_field.shape):
                    p = self.strain_param[idx][0]
                    strain_field[idx] = cos_(angle, *p)
        
        if stress:
            data = self.sig_eqn(e_xx, e_yy) if not shear else e_xy * self.G
        else:
            data = e_xx if not shear else e_xy
        
        return data
        
    @line_dec
    def extract_line(self, phi = 0, detector = None, q_idx = 0,  pnt = (0,0),
                     line_angle = 0, npts = 100, method = 'linear', 
                     stress = False, shear = False):
        """
        Extracts line profile through 2D strain field.
        
        # phi:        Define angle (in rad) from which to calculate strain. 
                      Default - 0.
        # detector:   Define detector to see strain/stress from cake not 
                      full ring.
        # q_idx:      Specify lattice parameter/peak to save data from. 
        # line_angle: Angle across array to extract strain from
        # pnt:        Centre point for data extraction  
        # npts:       Number of points (default = 100) to extract along line
        # method:     Interpolation mehod (default = 'linear')
        # shear:      Extract shear.
        """
        error = 'Extract_line method only compatible with 1D/2D data sets.'
        assert len(self.dims) <= 2, error
        
        positions = [self.co_ords[x] for x in self.dims]
        data = self.extract_slice(phi = phi, detector = detector, 
                                  stress = stress, shear = shear)
        
        return positions, data, pnt, npnts, line_angle, method              