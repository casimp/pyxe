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
from pyxe.plotting_tools import line_ext, az90
from pyxe.fitting_functions import cos_


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

            
    def reverse(self, rev_ind = []):
        """
        Reverses specified dimensions (i.e. negative transform). 
        Not reflected in .nxs file unless. Accept reversal for both 2D and 
        3D data sets (x, y, z).
        Reversal completed in the order in which data was acquired.
        
        # rev_ind:    List of dimensions to reverse
        """
        reverse = [rev_ind] if isinstance(rev_ind, int) else rev_ind
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
        
           

    def extract_slice(self, phi = 0, detector = [], q_idx = 0,
                          stress = False, shear = False):
            """
            Extract slice of strain or stress.
            """                  
            if detector != []:
                error = "Can't calculate shear from single detector/cake slice"
                assert shear == False, error
                e_xx = self.strain[..., detector, q_idx]
                if stress:
                    e_yy = self.strain[..., az90(self.phi, detector), q_idx]
            else:
                angles = [phi, phi + np.pi/2, phi]
                strains = [np.nan * self.strain[..., 0, 0] for i in range(3)]
                for e_idx, (angle, strain) in enumerate(zip(angles, strains)):
                    for idx in np.ndindex(strain.shape):
                        p = self.strain_param[idx][0]
                        if e_idx == 2:
                            strain[idx] = -np.sin(2 * (p[1] + angle) ) * p[0]
                        else:
                            strain[idx] = cos_(angle, *p)
                e_xx, e_yy, e_xy = strains
            if stress:
                data = self.sig_eqn(e_xx, e_yy) if not shear else e_xy * self.G
            else:
                data = e_xx if not shear else e_xy
            
            return data      
    
    def extract_line(self, phi = 0, detector = [], q_idx = 0,  pnt = (0,0),
                     line_angle = 0, npnts = 100, method = 'linear', 
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
        print(data.shape)
        return line_ext(positions, data, pnt, npnts, line_angle, method)
    
    
    def save_to_text(self, fname, angles = [0, np.pi/2], detectors = [],
                     q_idx = 0, strain = True, shear = True, stress = False):
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
            
        order = strain, strain and shear, stress, stress and shear
        stress = False, False, True, True
        shear = False, True, False, True
        
        if detectors != []:
            for detector in detectors:
                for i, xx, xy  in zip(order, stress, shear):
                    if i:
                        data = self.extract_slice(detector = detector, 
                                    q_idx = q_idx, stress = xx, shear = xy)
                        data_array += (data.flatten(), )
        else:
            for phi in angles:
                for i, xx, xy  in zip(order, stress, shear):
                    if i:
                        data = self.extract_slice(phi, q_idx = q_idx, 
                                                  stress = xx, shear = xy)
                        data_array += (data.flatten(), )

        np.savetxt(fname, np.vstack(data_array).T, delimiter=',')
                         
                
    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()
        
