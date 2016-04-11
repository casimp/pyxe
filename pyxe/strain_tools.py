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
    Takes post-processed .nxs file and allows for further analysis/vizulisation. 
    File should have been created with the EDI12 or Area analysis tools.
    """
    def __enter__(self):
        return self


    def define_matprops(self, E=200*10**9, v=.3, G=None, state='plane strain'):
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
            self.sig_eqn = lambda e_xx, e_yy: (E /(1 - v**2)) * (e_xx + v*e_yy)
        else:
            self.sig_eqn = lambda e_xx, e_yy: (E * ((1 - v) * e_xx + v * e_yy)/
                                                   ((1 + v) * (1 - 2 * v)))
        
        
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


    def extract_peak_slice(self, phi=0, az_idx=None, q_idx=0, z_idx=0, 
                           err=False, fwhm=False):  
        """ 
        Extracts line profile through 2D/3D peak array.
        
        Must define **either** an azimuthal angle, phi, or azimuthal (cake)
        index. The azimuthal angle leverages the fitted strain profiles, 
        the az_idx plots that specific azimuthal slice. Note that for the
        EDXD detector in I12, az_idx == detector_idx.
        
        # phi:        Define angle (in rad) from which to calculate strain.
        # az_idx:     Index for azimuthal slice.
        # q_idx:      Specify lattice parameter/peak to save data from. 
        # z_idx:      Slice height (index) for 3D data
        # err:        Extract peak error (True/False)
        # FWHM:       Extract FWHM (True/False)
        """                        
        if fwhm:
            assert az_idx == None, "Can't extract fwhm from fitted data"
            if len(self.dims) != 3:            
                fwhm = self.fwhm[..., az_idx, q_idx]
            else:
                fwhm = self.fwhm[..., z_idx, az_idx, q_idx]                
            if len(self.dims) == 1:
                return self.co_ords[self.dims[0]], fwhm
            else:
                return [self.co_ords[dim] for dim in self.dims[:2]], fwhm
            
        dims, e = self.extract_strain_slice(phi, az_idx, q_idx, z_idx, err)
        return dims, self.q0[q_idx] - e * self.q0[q_idx]   


    def extract_strain_slice(self, phi=0, az_idx=None, q_idx=0, z_idx=0, 
                             err=False, shear=False):  
        """ 
        Extracts slice from 2D/3D strain array.
        
        Must define **either** an azimuthal angle, phi, or azimuthal (cake)
        index. The azimuthal angle leverages the fitted strain profiles, 
        the az_idx plots that specific azimuthal slice. Note that for the
        EDXD detector in I12, az_idx == detector_idx.
        
        # phi:        Define angle (in rad) from which to calculate strain.
        # az_idx:     Index for azimuthal slice.
        # q_idx:      Specify lattice parameter/peak to save data from. 
        # z_idx:      Slice height (index) for 3D data
        # err:        Extract strain error (True/False)
        # shear:      Extract shear strain (True/False)
        """  
        if az_idx != None:
            assert not shear, "Can't calc shear from individual az_idx"
            data = self.strain_err if err else self.strain
            if len(self.dims) != 3:            
                e = data[..., az_idx, q_idx]
            else:
                e = data[..., z_idx, az_idx, q_idx]

        else:
            assert not err, "Can't extract error from fitted data"
            
            if len(self.dims) == 3:
                e = np.nan * np.ones(self.strain.shape[:-3])
                params = self.strain_param[..., z_idx, q_idx, :]
            else:
                e = np.nan * np.ones(self.strain.shape[:-2])
                params = self.strain_param[..., q_idx, :]

            for idx in np.ndindex(e.shape):
                p = params[idx]
                if shear:
                    e[idx] = -np.sin(2 * (p[1] + phi) ) * p[0]
                else:
                    e[idx] = cos_(phi, *p)
        if len(self.dims) == 1:
            return self.co_ords[self.dims[0]], e
        elif len(self.dims) == 3:
            return [self.co_ords[dim][..., z_idx] for dim in self.dims[:2]], e
        else:
            return [self.co_ords[dim] for dim in self.dims], e


    def extract_stress_slice(self, phi=0, az_idx=None, q_idx=0, z_idx=0, 
                             err=False, shear=False):  
        """ 
        Extracts line profile through 2D/3D stress array.
        
        Must define **either** an azimuthal angle, phi, or azimuthal (cake)
        index. The azimuthal angle leverages the fitted strain profiles, 
        the az_idx plots that specific azimuthal slice. Note that for the
        EDXD detector in I12, az_idx == detector_idx.
        
        
        
        # phi:        Define angle (in rad) from which to calculate strain.
        # az_idx:     Index for azimuthal slice.
        # q_idx:      Specify lattice parameter/peak to save data from. 
        # z_idx:      Slice height (index) for 3D data
        # err:        Extract stress error (True/False)
        # shear:      Extract shear stress (True/False)
        """  
        dims, e_xx = self.extract_strain_slice(phi, az_idx, q_idx, z_idx, err)
        if shear:
            x = self.extract_strain_slice(phi, az_idx, q_idx, z_idx, err, True)
            e_xy = x[1]
        az_idx = None if az_idx == None else az90(self.phi, az_idx)
        phi = None if phi == None else phi + np.pi/2
        _, e_yy = self.extract_strain_slice(phi, az_idx, q_idx, z_idx, err)
        
        return dims, self.sig_eqn(e_xx, e_yy) if not shear else e_xy * self.G        

    
    def extract_peak_line(self, phi=0, az_idx=None, q_idx=0, z_idx=0, 
                            err=False, fwhm=False, pnt=(0,0), line_angle=0, 
                            npnts=100, method = 'linear'):  
        """ 
        Extracts line profile through 1D/2D/3D peak array.
        
        Must define **either** an azimuthal angle, phi, or azimuthal (cake)
        index. The azimuthal angle leverages the fitted strain profiles, 
        the az_idx plots that specific azimuthal slice. Note that for the
        EDXD detector in I12, az_idx == detector_idx.       
        
        # phi:        Define angle (in rad) from which to calculate strain.
        # az_idx:     Index for azimuthal slice.
        # q_idx:      Specify lattice parameter/peak to save data from. 
        # z_idx:      Slice height (index) for 3D data
        # err:        Extract peak error (True/False)
        # FWHM:       Extract FWHM (True/False)
        # pnt:        Centre point for data extraction  
        # line_angle: Angle across array to extract strain from
        # method:     Interpolation mehod (default = 'linear')
        """  
        data = self.extract_peak_slice(phi, az_idx, q_idx, z_idx, err, fwhm)
        return line_ext(*data, pnt, npnts, line_angle, method)
            
            
    def extract_strain_line(self, phi=0, az_idx=None, q_idx=0, z_idx=0, 
                            err=False, shear=False, pnt=(0,0), line_angle=0, 
                            npnts=100, method = 'linear'):  
        """ 
        Extracts line profile through 1D/2D/3D strain array.
        
        Must define **either** an azimuthal angle, phi, or azimuthal (cake)
        index. The azimuthal angle leverages the fitted strain profiles, 
        the az_idx plots that specific azimuthal slice. Note that for the
        EDXD detector in I12, az_idx == detector_idx.    
        
        # phi:        Define angle (in rad) from which to calculate strain.
        # az_idx:     Index for azimuthal slice.
        # q_idx:      Specify lattice parameter/peak to save data from. 
        # z_idx:      Slice height (index) for 3D data
        # err:        Extract strain error (True/False)
        # shear:      Extract shear strain (True/False)
        # pnt:        Centre point for data extraction  
        # line_angle: Angle across array to extract strain from
        # method:     Interpolation mehod (default = 'linear')
        """  
        data = self.extract_strain_slice(phi, az_idx, q_idx, z_idx, err, shear)
        return line_ext(*data, pnt, npnts, line_angle, method)


    def extract_stress_line(self, phi=0, az_idx=None, q_idx=0, z_idx=0, 
                            err=False, shear=False, pnt=(0,0), line_angle=0, 
                            npnts=100, method = 'linear'):  
        """ 
        Extracts line profile through 1D/2D/3D stress array.
        
        Must define **either** an azimuthal angle, phi, or azimuthal (cake)
        index. The azimuthal angle leverages the fitted strain profiles, 
        the az_idx plots that specific azimuthal slice. Note that for the
        EDXD detector in I12, az_idx == detector_idx.    
        
        # phi:        Define angle (in rad) from which to calculate strain.
        # az_idx:     Index for azimuthal slice.
        # q_idx:      Specify lattice parameter/peak to save data from. 
        # z_idx:      Slice height (index) for 3D data
        # err:        Extract stress error (True/False)
        # shear:      Extract stress strain (True/False)
        # pnt:        Centre point for data extraction  
        # line_angle: Angle across array to extract strain from
        # method:     Interpolation mehod (default = 'linear')
        """  
        data = self.extract_stress_slice(phi, az_idx, q_idx, z_idx, err, shear)
        return line_ext(*data, pnt, npnts, line_angle, method)
    
    
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
        
