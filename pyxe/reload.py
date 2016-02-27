# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casimp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from pyxe.plotting import StrainPlotting
from pyxe.strain_tools import StrainTools
import h5py



def dimension_fill(data, dim_ID):
    """
    Extracts correct spatial array from hdf5 file. Returns None is the
    dimension doesn't exist.
    
    # data:       Raw data (hdf5 format)   
    # dim_ID:     Dimension ID (ss_x, ss2_y or ss2_z)
    """
    try:
        dimension_data = data['entry1/EDXD_elements/' + dim_ID][:]
    except KeyError:
        dimension_data = None
    return dimension_data



class Reload(StrainTools, StrainPlotting):
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
    def __init__(self, file):

        self.filename = file
        self.f = h5py.File(file, 'r') 
        group = self.f['entry1']['EDXD_elements']
        
        try:        
            self.strain = group['strain'][:]
        except KeyError as e:
            e.args += ('Invalid .nxs file - no strain data found.',
                       'Run XRD_analysis tool.')
            raise

        self.dims = group['dims'][:]
        self.phi = group['phi'][:]
        self.q0 = group['q0'][:]
        self.peak_windows = group['peak_windows'][:]
        self.peaks = group['peaks'][:]
        self.peaks_err = group['peaks_err'][:]    
        self.strain_err = group['strain_err'][:]
        self.strain_param = group['strain_param'][:]
        

        self.ss2_x = dimension_fill(self.f, 'ss2_x')   
        self.ss2_y = dimension_fill(self.f, 'ss2_y')
        self.ss2_z = dimension_fill(self.f, 'ss2_z')
        self.co_ords = {b'ss2_x': self.ss2_x,b'ss2_y': self.ss2_y, 
                        b'self_z': self.ss2_z} 
        


