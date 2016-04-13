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

        self.dims = list(group['dims'][:])
        self.phi = group['phi'][:]
        self.q0 = group['q0'][:]
        self.peak_windows = group['peak_windows'][:]
        self.peaks = group['peaks'][:]
        self.peaks_err = group['peaks_err'][:]    
    #try:
        self.fwhm = group['fwhm'][:]
        self.fwhm_err = group['fwhm_err'][:]
    #except:
        #pass
        self.strain_err = group['strain_err'][:]
        self.strain_param = group['strain_param'][:]
        
        self.q = group['q'][:] 
        self.I = group['data'][:]
        
        self.ss2_x = dimension_fill(self.f, 'ss2_x')   
        self.ss2_y = dimension_fill(self.f, 'ss2_y')
        self.ss2_z = dimension_fill(self.f, 'ss2_z')
        self.co_ords = {b'ss2_x': self.ss2_x,b'ss2_y': self.ss2_y, 
                        b'self_z': self.ss2_z} 
        
  
    def save_to_nxs(self, fname):
        """
        Saves all data back into an expanded .nxs file. Contains all original 
        data plus q0, peak locations and strain.
        
        # fname:      File name/location - default is to save to parent 
                      directory (*_md.nxs) 
        """
        fname = fname + '_md.nxs'
        
        with h5py.File(fname, 'w') as f:
            data_ids = ('dims', 'phi', 'slit_size', 'q0','peak_windows', 
                        'peaks', 'peaks_err', 'strain', 'strain_err', 
                        'strain_param', 'q', 'data') \
                        + tuple([dim.decode('utf8') for dim in self.dims])
            data_array = (self.dims, self.phi, self.slit_size, self.q0,  
                          self.peak_windows, self.peaks, self.peaks_err,  
                          self.strain, self.strain_err, self.strain_param, 
                          self.q, self.I) \
                          + tuple([self.co_ords[x] for x in self.dims])
            
            for data_id, data in zip(data_ids, data_array):
                base_tree = 'entry1/EDXD_elements/%s'
                if data_id == 'data':
                    f.create_dataset(base_tree % data_id, data = data, compression = 'gzip')
                else:
                    f.create_dataset(base_tree % data_id, data = data)



