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
from pyxe.fitting_functions import strain_transformation
from scipy.optimize import curve_fit


from pyxe.plotting import StrainPlotting
from pyxe.strain_tools import StrainTools
from pyxe.analysis_tools import dimension_fill


class Reload(StrainTools, StrainPlotting):
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
    def __init__(self, file):

        self.name = file
        self.f = h5py.File(file, 'r') 
        group = self.f['entry1/EDXD_elements']
        
        # 'raw' data
        self.dims = list(group['dims'][:])
        self.phi = group['phi'][:]
        self.I = group['data'][:]
        self.q = group['q'][:] 
        self.ss2_x = dimension_fill(self.f, 'ss2_x')   
        self.ss2_y = dimension_fill(self.f, 'ss2_y')
        self.ss2_z = dimension_fill(self.f, 'ss2_z')
        self.co_ords = {b'ss2_x': self.ss2_x, b'ss2_y': self.ss2_y,
                        b'self_z': self.ss2_z} 
        # 'calculated' data
        try:        
            self.strain = group['strain'][:]
        except KeyError as e:
            e.args += ('Invalid .nxs file - no strain data found.',
                       'Run XRD_analysis tool.')
            raise
        self.q0 = group['q0'][:]
        self.peak_windows = group['peak_windows'][:]
        self.peaks = group['peaks'][:]
        self.peaks_err = group['peaks_err'][:]    
        try:
            self.fwhm = group['fwhm'][:]
            self.fwhm_err = group['fwhm_err'][:]
        except KeyError:
            pass
        self.strain_err = group['strain_err'][:]
        self.strain_param = group['strain_param'][:]
        
    def full_ring_fit(self):
        """
        Fits a sinusoidal curve to the strain information from each detector. 
        """
        data_shape = self.peaks.shape[:-2] + self.peaks.shape[-1:] + (3, )
        self.strain_param = np.nan * np.ones(data_shape)
        name = 'peaks'
        raw_data = self.strain[..., :-1, :]
        param = self.strain_param
            
        for idx in np.ndindex(data_shape[:-1]):
            data = raw_data[idx[:-1]][..., idx[-1]]
            not_nan = ~np.isnan(data)
            count = 0
            if self.phi[not_nan].size > 2:
                # Estimate curve parameters
                p0 = [np.nanmean(data), 3 * np.nanstd(data)/(2**0.5), 0]
                try:
                    a, b = curve_fit(strain_transformation, self.phi[not_nan],
                                     data[not_nan], p0)
                    param[idx] = a
                except (TypeError, RuntimeError):
                    count += 1
            else:
                count += 1
        print('\nUnable to fit full ring (%s) data %i out of %i points'
              % (name, count, np.size(self.peaks[:, 0, 0])))

    def save_to_nxs(self, fname=None):
        """
        Saves all data back into an expanded .nxs file. Contains all original 
        data plus q0, peak locations and strain.
        
        # fname:      File name/location - default is to save to parent 
                      directory (*_md.nxs) 
        """
        if fname is None:
            fname = '%s_md.nxs' % self.name[:-4]
        
        with h5py.File(fname, 'w') as f:
            data_ids = ('dims', 'phi', 'q0', 'peak_windows',
                        'peaks', 'peaks_err', 'strain', 'strain_err', 
                        'strain_param', 'q', 'data') \
                        + tuple([dim.decode('utf8') for dim in self.dims])
            data_array = (self.dims, self.phi, self.q0, self.peak_windows,
                          self.peaks, self.peaks_err, self.strain,
                          self.strain_err, self.strain_param, self.q, self.I,
                          ) + tuple([self.co_ords[x] for x in self.dims])
            
            for data_id, data in zip(data_ids, data_array):
                base_tree = 'entry1/EDXD_elements/%s'
                if data_id == 'data':
                    f.create_dataset(base_tree % data_id, data=data,
                                     compression='gzip')
                else:
                    f.create_dataset(base_tree % data_id, data=data)
