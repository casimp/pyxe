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
import re
import shutil
import numpy as np
from scipy.optimize import curve_fit

from edi12.fitting_optimization import array_fit
from edi12.peak_fitting import cos_
from edi12.XRD_tools import XRD_tools


class XRD_analysis(XRD_tools):
    """
    Takes an un-processed .nxs file from the I12 EDXD detector and fits curves
    to all specified peaks for each detector. Calculates strain and details
    associated error. 
    """
   
    def __init__(self, file, q0, window, func = 'gaussian', 
                 error_limit = 1 * 10 ** -4, output = 'simple'):
        """
        Extract and manipulate all pertinent data from the .nxs file. Takes 
        either one or multiple (list) q0s.
        """
        super(XRD_tools, self).__init__(file)
        scan_command = self.f['entry1/scan_command'][:]
        self.dims = re.findall(b'ss2_\w+', scan_command)
        try:        
            self.slit_size = self.f['entry1/before_scan/s4/s4_xs'][0]
        except KeyError:
            self.slit_size = []              
        group = self.f['entry1/EDXD_elements']
        q, I = group['edxd_q'], group['data']
        
        # Convert int or float to list
        self.q0 = [q0] if isinstance(q0, (int, float, np.float64)) else q0
        self.peak_windows = [[q - window/2, q + window/2] for q in self.q0]
        
        # Accept detector specific q0 2d-array
        if len(np.shape(self.q0)) == 2:
            q0_av = np.nanmean(self.q0, 0)
            self.peak_windows = [[q - window/2, q + window/2] for q in q0_av]
 
        # Iterate across q0 values and fit peaks for all detectors
        array_shape = I.shape[:-1] + (np.shape(self.q0)[-1],)
        self.peaks = np.nan * np.ones(array_shape)
        self.peaks_err = np.nan * np.ones(array_shape)
        
        print('\nFile: %s - %s acquisition points\n' % 
             (self.filename, self.f['entry1/EDXD_elements/ss2_x'].size))
        
        for idx, window in enumerate(self.peak_windows):
            fit_data = array_fit(q, I, window, func, error_limit, output)
            self.peaks[..., idx], self.peaks_err[..., idx] = fit_data
        self.strain = (self.q0 - self.peaks)/ self.q0
        self.strain_err = (self.q0 - self.peaks_err)/ self.q0
        self.strain_fit(error_limit)

    def strain_fit(self, error_limit):
        """
        Fits a sinusoidal curve to the strain information from each detector. 
        """
        data_shape = self.strain.shape
        self.strain_param = np.nan * np.ones(data_shape[:-2] + \
                            (data_shape[-1], ) + (3, ))
        for idx in np.ndindex(data_shape[:-2] + (data_shape[-1],)):
            data = self.strain[idx[:-1]][:-1][..., idx[-1]]
            not_nan = ~np.isnan(data)
            angle = np.linspace(0, np.pi, 23)
            if angle[not_nan].size > 2:
                # Estimate curve parameters
                p0 = [np.nanmean(data), 3*np.nanstd(data)/(2**0.5), 0]
                try:
                    a, b = curve_fit(cos_, angle[not_nan], data[not_nan], p0)
                    perr_ = np.diag(b)
                    perr = np.sqrt(perr_[0] + perr_[2])
                    if perr < 2 * error_limit:              
                        self.strain_param[idx] = a
                except (TypeError, RuntimeError):
                    print('Unable to fit peak.')
            else:
                print('Insufficient data to attempt curve_fit.')
                
        

    def save_to_nxs(self, fname = None):
        """
        Saves all data back into an expanded .nxs file. Contains all original 
        data plus q0, peak locations and strain.
        
        # fname:      File name/location - default is to save to parent 
                      directory (*_md.nxs) 
        """
        if fname == None:        
            fname = '%s_md.nxs' % self.filename[:-4]
        
        shutil.copy(self.filename, fname)
        with h5py.File(fname, 'r+') as f:
            data_ids = ('dims', 'slit_size', 'q0','peak_windows', 'peaks',  
                        'peaks_err', 'strain', 'strain_err', 'strain_param')
            data_array = (self.dims, self.slit_size, self.q0,  
                          self.peak_windows, self.peaks, self.peaks_err,  
                          self.strain, self.strain_err, self.strain_param)
            
            for data_id, data in zip(data_ids, data_array):
                base_tree = 'entry1/EDXD_elements/%s'
                f.create_dataset(base_tree % data_id, data = data)
                
                