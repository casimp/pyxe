# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casimp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import os
import fabio
import numpy as np
import pyFAI
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pyxe.fitting_tools import array_fit
from pyxe.fitting_functions import cos_
from pyxe.strain_tools import StrainTools
from pyxe.plotting import StrainPlotting




def exclusion(file_list, exclusion_list):
    fnames = []
    for fname in file_list:
        if not any([s.upper() in fname.upper() for s in exclusion_list]):
            fnames.append(fname)
    return fnames
    
def dim_fill(data):
    print(data)

    co_ords = []
    dims = []
    
    if data.ndim == 1:
        return [data, None, None], [b'ss2_x']
    for axis, dim in zip(range(3), [b'ss2_x', b'ss2_y', b'ss2_z']):
        try:
            co_ords.append(data[:, axis])
            dims.append(dim)
        except IndexError:
            co_ords.append(None)
    print(co_ords)
    return co_ords, dims


class Area(StrainTools, StrainPlotting):
    """
    Takes an un-processed .nxs file from the I12 EDXD detector and fits curves
    to all specified peaks for each detector. Calculates strain and details
    associated error. 
    """
   
    def __init__(self, folder, pos_data, det_params, q0, window, f_ext = '.edf', 
                 mirror = True, func = 'gaussian', error_limit = 2 * 10 ** -4, output = 'simple',
                 pos_file_delimiter = ',', exclude = [], npt_rad = 1024, npt_azim = 36, 
                 azimuth_range = [-180, 180]):
        """
        
        # det_params: Accepts a file location for a .poni parameter file 
                      produced by pyFAI-calib. Alternative directly enter 
                      calibration details as take from Fit2D in form:
                      (sample_to_detector (mm), centre_x (pix), centre_y (pix), 
                      tilt (deg), tilt_plane (deg), pixel_size_x (micron), 
                      pixel_size_y (micron), wavelength (m))
                      
        # azim_range: Range of azimuthal values to investigate - note that 0
                      degrees is defined at the eastern edge of the circle.
        """
        error = 'Azimuthal range must be less than or equal to 360deg'
        assert abs(np.max(azimuth_range) - np.min(azimuth_range)) <= 360, error
        if isinstance(det_params, ("".__class__, u"".__class__)):
            ai = pyFAI.load(det_params) ## CORRECT??
            
        else:
            ai = pyFAI.AzimuthalIntegrator()
            ai.setFit2D(*det_params[:-1])
            ai.set_wavelength(det_params[-1])
        print(ai)
        
        
        #exclude.append(os.path.split(pos_data)[1]) # Now search for actual files...
        fnames = sorted([x for x in os.listdir(folder) if x[-len(f_ext):] == f_ext])
        
        print(len(fnames))
        fnames = exclusion(fnames, exclude)
        print(fnames)
        error = ('Position data not supplied in correct format: \n'
                 ' pos_data variable must be the (str) path to a csv file, '
                 'a numpy array containing 1, 2 or 3d co-ordinates')
        assert isinstance(pos_data, ("".__class__, u"".__class__, np.ndarray, tuple)), error
        if isinstance(pos_data, ("".__class__, u"".__class__)):
            positions = np.loadtxt(pos_data, delimiter = pos_file_delimiter)
        else:
            positions = pos_data
        
        print(fnames, positions)
        error = 'Number of positions not equal to number of files (pos = %s,' \
                ' files = %s)' % (positions.shape[0], len(fnames))
        assert positions.shape[0] == len(fnames), error

        (self.ss2_x, self.ss2_y, self.ss2_z), self.dims = dim_fill(positions)   
        self.co_ords = {b'ss2_x': self.ss2_x, b'ss2_y': self.ss2_y, 
                        b'self_z': self.ss2_z} 
        
        # Convert int or float to list
        self.q0 = [q0] if isinstance(q0, (int, float, np.float64)) else q0
        self.peak_windows = [[q_ - window/2, q_ + window/2] for q_ in self.q0]
        
        # Accept detector specific q0 2d-array
        if len(np.shape(self.q0)) == 2:
            q0_av = np.nanmean(self.q0, 0)
            self.peak_windows = [[q_ - window/2, q_ +window/2] for q_ in q0_av]
        
        array_shape = (positions.shape[0], npt_azim, ) + (np.shape(self.q0)[-1],)
        data = [np.nan * np.ones(array_shape) for i in range(4)]
        self.peaks, self.peaks_err, self.fwhm, self.fwhm_err = data

        # consider saving all profiles?        
        shape_I = (positions.shape[0], npt_azim, npt_rad)
        self.I = np.nan * np.ones((shape_I))
        for fidx, fname in enumerate(fnames):
            img = fabio.open(os.path.join(folder, fname)).data
            I, q_, phi = ai.integrate2d(img, npt_rad = npt_rad, npt_azim = npt_azim, azimuth_range = azimuth_range, unit='q_A^-1')
            self.I[fidx] = I         
            
            q = np.repeat(q_[None, :], npt_azim, axis = 0)

            for idx, window in enumerate(self.peak_windows):
                a, b, c, d = array_fit(q, I, window, func, error_limit, output, unused_detectors = [])
                self.peaks[..., idx], self.peaks_err[..., idx] = a, b
                self.fwhm[..., idx], self.fwhm_err[..., idx] = c, d
        self.q = q
        print(phi)        
        self.phi = phi * np.pi / 180        
        
        if mirror:
            # has to be even number of slices but uneven number of boundaries.
            angles = self.phi[:int(self.phi[:].shape[0]/2)]
            #discard = self.phi[int(self.phi[:].shape[0]/2):]
            peak_shape = self.peaks.shape
            phi_len = int(peak_shape[-2]/2)
            new_shape = (peak_shape[:-2] + (phi_len, ) + peak_shape[-1:]) 
            print(new_shape)
            mirror_peak = np.nan * np.zeros(new_shape)
            mirror_err = np.nan * np.zeros(new_shape)
            for i in range(phi_len):
                mirror_peak[:, i] = (self.peaks[:, i] + self.peaks[:, i + new_shape[-2]]) / 2
                mirror_err[:, i] = (self.peaks_err[:, i] + self.peaks_err[:, i + new_shape[-2]]) / 2
            self.peaks = mirror_peak
            self.phi = angles
            self.peaks_err = mirror_err
            
        self.slit_size = []
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
            
            data = self.strain[idx[:-1]][..., idx[-1]]
            not_nan = ~np.isnan(data)
            
            if self.phi[not_nan].size > 2:
                # Estimate curve parameters
                p0 = [np.nanmean(data), 3*np.nanstd(data)/(2**0.5), 0]
                try:
                    a, b = curve_fit(cos_, self.phi[not_nan], data[not_nan], p0)
                    perr_ = np.diag(b)
                    perr = np.sqrt(perr_[0] + perr_[2])
                    if perr < 2 * error_limit:              
                        self.strain_param[idx] = a
                except (TypeError, RuntimeError):
                    print('Unable to fit curve to data.')
            else:
                print('Insufficient data to attempt curve_fit.')        
        
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
                
         

class MonoTestCase(unittest.TestCase):
    """
    Tests parts of XRD analysis
    """

    def setUp(self):
        pass
        
    def test_exclusion(self):
        """
        Test exclusion - check for correct return.
        """
        file_list = ['Purgative', 'maccabees', 'kuwait', 'hypostatizing', 
                     'patternable', '', 'UNDENIABLE']
        exclusion_list = ['ble', 'maccabee', 'wait']
        self.assertEqual(exclusion(file_list, exclusion_list), 
                         ['Purgative', 'hypostatizing', ''])
        
if __name__ == '__main__':
    unittest.main()

