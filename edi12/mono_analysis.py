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

from edi12.fitting_optimization import array_fit


def exclusion(file_list, exclusion_list):
    fnames = []
    for fname in file_list:
        if not any([s.upper() in fname.upper() for s in exclusion_list]):
            fnames.append(fname)
    return fnames
    
def dim_fill(data):

    co_ords = []
    dims = []
    for axis, dim in zip(range(3), ['ss2_x', 'ss2_y', 'ss2_z']):
        try:
            co_ords.append(data[:, axis])
            dims.append(dim)
        except IndexError:
            co_ords.append(None)
    return co_ords, dims


class Mono_analysis():
    """
    Takes an un-processed .nxs file from the I12 EDXD detector and fits curves
    to all specified peaks for each detector. Calculates strain and details
    associated error. 
    """
   
    def __init__(self, folder, pos_file, det_params, q0, window, 
                 func = 'gaussian', error_limit = 1 * 10 ** -4, output = 'simple',
                 pos_delimiter = ',', exclude = [], npt_rad = 1024, npt_azim = 36, 
                 azimuth_range = [-175, 185]):
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
        if isinstance(det_params, str):
            ai = pyFAI.AzimuthalIntegrator(det_params)
        else:
            ai = pyFAI.AzimuthalIntegrator()
            ai.setFit2D(*det_params[:-1])
            ai.set_wavelength(det_params[-1])
        print(ai)
        
        exclude.append(os.path.split(pos_file)[1])
        fnames = exclusion(os.listdir(folder), exclude)
        
        positions = np.loadtxt(pos_file, delimiter=pos_delimiter)
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
        self.peaks = np.nan * np.ones((array_shape))   
        self.peaks_err = np.nan * np.ones(array_shape)

        # consider saving all profiles?        
        
        for fidx, fname in enumerate(fnames):
            img = fabio.open(os.path.join(folder, fname)).data
            I, q, phi = ai.integrate2d(img, npt_rad = npt_rad, npt_azim = npt_azim, azimuth_range = azimuth_range, unit='q_A^-1')
            
            q = np.repeat(q[None, :], npt_azim, axis = 0)
            for i in range(npt_azim):            
                plt.plot(q[i], I[i])
            plt.xlim([2.5, 5])

            for idx, window in enumerate(self.peak_windows):
                fit_data = array_fit(q, I, window, func, error_limit, output, unused_detectors = [])
                self.peaks[fidx, ..., idx], self.peaks_err[fidx, ..., idx] = fit_data
                
        self.slit_size = []
        self.strain = (self.q0 - self.peaks)/ self.q0
        self.strain_err = (self.q0 - self.peaks_err)/ self.q0
        
        
    def save_to_nxs(self, fname):
        """
        Saves all data back into an expanded .nxs file. Contains all original 
        data plus q0, peak locations and strain.
        
        # fname:      File name/location - default is to save to parent 
                      directory (*_md.nxs) 
        """
        fname = '_md.nxs'
        
        with h5py.File(fname, 'w') as f:
            data_ids = ('dims', 'slit_size', 'q0','peak_windows', 'peaks',  
                        'peaks_err', 'strain', 'strain_err', 'strain_param')
            data_array = (self.dims, self.slit_size, self.q0,  
                          self.peak_windows, self.peaks, self.peaks_err,  
                          self.strain, self.strain_err, self.strain_param)
            
            for data_id, data in zip(data_ids, data_array):
                base_tree = 'entry1/EDXD_elements/%s'
                f.create_dataset(base_tree % data_id, data = data)

            

poni = (741.577, 1053.137, 1027.562, 0.153, 41.314, 200, 200, 1.631371*10**-11)
base_folder = 'N:/Work Data/ee11080/Test15_CNTI6/'

bidge_holder = []
for i in [0, 602, 610, 751, 753]:
    folder= base_folder + str(i)
    pos_file = folder + '/positions.csv'
    bidge = Mono_analysis(folder, pos_file, poni, 3.505, 0.25, 
                          pos_delimiter = ',', exclude = ['dark'], output = 'simple', 
                          error_limit = 5 * 10 ** -4, azimuth_range = [-160, -20],
                          npt_azim = 8)
                          
    bidge_holder.append(bidge)
#2.5286991970803694
print('hi')
plt.figure(figsize = (12,6))

for bidge in bidge_holder:
    plt.plot(bidge.ss2_x[:-1], bidge.strain[:-1, 4], '*-')
    
#plt.xlim([0, 5.0])
#plt.ylim([-0.0005, 0.001])
plt.legend([0, 602, 610, 751, 753], loc = 4)



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

