# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casimp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six import string_types
import os
import fabio
import numpy as np
import pyFAI
import h5py
import sys
from scipy.optimize import curve_fit

from pyxe.fitting_tools import array_fit
from pyxe.fitting_functions import cos_
from pyxe.strain_tools import StrainTools
from pyxe.plotting import StrainPlotting
from pyxe.analysis_tools import dim_fill, mirror_data


class Area(StrainTools, StrainPlotting):
    """
    Takes a folder containing image files from area detectors and cakes the 
    data while associating it with spatial information. The caked data can then
    be analysed (peak_fit/strain calculations). 
    """
   
    def __init__(self, folder, pos_data, params, f_ext='.edf', progress=True,
                 pos_delim=',', npt_rad=1024, npt_az=36, az_range=[-180, 180]):
        """
        # folder:     Folder containing the image files for analysis
        # pos_data:   Either csv file or numpy array containing position data 
        # params:     Accepts a file location for a .poni parameter file 
                      produced by pyFAI-calib. Alternative directly enter 
                      calibration details as take from Fit2D in form:
                      (sample_to_detector (mm), centre_x (pix), centre_y (pix), 
                      tilt (deg), tilt_plane (deg), pixel_size_x (micron), 
                      pixel_size_y (micron), wavelength (m))
        # pos_delim:  If in .csv format the file delimiter
        # npt_rad:    Number of radial bins, should equal half detector pix
        # npt_az:     Number of azimuthal wedges           
        # az_range:   Range of azimuthal values to investigate - note that 0
                      degrees is defined at the eastern edge of the circle.
        """
        self.name = folder
        error = 'Azimuthal range must be less than or equal to 360deg'
        assert abs(np.max(az_range) - np.min(az_range)) <= 360, error
        if isinstance(params, ("".__class__, u"".__class__)):
            ai = pyFAI.load(params) ## CORRECT??
        else:
            ai = pyFAI.AzimuthalIntegrator()
            ai.setFit2D(*params[:-1])
            ai.set_wavelength(params[-1])
        
        fnames = sorted([x for x in os.listdir(folder) if x.endswith(f_ext)])

        error = ('Position data not supplied in correct format: \n'
                 ' pos_data variable must be the (str) path to a csv file, '
                 'a numpy array containing 1, 2 or 3d co-ordinates')
        assert isinstance(pos_data, (string_types, np.ndarray, tuple)), error
        if isinstance(pos_data, string_types):
            positions = np.loadtxt(pos_data, delimiter = pos_delim)
        else:
            positions = pos_data
        
        error = ('Number of positions not equal to number of files (pos = %s,' 
                ' files = %s)' % (positions.shape[0], len(fnames)))
        assert positions.shape[0] == len(fnames), error

        (self.ss2_x, self.ss2_y, self.ss2_z), self.dims = dim_fill(positions)   
        self.co_ords = {b'ss2_x': self.ss2_x, b'ss2_y': self.ss2_y, 
                        b'self_z': self.ss2_z} 
              
        self.I = np.nan * np.ones((positions.shape[0], npt_az, npt_rad))

        print('\nLoading files and carrying out azimuthal integration:\n')
        for fidx, fname in enumerate(fnames):
            img = fabio.open(os.path.join(folder, fname)).data
            I, q_, phi = ai.integrate2d(img, npt_rad=npt_rad, npt_azim=npt_az, 
                                        azimuth_range=az_range, unit='q_A^-1')
            self.I[fidx] = I 
            if progress:
                sys.stdout.write("\rProgress: [{0:20s}] {1:.0f}%".format('#' * 
                int(20*(fidx + 1)/ len(fnames)), 100*((fidx + 1)/len(fnames))))
                sys.stdout.flush()
            
            
        self.q = np.repeat(q_[None, :], npt_az, axis = 0)
        self.phi = phi * np.pi / 180
        
        
    def peak_fit(self, q0, window, mirror = True, func = 'gaussian', 
                     error_limit = 2 * 10 ** -4, progress = True):
            
        # Convert int or float to list
        self.q0 = [q0] if isinstance(q0, (int, float, np.float64)) else q0
        self.peak_windows = [[q - window/2, q + window/2] for q in self.q0]
    
        # Accept detector specific q0 2d-array
        if len(np.shape(self.q0)) == 2:
            q0_av = np.nanmean(self.q0, 0)
            self.peak_windows = [[q - window/2, q + window/2] for q in q0_av]
        
        # Iterate across q0 values and fit peaks for all detectors
        array_shape = self.I.shape[:-1] + (np.shape(self.q0)[-1],)
        data = [np.nan * np.ones(array_shape) for i in range(4)]
        self.peaks, self.peaks_err, self.fwhm, self.fwhm_err = data

        print('\n%s: - %s acquisition points\n' % (self.name, self.ss2_x.size))
        
        for idx, window in enumerate(self.peak_windows):
            fit = array_fit(self.q,self.I, window, func, error_limit, progress)
            self.peaks[..., idx], self.peaks_err[..., idx] = fit[0], fit[1]
            self.fwhm[..., idx], self.fwhm_err[..., idx] = fit[2], fit[3]
        
        if mirror:
            _, self.peaks = mirror_data(self.phi, self.peaks)
            _, self.peaks_err = mirror_data(self.phi, self.peaks_err)
            _, self.fwhm = mirror_data(self.phi, self.fwhm)
            self.phi, self.fwhm_err = mirror_data(self.phi, self.fwhm_err)

        self.strain = (self.q0 - self.peaks)/ self.q0
        self.strain_err = (self.q0 - self.peaks_err)/ self.q0
        self.full_ring_fit()


    def full_ring_fit(self):
        """
        Fits a sinusoidal curve to the strain information from each detector. 
        """
        data_shape = self.peaks.shape[:-2] + self.peaks.shape[-1:] + (3, )
        self.strain_param = np.nan * np.ones(data_shape)
        self.fwhm_param = np.nan * np.ones(data_shape)
        for name, raw_data, param in zip(['peaks', 'fwhm'],
                                         [self.strain, self.fwhm],
                                         [self.strain_param, self.fwhm_param]):
            for idx in np.ndindex(data_shape[:-1]):
                data = raw_data[idx[:-1]][..., idx[-1]]
                not_nan = ~np.isnan(data)
                count = 0
                if self.phi[not_nan].size > 2:
                    # Estimate curve parameters
                    p0 = [np.nanmean(data), 3 * np.nanstd(data)/(2**0.5), 0]
                    try:
                        a, b = curve_fit(cos_,self.phi[not_nan], data[not_nan], p0)
                        param[idx] = a
                    except (TypeError, RuntimeError):
                        count += 1
                        #print('Unable to fit curve to data.')
                else:
                    count += 1
                    #print('Insufficient data to attempt curve_fit.')   
            print('\nUnable to fit full ring (%s) data %i out of %i points'
                  % (name, count, np.size(self.peaks[:, 0, 0])))
            
    def save_to_nxs(self, fname=None):
        """
        Saves all data back into an expanded .nxs file. Contains all original 
        data plus q0, peak locations and strain.
        
        # fname:      File name/location - default is to save to parent 
                      directory (*_md.nxs) 
        """
        if fname == None:
            fname = self.name + '_md.nxs'
        data_dict = {'dims': self.dims,
                     'phi': self.phi,
                     'peak_windows': self.peak_windows,
                     'q0': self.q0,
                     'peak_windows': self.peak_windows,
                     'peaks': self.peaks,
                     'peaks_err': self.peaks_err,
                     'fwhm': self.fwhm,
                     'fwhm_err': self.fwhm_err,
                     'strain': self.strain,
                     'strain_err': self.strain_err,
                     'strain_param': self.strain_param,
                     'q': self.q,
                     'data': self.I}
        dims = tuple([dim.decode('utf8') for dim in self.dims])
        dim_data = tuple([self.co_ords[x] for x in self.dims])
        for idx, dim in enumerate(dims):
            data_dict[dim] = dim_data[idx]        
        
        with h5py.File(fname, 'w') as f:
            for data in data_dict:
                base_tree = 'entry1/EDXD_elements/%s'
                if data == 'data':
                    f.create_dataset(base_tree % data, data=data_dict[data], 
                                     compression = 'gzip')
                else:
                    f.create_dataset(base_tree % data, data=data_dict[data])
                
