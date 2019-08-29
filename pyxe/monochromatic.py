# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casimp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import warnings
import sys
import h5py
import numpy as np
import scipy.io as io

try:
    import fabio
    import pyFAI
    import pyFAI.azimuthalIntegrator
except ModuleNotFoundError:
    warnings.warn(
        "fabIO/pyFAI not installed. Azimuthal Integration not possible within"
        "pyxe.",
        ImportWarning,
    )
                

from pyxe.data_io import dim_fill, extract_fnames, dimension_fill_pixium10
from pyxe.peak_analysis import PeakAnalysis
from pyxpb.detectors import MonoDetector


# DLS Classes

class MonoDLS(PeakAnalysis):

    def __init__(self, fpath, 
                 d1='entry/result/ss1_x', 
                 d2='entry/result/ss1_y', d3=None,
                 q='entry/result/q',
                 I='entry/result/data',
                 phi='entry/result/azimuthal angle (degrees)',
                 detector=None):
        
        """ Extract data from pre-processed .nxs/.h5 file.

        Analysis of pre-processed (azimuthally integrated) data from the DLS
        (I12). Should re-arrange and scrape pre-processed data to correct form.

        Args:
            fpath (str): Path to processed datafile (.nxs)
            d1, d2, d3 (str): Path within .nxs file to co-ords
            q, I, phi (str): Path within .nxs file to q, intensity and phi
            detector (tuple, object): pyFAI detector object or Fit2D params:
                (sample_to_detector (mm), centre_x (pix), centre_y (pix),
                tilt (deg), tilt_plane (deg), pixel_size_x (micron),
                pixel_size_y (micron)).
        """
        self.fpath = fpath
        f = h5py.File(fpath, 'r')

        self.d1 = f[d1][()]
        self.d2 = None if d2 == None else f[d2][()]
        self.d3 = None if d3 == None else f[d3][()]
        self.ndim = sum(x is not None for x in [self.d1, self.d2, self.d3])

        q = f[q][()] 
        self.I = f[I][()] 
        self.phi = np.pi * f[phi][()] / 180 
        self.q = np.repeat(q[None, :], self.phi.size, axis=0)        
        self.analysis_state = 'integrated'

        if detector is None:
            # scrape from file..?
            self.detector = MonoDetector((2000, 2000), 0.1, 1000, 100, 1)
        else:
            self.detector = MonoDetector((2000, 2000), 0.1, 1000, 100, 1)
            
            
class MonoDLS_re(PeakAnalysis):

    def __init__(self, fpath, rpath,                  
                 d1='entry/result/ss1_x', 
                 d2='entry/result/ss1_y', d3=None,
                 q='entry/result/q',
                 I='entry/result/data',
                 phi='entry/result/azimuthal angle (degrees)',
                 detector=None):
        
        """ Extract intensity profiles from caked data file and the positions
        and parameters from the initial raw data file.

        Analysis of pre-processed (azimuthally integrated) data from the I12
        beamline. Should re-arrange and scrape pre-processed
        data to correct form.

        Args:
            fpath (str): Path to processed datafile (.nxs/.h5)
            rpath (str): Path to raw datafile (.nxs/.h5)
            detector (tuple, object): pyFAI detector object or Fit2D params:
                (sample_to_detector (mm), centre_x (pix), centre_y (pix),
                tilt (deg), tilt_plane (deg), pixel_size_x (micron),
                pixel_size_y (micron)).
        """
        self.fpath = fpath
        f = h5py.File(fpath, 'r')
        raw = h5py.File(rpath, 'r')
        
        # Extract position data from raw data file
        self.d1 = raw[d1][()]
        self.d2 = None if d2 == None else raw[d2][()]
        self.d3 = None if d3 == None else raw[d3][()]
        self.ndim = sum(x is not None for x in [self.d1, self.d2, self.d3])
        
        # Extract processed data from processed file
        q = f[q][()] 
        self.I = f[I][()] 
        self.phi = np.pi * f[phi][()] / 180 
        self.q = np.repeat(q[None, :], self.phi.size, axis=0)        
        self.analysis_state = 'integrated'

        if detector is None:
            # scrape from file..?
            self.detector = MonoDetector((2000, 2000), 0.1, 1000, 100, 1)
        else:
            self.detector = MonoDetector((2000, 2000), 0.1, 1000, 100, 1)


# ESRF Classes


def import_from_mat(fpath):
    mat = io.loadmat(fpath, squeeze_me=True)

    counts,unc,exposure,epoch,samplez,sampley,mondio,fnames = [],[],[],[],[],[],[],[]
    for scan in range(len(mat['w'])):
        try:
            q = mat['w'][scan][()][0]
            _counts = mat['w'][scan][()][1]
            _unc = mat['w'][scan][()][2] #error on pixel?
            _exposure = mat['w'][scan][()][3]
            _epoch = mat['w'][scan][()][4]
            _laserz = mat['w'][scan][()][5] #z laser-sample separation - not varied
            _samplez = mat['w'][scan][()][6] 
            #_samplez = mat['w'][scan][()][7] #ignore, same as above
            _sampley = mat['w'][scan][()][8] #pp01 
            _fnames = mat['w'][scan][()][9]
            _mondio = mat['w'][scan][()][10] #region of interest on 
        except:
            pass
            #print('Could not get scan %i'%scan)
        else:
            counts.append(_counts)
            unc.append(_unc)
            exposure.append(_exposure)
            epoch.append(_epoch)
            samplez.append(_samplez)
            sampley.append(_sampley)
            mondio.append(_mondio)
            fnames.append(_fnames)

    counts = np.dstack(counts)
    counts = np.swapaxes(counts, 0, 2)
    unc = np.dstack(unc)
    unc = np.swapaxes(unc, 0, 2)
    exposure = np.hstack(exposure)
    epoch = np.hstack(epoch)
    samplez = np.hstack(samplez)
    sampley = np.hstack(sampley)
    mondio = np.hstack(mondio)
    fnames = np.hstack(fnames)
    del mat
    
    return dict(q=q,counts=counts,unc=unc,exposure=exposure,epoch=epoch,
                samplez=samplez,sampley=sampley,mondio=mondio,fnames=fnames)
        
        
class MonoESRF_mat(PeakAnalysis):
    
    def __init__(self, fpath, detector=None):
        data = import_from_mat(fpath)
        self.ndim = 2
        self.d1, self. d2, self.d3 = data['samplez'], data['sampley'], None
        self.T = None
        q, self.I = data['q'], data['counts']
        self.phi = np.linspace(-np.pi, np.pi, 36)
        self.q = np.repeat(q[None, :], self.phi.size, axis=0)
        
        self.analysis_state = 'integrated'

        if detector is None:
            # scrape from file..?
            self.detector = MonoDetector((2000, 2000), 0.1, 1000, 100, 1)
        else:
            self.detector = MonoDetector((2000, 2000), 0.1, 1000, 100, 1)


## Built in azimuthal integration (not recommended)

class MonoPyFAI(PeakAnalysis):

    def __init__(self, folder, co_ords, detector, wavelength=None,
                 f_ext='.edf', progress=True, npt_rad=1024, npt_az=36,
                 az_range=(-180, 180)):
        """
        Takes a folder containing image files from area detectors and cakes
        the data while associating it with spatial information. The caked data
        can then be analysed (peak_fit/strain calculations).

        Args:
            folder (str): Folder containing the image files for analysis
            co_ords (ndarray): 1D/2D/3D numpy array containing data co-ords
            detector (tuple, object): pyFAI detector object or Fit2D params:
                (sample_to_detector (mm), centre_x (pix), centre_y (pix),
                tilt (deg), tilt_plane (deg), pixel_size_x (micron),
                pixel_size_y (micron)).
            wavelength (float): Wavelength (nm) (if not defined in pyFAI file)
            f_ext (str): File extension of image files
            progress (bool): Live progress bar
            npt_rad (int): Number of radial bins (approx. half detect width)
            npt_az (int): Number of azimuthal wedges
            az_range (tuple): Azimtuhal range (deg) - default (-180, 180).
                Note that the 0deg is at eastern edge of circle.
        """
        
        warnings.warn(
            "pyFAI has been upgrading, deprecating certain functions and "
            "likely breaking this class. Testing has been removed.",
            DeprecationWarning
        )
        
        
        fname = '{}.h5'.format(os.path.split(folder)[1])
        self.fpath = os.path.join(folder, fname)

        error = 'Azimuthal range must be less than or equal to 360deg'
        assert abs(np.max(az_range) - np.min(az_range)) <= 360, error

        # Allow for use of pyFAI or Fit2D detector params
        # Currently checking if folder string - perhaps not the best method!

        if isinstance(detector, pyFAI.azimuthalIntegrator.AzimuthalIntegrator):
            ai = detector
            if wavelength is None:
                ai.get_wavelength()
            else:
                ai.set_wavelength(wavelength)
        else:
            ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator()
            ai.setFit2D(*detector)
            ai.set_wavelength(wavelength)
        fnames = extract_fnames(folder, f_ext)

        error = ('Number of positions not equal to number of files (pos = %s,'
                 ' files = %s)' % (co_ords.shape[0], len(fnames)))
        assert co_ords.shape[0] == len(fnames), error
        (self.d1, self.d2, self.d3), self.dims = dim_fill(co_ords)
        self.T = None
        self.ndim = len(self.dims)
        self.I = np.nan * np.ones((co_ords.shape[0], npt_az, npt_rad))

        print('\nLoading files and carrying out azimuthal integration:\n')
        for fidx, fname in enumerate(fnames):
            img = fabio.open(os.path.join(folder, fname)).data
            npt_rad, npt_az = int(npt_rad), int(npt_az)
            I, q_, phi = ai.integrate2d(img, npt_rad=npt_rad, npt_azim=npt_az,
                                        azimuth_range=az_range, unit='q_A^-1')
            self.I[fidx] = I
            if progress:
                sys.stdout.write('\rProgress: [{0:20s}] {1:.0f}%'.format('#' *
                                 int(20*(fidx + 1) / len(fnames)),
                                 100*((fidx + 1)/len(fnames))))
                sys.stdout.flush()

        # Create az_slice 'specific' q values - to work with edxd data
        self.q = np.repeat(q_[None, :], npt_az, axis=0)
        self.phi = phi * np.pi / 180
        self.analysis_state = 'integrated'
        # Temporary - extract from ai!
        self.detector = MonoDetector((2000,2000), 0.1, 1000, 100, 1)
        
        


            
