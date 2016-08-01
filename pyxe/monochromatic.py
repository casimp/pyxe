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
import fabio
import numpy as np
import pyFAI
import sys

from pyxe.analysis_tools import dim_fill
from pyxe.peak_analysis import PeakAnalysis
from pyxpb.detectors import MonoDetector


def extract_fnames(folder, f_ext):
    fnames = sorted([x for x in os.listdir(folder) if x.endswith(f_ext)])
    return fnames


class Mono(PeakAnalysis):
    """
    Takes a folder containing image files from area detectors and cakes the 
    data while associating it with spatial information. The caked data can then
    be analysed (peak_fit/strain calculations).
    """

    def __init__(self, folder, co_ords, detector, wavelength=None,
                 f_ext='.edf', progress=True, npt_rad=1024, npt_az=36,
                 az_range=(-180, 180)):
        """
        # folder:     Folder containing the image files for analysis
        # co_ords:   1D/2D/3D numpy array containing data point co_ords
        # params:     Accepts a file location for a .poni parameter file 
                      produced by pyFAI-calib. Alternative directly enter 
                      calibration details as take from Fit2D in form:
                      (sample_to_detector (mm), centre_x (pix), centre_y (pix), 
                      tilt (deg), tilt_plane (deg), pixel_size_x (micron), 
                      pixel_size_y (micron))
        # npt_rad:    Number of radial bins, should equal half detector pix
        # npt_az:     Number of azimuthal wedges           
        # az_range:   Range of azimuthal values to investigate - note that 0
                      degrees is defined at the eastern edge of the circle.
        """
        fname = '{}.h5'.format(os.path.split(folder)[1])
        self.fpath = os.path.join(folder, fname)

        error = 'Azimuthal range must be less than or equal to 360deg'
        assert abs(np.max(az_range) - np.min(az_range)) <= 360, error

        # Allow for use of pyFAI or Fit2D detector params
        # Currently checking if folder string - perhaps not the best method!
        if isinstance(detector, ("".__class__, u"".__class__)):
            ai = pyFAI.load(detector)  # CORRECT??
            # check if wavelength exists?
        else:
            ai = pyFAI.AzimuthalIntegrator()
            ai.setFit2D(*detector)
            ai.set_wavelength(wavelength)
        fnames = extract_fnames(folder, f_ext)

        error = ('Number of positions not equal to number of files (pos = %s,'
                 ' files = %s)' % (co_ords.shape[0], len(fnames)))
        assert co_ords.shape[0] == len(fnames), error
        (self.d1, self.d2, self.d3), self.dims = dim_fill(co_ords)
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
        self.E, self.v, self.G, self.stress_state = None, None, None, None
        self.analysis_state = 'integrated'
        # Temporary - extract from ai!
        self.detector = MonoDetector((2000,2000), 0.1, 1000, 100, 1)