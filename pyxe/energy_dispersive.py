# -*- coding: utf-8 -*-
"""
@author: casimp
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

import h5py
import numpy as np
import os

from pyxe.analysis_tools import dimension_fill
from pyxe.peak_analysis import PeakAnalysis


class EDI12(PeakAnalysis):
    """
    Takes an un-processed .nxs file from the I12 EDXD detector and fits curves
    to all specified peaks for each detector. Calculates strain and details
    associated error. 
    """
   
    def __init__(self, fpath, unused_detector=23, phi=None, progress=True):
        """
        Extract useful data from raw .nxs file. Removes data from unused 
        detector. Allows definition of az_angle (phi) if the unused detector is
        not 23.
        """
        self.fpath = fpath
        f = h5py.File(fpath, 'r')

        # Use scan command to find number of dimensions and the order in which
        # they were acquired. Important/useful for plotting!
        scan_command = f['entry1/scan_command'][()][0]
        dims = re.findall(b'ss2_\w+', scan_command)
        self.ndim = len(dims)
        all_dims = [b'ss2_x', b'ss2_y', b'ss2_z']
        dims = dims + [dim for dim in all_dims if dim not in dims]
        co_ords = []
        for dim in dims:
            co_ords.append(dimension_fill(f, dim.decode("utf-8")))
        self.d1, self.d2, self.d3 = co_ords

        # Remove unused detector - resulting detector array is almost certainly
        # arrayed in ascending order from from -np.pi to 0 (phi). Option exists
        # to specify the order if this isn't true.
        self.q = f['entry1/EDXD_elements/edxd_q'][()]
        self.q = np.delete(self.q, unused_detector, 0)
        self.I = f['entry1/EDXD_elements/data'][()]
        self.I = np.delete(self.I, unused_detector, -2)
        self.phi = np.linspace(-np.pi, 0, 23) if phi is None else phi
        self.E, self.v, self.G, self.stress_state = None, None, None, None
        self.analysis_state = 'integrated'
