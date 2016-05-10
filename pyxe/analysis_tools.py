# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casimp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def dim_fill(data):

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
    return co_ords, dims
    

def mirror_data(phi, data):
    # has to be even number of slices but uneven number of boundaries.
    angles = phi[:int(phi[:].shape[0]/2)]
    peak_shape = data.shape
    phi_len = int(peak_shape[-2]/2)
    new_shape = (peak_shape[:-2] + (phi_len, ) + peak_shape[-1:])
    d2 = np.nan * np.zeros(new_shape)
    for i in range(phi_len):
        d2[:, i] = (data[:, i] + data[:, i + new_shape[-2]]) / 2
    return angles, d2
    

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


def scrape_slits(data):
    try:        
        slit_size = data['entry1/before_scan/s4/s4_xs'][0]
    except KeyError:
        slit_size = []   
    return slit_size
