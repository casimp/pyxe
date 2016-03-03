# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:51:07 2015

@author: Chris
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def find_limits(data_array):
    """
    Finds max and min for from 
    """
    minimum = np.min([np.min(i) for i in data_array])
    maximum = np.max([np.max(i) for i in data_array])
    return [minimum, maximum]


def mask_generator(data, limits, padding = 0):
    """
    Returns a mask defining all points in a point cloud that are outside of a 
    masked square or cube.
    
    Pass in tuple/list containing n x nd arrays with co-ordinates of the points. 
    Limits contain the max and min for each axis (list of lists). 
    Allows for the specification of padding around the initial boundaries.
    
    >> mask_generator((x1, y1), [[xmin, xmax], [ymin, ymax]], 0.05)
    """
    #co_ords = (data.ss2_x, data.ss2_y, data.ss2_z)
    co_ords = [data.co_ords[x] for x in data.dims]
        
    mask = np.zeros(np.shape(co_ords[0]), dtype = bool)    
    
    for axis, limit in enumerate(limits):
        valid = (co_ords[axis] < (np.min(limit) - padding)) + \
                (co_ords[axis] > (np.max(limit) + padding))
        mask += valid
        
    return mask     

def masked_merge(unmerged_data, mask_array = None):
    """
    Pass in tuple with all data to be merged
    """
    if mask_array == None:
        mask_array = [None for i in unmerged_data]
    
    strain = []; strain_err = [];
    peaks = []; peaks_err = []
    ss2_x = []; ss2_y = []; ss2_z = []    
    strain_param = []
    I = []
    
    
    for data, mask in zip(unmerged_data, mask_array):
        shape = (data.peaks[..., 0, 0][mask].size, ) + data.peaks.shape[-2:]
        strain.append(data.strain[mask].reshape(shape))
        strain_err.append(data.strain_err[mask].reshape(shape))
        peaks.append(data.peaks[mask].reshape(shape))
        peaks_err.append(data.peaks_err[mask].reshape(shape))
        
        

        for ss2, posn in zip((ss2_x, ss2_y, ss2_z), 
                             (data.ss2_x, data.ss2_y, data.ss2_z)):
            try:        
                ss2.append(posn[mask].flatten())
            except (AttributeError, TypeError):
                pass
        shape2 = (data.strain_param[..., 0, 0][mask].size, ) + data.strain_param.shape[-2:]
        strain_param.append(data.strain_param[mask].reshape(shape2))
        
        shape3 = (data.I[..., 0, 0][mask].size, ) + data.I.shape[-2:]
        I.append(data.I[mask].reshape(shape3))

    
    merged_data = (np.vstack(I), np.vstack(strain), np.vstack(strain_err), 
                   np.vstack(strain_param), np.vstack(peaks), np.vstack(peaks_err))
                  
    for ss2 in [ss2_x, ss2_y, ss2_z]:
        merged_data += (np.concatenate(ss2), ) if ss2 !=[] else (None, )

    return merged_data    
    
        