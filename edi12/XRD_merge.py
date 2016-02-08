# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casim
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from edi12.XRD_analysis import *
from edi12.merge_tools import *


class XRD_merge(XRD_tools):
    """
    Tool to merge mutliple XRD data sets - inherits tools for XRD_tools.
    """
    def __init__(self, data, name, order = 'slit', padding = 0.1):
        """
        Merge data, specifying mering method/order
        
        # data:       Tuple or list containing data objects analysed with the 
                      XRD_analysis tool.
        # name:       Experiment name/ID.
        # order:      Merging method/order. Specify 'simple' merge (keeps all 
                      data) or by 'slit' size (default)/ user defined order.
                      Slit/user defined order allows for the supression/removal
                      of overlapping data. User defined should be a list of
                      the same length as the data tuple.
        """
        self.data = np.array(data)
        self.q0 = self.data[0].q0
        self.peak_windows = self.data[0].peak_windows
        
        # Check that you are merging similar data
        for i in self.data:
            assert self.data[0].dims == i.dims
        self.dims = self.data[0].dims
        
        if order == 'slit':
            priority = [data.slit_size for data in self.data]
        elif order == 'simple':
            priority = [0 for data in self.data]
        else:
            priority = order

        print(priority)
        priority_set, inds = np.unique(priority, return_inverse=True)    
        data_mask = [self.data[inds == 0],  [None] * len(self.data[inds == 0])]
        
        for idx, _ in enumerate(priority_set[1:]):
            idx += 1
            generate_mask_from = self.data[inds < idx]
            data_for_masking = self.data[inds == idx]
            
            x_lim = find_limits([i.ss2_x for i in generate_mask_from])
            y_lim = find_limits([i.ss2_y for i in generate_mask_from])
            
            if generate_mask_from[0].ss2_z == None:
                limits = [x_lim, y_lim]
            else:
                z_lim = find_limits([i.ss2_z for i in generate_mask_from])
                limits = [x_lim, y_lim, z_lim]

            data_mask[0] = np.append(data_mask[0], data_for_masking)
            data_mask[1] += [mask_generator(data, limits, padding) 
                             for data in data_for_masking]

        self.strain, self.strain_err, self.peaks, self.peaks_err, self.ss2_x, \
        self.ss2_y, self.ss2_z = masked_merge(data_mask[0], data_mask[1])
        self.co_ords = {b'ss2_x': self.ss2_x,b'ss2_y': self.ss2_y, 
                        b'self_z': self.ss2_z} 
        self.slit_size = None
        self.strain_fit()
        
    def strain_fit(self):
        """
        Fits a sin function to the 
        ***** Should ONLY Need this here - not in tools!
        """
        data_shape = self.strain.shape
        self.strain_param = np.nan * np.ones(data_shape[:-2] + \
                            (data_shape[-1], ) + (3, ))
        for idx in np.ndindex(data_shape[:-2] + (data_shape[-1],)):
            data = self.strain[idx[:-1]][:-1][..., idx[-1]]
            not_nan = ~np.isnan(data)
            angle = np.linspace(0, np.pi, 23)
            p0 = [np.nanmean(data), 3*np.nanstd(data)/(2**0.5), 0]
            try:
                a, b = curve_fit(cos_, angle[not_nan], data[not_nan], p0)
                self.strain_param[idx] = a
            except (TypeError, RuntimeError):
                print('Type or runtime error...')
        

    def __exit__(self, exc_type, exc_value, traceback):
        pass
 