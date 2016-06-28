# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casim
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py
import numpy as np

from pyxe.plotting import StrainPlotting
from pyxe.strain_tools import StrainTools
from pyxe.merge_tools import find_limits, mask_generator, masked_merge


class MergeIntensity(object):
    """
    Tool to merge q v I datasets for multi-scan acquisitions e.g. high
    resolution plus low resolution scans.

    This is the recommended tool to use for merging! MergeStrain to be added
    as a solution for situations in which there are changes in setup
    (e.g. this may cause a slight variation in q0 wrt. phi)
    """
    def __init__(self, data, order=None, padding=0.1):
        """
        Merge data, specifying mering method/order
        
        # data:       Tuple or list containing data objects analysed with the 
                      XRD_analysis tool.
        # name:       Experiment name/ID.
        # order:      Merging method/order. Specify 'simple' merge (keeps all 
                      data) or by user defined order. User defined order allows 
                      for the supression/removal of overlapping data. User 
                      defined should be a list of the same length as the data 
                      tuple.
        """
        self.data = np.array(data)

        for i in self.data:
            error = 'Trying to merge incompatible data - %s'
            assert self.data[0].n_dims == i.n_dims, error % 'e.g. 2D with 3D'
            assert self.data[0].phi == i.phi, error % 'diff number of az bins'
            assert self.data[0].q == i.q, error % 'diff number of q bins'

        self.q = self.data[0].q
        self.phi = self.data[0].phi
        self.n_dims = self.data[0].n_dims

        # Merge priority order - either keep all data or delete overlapping
        # regions (e.g. high resolution scan on top of low resolution)
        priority = [0 for data_ in self.data] if order is None else order

        # Determines the number of different priority levels and the data
        # inidices for each set
        priority_set, inds = np.unique(priority, return_inverse=True)

        data_mask = [self.data[inds == 0],  [None] * len(self.data[inds == 0])]
        
        for idx, _ in enumerate(priority_set[1:]):
            mask_gen = self.data[inds < idx + 1]
            mask_data = self.data[inds == idx + 1]
            limits = []
            for dim in mask_gen[0].dims:
                limits.append(find_limits([i.co_ords[dim] for i in mask_gen]))

            data_mask[0] = np.append(data_mask[0], mask_data)
            data_mask[1] += [mask_generator(data_, limits, padding) 
                             for data_ in mask_data]

        merged_data = masked_merge(data_mask[0], data_mask[1])

        self.d1, self.d2, self.d3, I = merged_data

    def save_to_nxs(self, fpath=None, overwrite=False):
        """
        Saves all data back into an expanded .nxs file. Contains all original
        data plus q0, peak locations and strain.

        # fpath:      Abs. path for new file - default is to save to parent
                      directory (*_pyxe.nxs)
        # overwrite:  Overwrite file if it already exists (True/[False])
        """
        if fpath is None:
            fpath = '%s_pyxe.nxs' % os.path.splitext(self.fpath)[0]

        data_array = (self.d1, self.d2, self.d3, self.phi,
                      self.q, self.I, self.n_dims)

        pyxe_to_nxs(fpath, data_array, overwrite)


class MergeStrain(object):
    """
    Tool to merge analyzed strain data from multi-scan acquisitions e.g. high
    resolution plus low resolution scans.

    Recommended to use MergeIntensity where possible! MergeStrain only
    preferable in situations in which peak data is incomparable (e.g. change of
    setup may alter q0 wrt phi).
    """
    def __init__(self, data, order=None, padding=0.1):
        """
        Merge data, specifying mering method/order

        # data:       Tuple or list containing data objects analysed with the
                      XRD_analysis tool.
        # name:       Experiment name/ID.
        # order:      Merging method/order. Specify 'simple' merge (keeps all
                      data) or by user defined order. User defined order allows
                      for the supression/removal of overlapping data. User
                      defined should be a list of the same length as the data
                      tuple.
        """
        self.data = np.array(data)

        for i in self.data:
            error = 'Trying to merge incompatible data - %s'
            assert self.data[0].n_dims == i.n_dims, error % 'e.g. 2D with 3D'
            assert self.data[0].phi == i.phi, error % 'diff number of az bins'
            assert self.data[0].q == i.q, error % 'diff number of q bins'

        self.q = self.data[0].q
        self.phi = self.data[0].phi
        self.n_dims = self.data[0].n_dims

        # Merge priority order - either keep all data or delete overlapping
        # regions (e.g. high resolution scan on top of low resolution)
        priority = [0 for data_ in self.data] if order is None else order

        # Determines the number of different priority levels and the data
        # inidices for each set
        priority_set, inds = np.unique(priority, return_inverse=True)

        data_mask = [self.data[inds == 0], [None] * len(self.data[inds == 0])]

        for idx, _ in enumerate(priority_set[1:]):
            mask_gen = self.data[inds < idx + 1]
            mask_data = self.data[inds == idx + 1]
            limits = []
            for dim in mask_gen[0].dims:
                limits.append(find_limits([i.co_ords[dim] for i in mask_gen]))

            data_mask[0] = np.append(data_mask[0], mask_data)
            data_mask[1] += [mask_generator(data_, limits, padding)
                             for data_ in mask_data]

        merged_data = masked_merge(data_mask[0], data_mask[1])

        self.d1, self.d2, self.d3, self.I, self.strain, \
        self.strain_err = merged_data