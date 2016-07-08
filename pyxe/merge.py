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
from pyxe.merge_tools import find_limits, mask_generator, masked_merge


def merge(pyxe_objects, order=None, padding=0.1):
    data = np.array(pyxe_objects)

    for i in data:
        error = 'Trying to merge incompatible data - %s'
        assert data[0].n_dims == i.n_dims, error % 'e.g. 2D with 3D'
        assert data[0].phi == i.phi, error % 'diff number of az bins'
        assert data[0].q == i.q, error % 'diff number of q bins'

    q = data[0].q
    phi = data[0].phi
    n_dims = data[0].n_dims

    # Merge priority order - either keep all data or delete overlapping
    # regions (e.g. high resolution scan on top of low resolution)
    priority = [0 for data_ in data] if order is None else order

    # Determines the number of different priority levels and the data
    # inidices for each set
    priority_set, inds = np.unique(priority, return_inverse=True)

    data_mask = [data[inds == 0],  [None] * len(data[inds == 0])]

    for idx, _ in enumerate(priority_set[1:]):
        mask_gen = data[inds < idx + 1]
        mask_data = data[inds == idx + 1]
        limits = []
        for dim in mask_gen[0].dims:
            limits.append(find_limits([i.co_ords[dim] for i in mask_gen]))

        data_mask[0] = np.append(data_mask[0], mask_data)
        data_mask[1] += [mask_generator(data_, limits, padding)
                         for data_ in mask_data]

    merged_data = masked_merge(data_mask[0], data_mask[1])

    d1, d2, d3, I = merged_data