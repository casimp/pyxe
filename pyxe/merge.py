# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casim
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import numpy as np
import copy
from pyxe.command_parsing import analysis_state_comparison


def state_compare(analysis_states):
    """
    Will return at earliest opportunity
    :param analysis_states:
    :return:
    """
    states = ['integrated', 'peaks', 'strain', 'strain fit',
              'stress', 'stress fit']
    for state in states:
        if state in analysis_states:
            return state


def extract_limits(data):

    d1_lim = [np.min(data.d1), np.max(data.d1)]
    d2_lim = [np.min(data.d2), np.max(data.d2)]
    d3_lim = [np.min(data.d3), np.max(data.d3)]

    return d1_lim, d2_lim, d3_lim


def remove_data(data, limit):

    mask = np.ones_like(data.d1, dtype='bool')
    for lim, d in zip(limit, [data.d1, data.d2, data.d3]):
        try:
            mask = np.logical_and(mask, np.logical_and(d > lim[0], d < lim[1]))
        except TypeError:
            pass
    all_data = [data.d1, data.d2, data.d3, data.I, data.peaks, data.peaks_err,
                data.fwhm, data.fwhm_err, data.strain, data.strain_err,
                data.strain_tensor]
    for d in all_data:
        d = d[mask] if d is not None else d

    return data # potentially not needed (change in place)


def none_merge(data, current_state, required_state, axis=-1):
    if analysis_state_comparison(current_state, required_state):
        assert not any([d is None for d in data]), 'Invalid data (None types)'
        if axis is None:
            merged_data = np.append(*data)
        else:
            reshaped_data = [d.reshape(axis, d.shape[axis]) for d in data]
            merged_data = np.concatenate(reshaped_data)
        return merged_data
    else:
        return None


def basic_merge(data):
    """
    Flat merge two pyxe data object together

    :param pyxe_objects:
    :return:
    """
    new = copy.deepcopy(data[0])
    if len(data) == 1:
        return new

    for d in data:
        error = 'Trying to merge incompatible data - %s'
        assert new.n_dims == d.n_dims, error % 'e.g. 2D with 3D'
        assert new.phi == d.phi, error % 'different number of az bins'
        assert new.q == d.q, error % 'different number of q bins'

    new.d1 = np.append([d.d1 for d in data]) if new.n_dims > 0 else None
    new.d2 = np.append([d.d2 for d in data]) if new.n_dims > 1 else None
    new.d3 = np.append([d.d3 for d in data]) if new.n_dims > 2 else None

    state = state_compare([d.analysis_state for d in data])

    for i in data:
        valid = all(new.E == i.E, new.v == i.v, new.G == i.G,
                    new.stress_state == i.stress_state)
        if not valid:
            print('Material properties inconsistent - resetting all.')
            new.E, new.v, new.G, new.stress_state = None, None, None, None
            state = state.replace('stress', 'strain')

    new.analysis_state = state

    new.I = none_merge([d.I for d in data], state, 'intensity', axis=-2)
    new.peaks = none_merge([d.peaks for d in data], state, 'peaks')
    new.peaks_err = none_merge([d.peaks_err for d in data], state, 'peaks')
    new.fwhm = none_merge([d.fwhm for d in data], 'peaks', state)
    new.fwhm_err = none_merge([d.fwgm_err for d in data], state, 'peaks')
    new.strain = none_merge([d.strain for d in data], state, 'strain')
    new.strain_err = none_merge([d.strain_err for d in data], state, 'strain')
    strain_tensor = [d.strain_tensor for d in data]
    new.strain_tensor = none_merge(strain_tensor, state, 'strain fit')

    return new


def ordered_merge(data, order=None, padding=0.1):

    if order is None:
        return basic_merge(data)

    # use default dict to group key/value pairs
    order_data = defaultdict(zip(order, data))

    # merge all data of same order level
    keys = sorted(order_data.keys())
    basic_merged = []
    for key in keys:
        basic_merged.append(basic_merge(order_data[key]))

    # merge levels
    new = basic_merged[0]
    for idx in range(len(basic_merged) - 1):
        limits = extract_limits(new)
        new = basic_merge(new, remove_data(basic_merged[idx], limits))

    return new
