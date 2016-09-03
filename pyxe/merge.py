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


def lowest_state(analysis_states):
    """ Returns lowest analysis state from list of states
    Args:
        analysis_states (tuple, list): List of analysis states
    Returns:
        str: Minimum analysis state
    """
    states = ['integrated', 'peaks', 'strain', 'strain fit',
              'stress', 'stress fit']
    for state in states:
        if state in analysis_states:
            return state


def extract_limits(data):
    """ Finds limits for data cube.
    Args:
        data: pyxe data object
    Returns:
        tuple: d1_lim (min, max), d2_lim (min, max), d3_lim (min, max)
    """
    d1_lim = [np.min(data.d1), np.max(data.d1)]
    d2_lim = [np.min(data.d2), np.max(data.d2)]
    d3_lim = [np.min(data.d3), np.max(data.d3)]

    return d1_lim, d2_lim, d3_lim


def remove_data(data, limit):
    """ Crop pyxe data object according to positional limits.

    Args:
        data: pyxe data object
        limit (tuple): d1_lim, d2_lim, d3_lim - cube limits for data removal

    Returns:
        pyxe data object: Cropped pyxe object
    """
    mask = np.ones_like(data.d1, dtype='bool')
    for lim, d in zip(limit, [data.d1, data.d2, data.d3]):
        try:
            # Checking whether points lie WITHIN limits (i.e. to be removed)
            mask_temp = np.logical_and(d >= lim[0], d <= lim[1])
            # Must be within limits wrt. all dimensions
            mask = np.logical_and(mask, mask_temp)
        except TypeError:
            pass

    # Creating a mask, so inverse (not) of values calculated
    mask = np.logical_not(mask)

    all_data = ['d1', 'd2', 'd3', 'I', 'peaks', 'peaks_err', 'fwhm',
                'fwhm_err', 'strain', 'strain_err', 'strain_tensor']

    for name in all_data:
        d = getattr(data, name)
        masked = d[mask] if d is not None else d
        setattr(data, name, masked)

    return data  # potentially not needed (change in place)


def none_merge(data, current_state, required_state, axis=-1):
    """ Merge data if present/exists else return None

    Args:
        data (list): List of ndarrays
        current_state (str): Current state for comparison
        required_state (str): Required state for comparison
        axis (int): Axis for data concatenation

    Returns:
        ndarray: Merged data (or None)
    """
    try:
        analysis_state_comparison(current_state, required_state)
        assert not any([d is None for d in data]), 'Invalid data (None types)'
        if axis is None:
            merged_data = np.append(*data)
        else:
            reshaped_data = [d.reshape(axis, *d.shape[axis:]) for d in data]
            merged_data = np.concatenate(reshaped_data)
        return merged_data
    except AssertionError:
        return None


def basic_merge(data):
    """ Simple merge of multiple pyxe data objects (keep all data).

    Args:
        data (list): List of pyxe data objects

    Returns:
        pyxe data object: Merged pyxe object
    """
    new = copy.deepcopy(data[0])
    if len(data) == 1:
        return new

    for d in data:
        error = 'Trying to merge incompatible data - %s'
        assert new.ndim == d.ndim, error % 'e.g. 2D with 3D'
        assert np.array_equal(new.phi, d.phi), error % 'number of az bins'
        assert np.array_equal(new.q, d.q), error % 'number of q bins'

    new.d1 = np.concatenate([d.d1 for d in data]) if new.ndim > 0 else None
    new.d2 = np.concatenate([d.d2 for d in data]) if new.ndim > 1 else None
    new.d3 = np.concatenate([d.d3 for d in data]) if new.ndim > 2 else None

    state = lowest_state([d.analysis_state for d in data])

    for i in data:
        try:
            valid = all([new.E == i.E, new.v == i.v, new.G == i.G,
                        new.stress_state == i.stress_state])
            if not valid:
                print('Material properties inconsistent - resetting all.')
                new.E, new.v, new.G, new.stress_state = None, None, None, None
                state = state.replace('stress', 'strain')
        except AttributeError:
            pass

    new.analysis_state = state

    new.I = none_merge([d.I for d in data], state, 'integrated', axis=-2)
    new.peaks = none_merge([d.peaks for d in data], state, 'peaks')
    new.peaks_err = none_merge([d.peaks_err for d in data], state, 'peaks')
    new.fwhm = none_merge([d.fwhm for d in data], state, 'peaks')
    new.fwhm_err = none_merge([d.fwhm_err for d in data], state, 'peaks')
    new.strain = none_merge([d.strain for d in data], state, 'strain')
    new.strain_err = none_merge([d.strain_err for d in data], state, 'strain')
    strain_tensor = [d.strain_tensor for d in data]
    new.strain_tensor = none_merge(strain_tensor, state, 'strain fit')

    return new


def ordered_merge(data, order=None, pad=0.01):
    """ Ordered merge - keep/reject overlapping data based on order.

    Args:
        data (list): List of pyxe data objects
        order (list, tuple): Priority list to keep/reject the overlapping data
        pad (float): For overlapping data apply a pad to separate

    Returns:
        pyxe data object: Merged pyxe object
    """
    if order is None:
        return basic_merge(data)

    # use default dict to group key/value pairs
    order_data = defaultdict(list)
    for k, v in zip(order, data):
        order_data[k].append(v)

    # merge all data of same order level
    keys = sorted(order_data.keys())
    basic_merged = []
    for key in keys:
        basic_merged.append(basic_merge(order_data[key]))

    new = basic_merged[0]
    for idx in range(1, len(basic_merged)):
        limits = extract_limits(new)
        padded_limits = []
        for lim in limits:
            try:
                padded_limits.append([lim[0] - pad, lim[1] + pad])
            except TypeError:
                padded_limits.append([None, None])

        cropped_data = remove_data(basic_merged[idx], padded_limits)
        new = basic_merge([new, cropped_data])

    return new
