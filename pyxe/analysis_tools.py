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


def full_ring_fit(strain, phi):
    """
    Fits the strain transformation equation to the strain information from each
    azimuthal slice.
    """
    strain_tensor = np.nan * np.ones(strain.shape[:-1] + (3,))

    error_count = 0
    for idx in np.ndindex(strain.shape[:-1]):
        data = strain[idx]
        not_nan = ~np.isnan(data)

        if phi[not_nan].size > 2: # Nope use sampling theory to define minimum
            # Estimate curve parameters
            p0 = [np.nanmean(data), 3 * np.nanstd(data) / (2 ** 0.5), 0]
            try:
                a, b = curve_fit(strain_transformation,
                                 phi[not_nan], data[not_nan], p0)
                strain_tensor[idx] = a
            except (TypeError, RuntimeError):
                error_count += 1
        else:
            error_count += 1
    print('\nUnable to fit full ring (%s) data %i out of %i points'
          % (name, error_count, np.size(strain[..., 0])))

    return strain_tensor


def pyxe_to_nxs(fname, pyxe_object, overwrite=False):
    """
    Saves all data back into an expanded .nxs file. Contains all original
    data plus q0, peak locations and strain.

    # fname:      File name/location - default is to save to parent
                  directory (*_pyxe.nxs)
    """
    data_ids = ['d1', 'd2', 'd3', 'phi', 'q', 'I', 'n_dims']
    data_array = [pyxe_object.d1, pyxe_object.d2, pyxe_object.d3,
                  pyxe_object.phi, pyxe_object.q, pyxe_object.I,
                  pyxe_object.n_dims]

    if pyxe_object.analysis_stage > 0:
        data_ids += ['peaks', 'peaks_err', 'fwhm', 'fwhm_err']
        data_array += [pyxe_object.peaks, pyxe_object.peaks_err,
                       pyxe_object.fwhm, pyxe_object.fwhm_err]
    if pyxe_object.analysis_stage == 2:
        data_ids += ['strain', 'strain_err', 'strain_tensor', 'q0']
        data_array += [pyxe_object.strain, pyxe_object.strain_err,
                       pyxe_object.strain_tensor]

    write = 'w' if overwrite else 'w-'
    with h5py.File(fname, write) as f:

        for data_id, data in zip(data_ids, data_array):
            d_path = 'entry1/pyxe_simplified/%s' % data_id

            if data_id == 'I':
                f.create_dataset(d_path, data=data, compression='gzip')
            else:
                f.create_dataset(d_path, data=data)


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
