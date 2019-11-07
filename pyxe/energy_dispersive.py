# -*- coding: utf-8 -*-
"""
@author: casimp
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re

import h5py
import numpy as np

from pyxe.data_io import dimension_fill, dim_fill
from pyxe.peak_analysis import PeakAnalysis
from pyxpb.detectors import i12_energy


class EDI12(PeakAnalysis):

    def __init__(self, fpath, unused_detector=23, phi=None):
        """ Extract useful data from raw .nxs file.

        Deletes data from unused  detector. Allows definition of az_angle (phi)
        if the unused detector is not 23.  Prepares the file for peak/profile
        analysis.

        Args:
            fpath (str): Path to NeXus file
            unused_detector (int): Unused detector (normally 23)
            phi (ndarray, list): Re-define phi if detector order is mixed
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
        self.T = None
        

        # Remove unused detector - resulting detector array is almost certainly
        # arrayed in ascending order from from -np.pi to 0 (phi). Option exists
        # to specify the order if this isn't true.
        self.q = f['entry1/EDXD_elements/edxd_q'][()]
        self.q = np.delete(self.q, unused_detector, 0)
        self.I = f['entry1/EDXD_elements/data'][()]
        self.I = np.delete(self.I, unused_detector, -2)
        self.phi = np.linspace(-np.pi, 0, 23) if phi is None else phi
        self.analysis_state = 'integrated'
        self.detector = i12_energy()
        
        
        
def mca_strip(f, d=27, p=14):
    with open(f,'r') as t:
        lines = t.readlines()
        data = lines[d:]
        y = [float(s) for s in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', lines[p])][-1]
        z = [float(s) for s in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', lines[p + 1])][1]
        d = [[float(s) for s in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', line)] for line in data]
        d = [val for sublist in d for val in sublist]
         
    return (y, z), d


def mca_array(directory='.', fend='mca'):
    
    fnames = sorted([i for i in os.listdir(directory) if i[-3:] == fend])
    
    
    #fnames = [i for i in fnames if  int(re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', i)[0]) == 0]
    
    fnames = [i for i in fnames if  int(i[i.find('xia') + 3:i.find('xia') + 5]) == 0]

    
    x, y = [], []
    data = []

    for idx, f in enumerate(fnames):

        try:
            f_0, f_90 = f, f.replace('00', '01', 1)
            (y0, z0), d0 = mca_strip(os.path.join(directory, f_0))
            (y1, z1), d1 = mca_strip(os.path.join(directory, f_90))
            assert (y0, z0) == (y1, z1), 'Point position not same for det 0/1'

            x.append(y0)
            y.append(z0)
            data.append([d0, d1])

        except:
            print(f_0, f_90)
            print('File (probably) not found for both detectors')

    y, z = np.array(x), np.array(y)
    co_ords = np.vstack((y, z)).swapaxes(0, 1)
    #print(np.array(data).shape)
    data = np.array(data)
    
    return co_ords, data


h = 6.62607004 * 10 ** -34
c = 2.99792458 * 10 ** 8
eV = 1.60218 * 10 ** -19


def e_to_w(energy):
    """ Takes photon energy (keV) -> returns wavelength (m) """
    energy_j = np.array(energy) * 1000 * eV
    wavelength = h * c / energy_j
    return wavelength

def e_to_q(energy, two_theta):
    """ Takes energy (keV) and 2theta (rad) -> returns q (A-1) """
    wavelength = e_to_w(energy)
    q_per_m = np.sin(two_theta / 2) * 4 * np.pi / wavelength
    q_per_a = q_per_m / (10**10)  # convert to A^-1
    return q_per_a


#
#
#def w_to_e(wavelength):
#    """ Takes wavelength (m) -> returns photon energy (keV) """
#    energy_j = h * c / np.array(wavelength)
#    energy_kev = energy_j / (eV * 1000)
#    return energy_kev
#



class EDID15(PeakAnalysis):

    def __init__(self, folder, fend='mca', fname=None, phi=[0, np.pi/2]):
        """
        Takes a folder containing .mca data files from ED detectors and merges
        the data while associating it with spatial information. The I data
        can then be analysed (peak_fit/strain calculations).

        Args:
            folder (str): Folder containing the files for analysis

        """
        fname = '{}.h5'.format(os.path.split(folder)[1])
        self.fpath = os.path.join(folder, fname)
        self.phi = np.array(phi)
        
        co_ords, self.I = mca_array(folder, fend)                            
        (self.d1, self.d2, self.d3), self.dims = dim_fill(co_ords)
        self.ndim = len(self.dims)
        
        # Not sure how data will be presented, but this can be remedied.
        E = np.linspace(0, 4095, 4096)
        q1 = e_to_q(0.07 + 0.07493 * E + 4.422e-8 * E, 5.10541851 * np.pi/180)
        q2 = e_to_q(0.1453 + 0.07483 * E + 9.026e-8 * E, 4.9718601 * np.pi/180)
        #E = np.vstack((q1, q2))

        
        
        self.q = np.vstack((q1, q2))
        # Create az_slice 'specific' q values - to work with edxd data
        
        #self.q = np.repeat(q[None, :], self.phi.size, axis=0)
        self.analysis_state = 'integrated'
        # Temporary - extract from ai!
        self.detector = i12_energy()


#folder = os.path.expanduser(r'~/Dropbox/Python/nc_ni/')
#test = EDID15(folder)