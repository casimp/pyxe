# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 08:34:51 2015

@author: Chris
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from scipy.optimize import curve_fit
import numpy as np
import h5py
from peak_fitting import gaussian, lorentzian, psuedo_voigt


def p0_approx(data, peak_window, func = gaussian):
    """
    Approximates p0 for curve fitting.
    Now estimates stdev.
    """
    x, y = data 
    
    if x[0] > x[1]:
        x = x[::-1]
        y = y[::-1]

    peak_ind = np.searchsorted(x, peak_window)
    x_ = x[peak_ind[0]:peak_ind[1]]
    I_ = y[peak_ind[0]:peak_ind[1]]
    max_index = np.argmax(I_)
    HM = min(I_) + (max(I_) - min(I_)) / 2
    stdev = x_[max_index + np.argmin(I_[max_index:] > HM)] - x_[max_index]
    if stdev <= 0:
        stdev = 0.1
    p0 = [min(I_), max(I_) - min(I_), x_[max_index], stdev]
    
    if func == psuedo_voigt:
        p0.append(0.5)

    return p0


def peak_fit(data, window, p0 = None, func = gaussian):
    """
    Takes a function (guassian/lorentzian) and information about the peak 
    and calculates the peak location and standard deviation. 
    """
    if data[0][0] > data[0][-1]:
        data[0] = data[0][::-1]
        data[1] = data[1][::-1]
 
    if p0 == None:
        p0 = p0_approx(data, window, func)
        
    peak_ind = np.searchsorted(data[0], window)
    x_ = data[0][peak_ind[0]:peak_ind[1]]
    I_ = data[1][peak_ind[0]:peak_ind[1]]

    coeff, var_matrix = curve_fit(func, x_, I_, p0)
    perr = np.sqrt(np.diag(var_matrix))
            
    return [coeff[2], perr[2]], coeff
    
    
def q0_analysis(folder, fnum, window, detector = 11):
    
    fname = '%d.nxs' % fnum
    f = h5py.File(folder + fname, 'r')
    group = f['entry1']['EDXD_elements']

    x_shape, y_shape = np.shape(group['ss2_x'])

    peaks = np.zeros(np.shape(group['ss2_x']))
    stdevs = np.zeros(np.shape(group['ss2_x']))
    
    for (x, y), _ in np.ndenumerate(peaks):
    
        data = (group['edxd_q'][detector], group['data'][x, y, detector])
        p0 = p0_approx(data, window)
    
        try:
            peak, stdev = peak_fit(data, window, p0, gaussian)[0]
            peaks[x, y] = peak
            stdevs[x, y] = stdev
        except RuntimeError:
            print('Peak not found at index (%d, %d)' % (x, y))
            peaks[x, y] = np.nan
            stdevs[x, y] = np.nan
    
        if stdev / peak > 10 **-3:
            print('Error too great for peak at index (%d, %d)' % (x, y))
            peaks[x, y] = np.nan
            stdevs[x, y] = stdev
    
    x = group['ss2_x']
    y = group['ss2_y']
    
    np.savetxt(r'./analysis/' + fname[:-4] + '_' + str(detector) + '.txt', [peaks.flatten(), stdevs.flatten(), x[:].flatten(), y[:].flatten()])
    np.savetxt(r'./analysis/q0_' + str(detector) + '.txt', [peaks.flatten(), stdevs.flatten(), x[:].flatten(), y[:].flatten()])
    f.close()
        
    return np.mean(peaks), np.std(peaks)
    
    
def array_fit(q, I, q0, window, func):
        
    peaks = np.zeros(np.shape(I[:, :, 0]))
    stdevs = np.zeros(np.shape(I[:, :, 0]))

    for (x, y), _ in np.ndenumerate(peaks):
    
        data = (q, I[x, y])
        p0 = p0_approx(data, window, func)
        
        try:
            peak, stdev = peak_fit(data, window, p0, func)[0]
            peaks[x, y] = peak
            stdevs[x, y] = stdev
        except RuntimeError:
            print('Peak not found at index (%d, %d)' % (x, y))
            peaks[x, y] = np.nan
            stdevs[x, y] = np.nan
        if stdev / peak > 1 * 10 **-4:
            print('Error too great for peak at index (%d, %d)' % (x, y))
            peaks[x, y] = np.nan
            stdevs[x, y] = np.nan
            
        if abs((peak - q0) / q0) > 6e-3:
            print('Strain outside bounds at index (%d, %d)' % (x, y))
            peaks[x, y] = np.nan
            stdevs[x, y] = np.nan
    
    return(peaks, stdevs)
    


