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
    
    
def window_optimize(folder, fnum, q0, delta_q = 0.01, steps = 100, 
                    detector = 11):
          
        
    fname = '%d.nxs' % fnum
    f = h5py.File(folder + fname, 'r')
    group = f['entry1']['EDXD_elements']

    windows = np.zeros(steps)    
    errors = np.zeros(steps)    
    
    for idx, i in enumerate(range(1, steps + 1)):
        
        window = [q0 - delta_q * i, q0 + delta_q * i]
        print(window)
        
        x, y, = 0, 0
        
        data = (group['edxd_q'][detector], group['data'][x, y, detector])
        p0 = p0_approx(data, window)
        
        try:
            peak, stdev = peak_fit(data, window, p0, gaussian)[0]
            errors[idx] = stdev
            
        except RuntimeError:
            print('Peak not found at index (%d, %d)' % (x, y))
            errors[idx] = np.nan
            
        windows[idx] = delta_q * i    
        
    return windows, errors


                
def array_fit(q_array, I_array, peak_window, func, unused_detectors = [23], 
              error_limit = 1 * 10 **-4):
        
    peaks = np.zeros(I_array.shape[:-1]) * np.nan
    stdevs = np.zeros(I_array.shape[:-1]) * np.nan
    
    detectors = [i for i in range(q_array.shape[0])]
    try:
        for detector in unused_detectors:
            detectors.remove(detector)
    except ValueError:
        detectors = [i for i in range(q_array.shape[0])]
        print('Unused detectors invalid. Analyzing all detectors.')

    for detector in detectors:
        "Load in detector calibrated q array"
        q = q_array[detector]
        for position in np.ndindex(I_array.shape[:-2]):
            index = position + (detector,)
            "Select 4096 count intensity vector to analyze"
            I = I_array[index]
            p0 = p0_approx((q, I), peak_window, func)
            "Fit peak across window"
            try:
                peak, stdev = peak_fit((q, I), peak_window, p0, func)[0]
                if stdev / peak > error_limit:
                    print('Error too great for peak at index %s' % (index, ))
                    peaks[index] = np.nan
                    stdevs[index] = np.nan
                else:
                    peaks[index] = peak
                    stdevs[index] = stdev
            except RuntimeError:
                print('Peak not found at index %s' % (index, ))
                peaks[index] = np.nan
                stdevs[index] = np.nan

    return(peaks, stdevs)
    

def q0_analysis(fname, q0_approx, window = 0.25, func = gaussian):
    
    f = h5py.File(fname, 'r')
    group = f['entry1']['EDXD_elements']
    q, I = group['edxd_q'], group['data']

    q_type = type(q0_approx)
    if q_type == int or q_type == float or q_type == np.float64:
        q0_approx = [q0_approx]    
    
    peak_windows = np.array([[q - window/2, q + window/2] for q in q0_approx])
    
    # Create empty arrays to store peaks and their stdevs 
    array_shape = I.shape[:-1] + (len(q0_approx),)
    peaks = np.nan * np.ones(array_shape)
    peaks_err = np.nan * np.ones(array_shape)
    
    # Iterate across q0 values and fit peaks for all detectors
    for idx, window in enumerate(peak_windows):
        peaks[..., idx], peaks_err[..., idx] = array_fit(q, I, window, func)
        
    
    q0_mean = np.nan * np.ones(peaks.shape[-2:])
    q0_mean_err = np.nan * np.ones(peaks.shape[-2:])
    for detector, q_app in np.ndindex(q0_mean.shape):
        q0_mean[detector, q_app] = np.mean(peaks[..., detector, q_app])
        q0_mean_err[detector, q_app] = np.mean(peaks_err[..., detector, q_app])

    return q0_mean, q0_mean_err