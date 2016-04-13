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
import sys
from pyxe.fitting_functions import gaussian, lorentzian, psuedo_voigt


def p0_approx(data, peak_window, func = 'gaussian'):
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
    
    
    if func == 'psuedo_voigt':
        p0.append(0.5)
    return p0


#def peak_fit(data, window, p0 = [], func = 'gaussian'):
#    """
#    Takes a function (guassian/lorentzian) and information about the peak 
#    and calculates the peak location and standard deviation. 
#    """
#    func_dict = {'gaussian': gaussian, 'lorentzian': lorentzian, 
#                 'psuedo_voigt': psuedo_voigt}
#    func_name = func
#    func = func_dict[func.lower()]
#    
#    if data[0][0] > data[0][-1]:
#        data[0] = data[0][::-1]
#        data[1] = data[1][::-1]
#        
#    if p0 == []:
#        p0 = p0_approx(data, window, func_name)
#        
#    peak_ind = np.searchsorted(data[0], window)
#    x_ = data[0][peak_ind[0]:peak_ind[1]]
#    I_ = data[1][peak_ind[0]:peak_ind[1]]
#
#    coeff, var_matrix = curve_fit(func, x_, I_, p0)
#    perr = np.sqrt(np.diag(var_matrix))
#            
#    return [coeff[2], perr[2]], coeff


    
#def winwidth_optimize(fname, q0, delta_q = 0.01, steps = 100, 
#                     detectors = [0, 5, 11], point = (), func = 'gaussian'):
#          
#    f = h5py.File(fname, 'r')
#    group = f['entry1']['EDXD_elements']
#    ndims = group['data'][..., 0, 0].ndim
#
#    if point == ():
#        point = (0, ) * ndims
#        
#    error = 'Must define point with correct number of dimensions.'
#    assert ndims == len(point), error
#    assert isinstance(point, tuple), 'Defined point must be a tuple'  
#    
#    windows = np.zeros((steps, len(detectors)))    
#    errors = np.zeros((steps, len(detectors)))
#    
#    for didx, detector in enumerate(detectors):    
#    
#        for idx, i in enumerate(range(1, steps + 1)):
#            
#            window = [q0 - delta_q * i, q0 + delta_q * i]
#            data = (group['edxd_q'][detector], group['data'][point][detector])
#            p0 = p0_approx(data, window)
#            
#            try:
#                peak, stdev = peak_fit(data, window, p0, 'gaussian')[0]
#                errors[idx, didx] = stdev
#            except RuntimeError:
#                errors[idx, didx] = np.nan
#                
#            windows[idx, didx] = delta_q * i    
#
#    return windows, errors
#

#def array_fit(q_array, I_array, peak_window, func = 'gaussian', error_limit = 10 **-4, 
#              output = 'verbose', unused_detectors = [23]):
#        
#    peaks = np.zeros(I_array.shape[:-1]) * np.nan
#    stdevs = np.zeros(I_array.shape[:-1]) * np.nan
#    
#    detectors = [i for i in range(q_array.shape[0])]
#    try:
#        for detector in unused_detectors:
#            detectors.remove(detector)
#    except ValueError:
#        detectors = [i for i in range(q_array.shape[0])]
#        print('Unused detectors invalid. Analyzing all detectors.')
#
#    total_err_exceed = 0
#    total_run_error = 0
#
#    for idx, detector in enumerate(detectors):
#
#        err_exceed = 0
#        run_error = 0
#        # Load in detector calibrated q array
#        q = q_array[detector]
#        for position in np.ndindex(I_array.shape[:-2]):
#            index = position + (detector,)
#            I = I_array[index]
#            p0 = p0_approx((q, I), peak_window, func)
#            # Fit peak across window
#            try:
#                peak, stdev = peak_fit((q, I), peak_window, p0, func)[0]
#                if stdev / peak > error_limit:
#                    #print('Error too great for peak at index %s' % (index, ))
#                    peaks[index] = np.nan
#                    stdevs[index] = np.nan
#                    err_exceed += 1
#                else:
#                    peaks[index] = peak
#                    stdevs[index] = stdev
#            except RuntimeError:
#                #print('Peak not found at index %s' % (index, ))
#                peaks[index] = np.nan
#                stdevs[index] = np.nan
#                run_error += 1
#                
#            if output == 'simple':
#                sys.stdout.write("\rProgress: [{0:20s}] {1:.0f}%".format('#' * 
#                int(20*(idx + 1) / len(detectors)), 100*((idx + 1)/len(detectors))))
#                sys.stdout.flush()
#                
#        if output == 'verbose':
#            print('D%02d: Peak not found at %i positions, err_limit exceeded '
#                  '%i times.' % (detector, run_error, err_exceed))
#                  
#        total_err_exceed += err_exceed
#        total_run_error += run_error 
#         
#    if output != 'none':
#        print('\nTotal points: %i (23 detectors x %i positions)'
#          '\nPeak not found in %i position/detector combintions'
#          '\nError limit exceeded (or pcov not estimated) %i times' % 
#          (23*peaks.size/24, peaks.size/24, total_run_error, total_err_exceed))                  
#    
#    return(peaks, stdevs)



def peak_fit(data, window, p0 = [], func = 'gaussian'):
    """
    Takes a function (guassian/lorentzian) and information about the peak 
    and calculates the peak location and standard deviation. 
    """
    func_dict = {'gaussian': gaussian, 'lorentzian': lorentzian, 
                 'psuedo_voigt': psuedo_voigt}
    func_name = func
    func = func_dict[func.lower()]
    
    if data[0][0] > data[0][-1]:
        data[0] = data[0][::-1]
        data[1] = data[1][::-1]
        
    if p0 == []:
        p0 = p0_approx(data, window, func_name)
        
    peak_ind = np.searchsorted(data[0], window)
    x_ = data[0][peak_ind[0]:peak_ind[1]]
    I_ = data[1][peak_ind[0]:peak_ind[1]]

    return curve_fit(func, x_, I_, p0)


def array_fit(q_array, I_array, peak_window, func='gaussian', 
              error_limit=10 **-4, progress = True):
        
    data = [np.zeros(I_array.shape[:-1]) * np.nan for i in range(4)]
    peaks, peaks_err, fwhm, fwhm_err = data 
    
    detectors = [i for i in range(q_array.shape[0])]

    total_err_exceed = 0
    total_run_error = 0

    for idx, detector in enumerate(detectors):

        err_exceed = 0
        run_error = 0
        # Load in detector calibrated q array
        q = q_array[detector]
        for position in np.ndindex(I_array.shape[:-2]):
            index = position + (detector,)
            I = I_array[index]
            p0 = p0_approx((q, I), peak_window, func)
            # Fit peak across window
            try:
                coeff, var_matrix = peak_fit((q, I), peak_window, p0, func)
                perr = np.sqrt(np.diag(var_matrix))                
                
                peak, peak_err = coeff[2], perr[2]
                if func == 'gaussian':
                    fw, fw_err = coeff[3] * 2.35482, perr[3] * 2.35482
                elif func == 'lorentzian':
                    fw, fw_err = coeff[3] * 2, perr[3] * 2
                
                if peak_err / peak > error_limit:
                    err_exceed += 1
                else:
                    peaks[index] = peak
                    peaks_err[index] = peak_err
                    fwhm[index] = fw
                    fwhm_err[index] = fw_err
            except RuntimeError:
                run_error += 1
                
            if progress:
                sys.stdout.write("\rProgress: [{0:20s}] {1:.0f}%".format('#' * 
                int(20*(idx + 1) / len(detectors)), 100*((idx + 1)/len(detectors))))
                sys.stdout.flush()
                  
        total_err_exceed += err_exceed
        total_run_error += run_error 
         
    print('\nTotal points: %i (%i az_angles x %i positions)'
      '\nPeak not found in %i position/detector combintions'
      '\nError limit exceeded (or pcov not estimated) %i times' % 
      (peaks.size, peaks.shape[-2], peaks[..., 0, 0].size, total_run_error, total_err_exceed))                  
    
    return peaks, peaks_err, fwhm, fwhm_err
    
    

def q0_analysis(fname, q0_approx, window = 0.25, func = 'gaussian'):
    
    f = h5py.File(fname, 'r')
    group = f['entry1']['EDXD_elements']
    q, I = group['edxd_q'], group['data']

    q_type = type(q0_approx)
    if q_type == int or q_type == float or q_type == np.float64:
        q0_approx = [q0_approx]    
    
    peak_windows = np.array([[q_ - window/2, q_ + window/2] for q_ in q0_approx])
    
    # Create empty arrays to store peaks and their stdevs 
    array_shape = I.shape[:-1] + (len(q0_approx),)
    peaks = np.nan * np.ones(array_shape)
    peaks_err = np.nan * np.ones(array_shape)
    
    # Iterate across q0 values and fit peaks for all detectors
    for idx, window in enumerate(peak_windows):
        peaks[..., idx], peaks_err[..., idx], _, _ = array_fit(q, I, window, func)
        
    q0_mean = np.nan * np.ones(peaks.shape[-2:])
    q0_mean_err = np.nan * np.ones(peaks.shape[-2:])
    for detector, q_app in np.ndindex(q0_mean.shape):
        q0_mean[detector, q_app] = np.mean(peaks[..., detector, q_app])
        q0_mean_err[detector, q_app] = np.mean(peaks_err[..., detector, q_app])

    return q0_mean, q0_mean_err
