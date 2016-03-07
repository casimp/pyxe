# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:31:03 2016

@author: casim
"""

import sys
import numpy as np
import h5py
import matplotlib
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Cursor
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons
import matplotlib.pyplot as plt

from pyxe.fitting_functions import gaussian, lorentzian, psuedo_voigt
from pyxe.fitting_tools import peak_fit, p0_approx 

backend = matplotlib.get_backend()
error = ("Matplotlib running inline. Plot interaction not possible. \n" +
        "Try running %matplotlib in the ipython console (and %matplotlib " +
        "inline to return to default behaviour). In standard console use " +
        "matplotlib.use('TkAgg') to interact.")
        
assert backend != 'module://ipykernel.pylab.backend_inline', error

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if not exponent:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if not precision:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)

class Q0(object):
    
    def __init__(self, fname):
        self.f = h5py.File(fname)
        self.I = self.f['entry1/EDXD_elements/data'][:]
        self.q = self.f['entry1/EDXD_elements/edxd_q'][:]
        self.data = [self.q[0], self.I[0,0,0]]
        self.window = 0     
        
    def find_peaks(self, delta_q = 0.02, steps = 100):
        
        self.peak_dict = {}        
        self.delta_q = delta_q
        self.steps = steps
        self.windows = np.zeros((steps, 1))    
        self.errors = np.zeros((steps, 1))
        func_dict = {'gaussian': gaussian, 'lorentzian': lorentzian, 
                     'psuedo_voigt': psuedo_voigt}
        self.function = 'gaussian'
        
        fig = plt.figure(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.2)
        ax = fig.add_subplot(311, axisbg='#FFFFCC')
        ax2 = fig.add_subplot(312, axisbg='#FFFFCC')
        ax3 = fig.add_subplot(313, axisbg='#FFFFCC')
        ax_slider = plt.axes([0.2, 0.07, 0.5, 0.03])    
        rax = plt.axes([0.81, 0.05, 0.1, 0.075])
        rax2 = plt.axes([0.13, 0.5, 0.125, 0.125])
        
        # Create all lines objects on axes
        ax.plot(self.q[0], self.I[0, 0, 0], '-')
        raw_line, = ax2.plot(self.q[0], self.I[0, 0, 0], '-')
        fit_line, = ax2.plot(0,0, 'r--')
        errors_line, = ax3.semilogy([0, delta_q*steps],[0.0001,0.0001])
        window_line, = ax3.semilogy([0, 0],[0.00001,0.0001], 'r--')
        window_slider = Slider(ax_slider, 'Window', 0, 1, valinit = 0.5)
        store_button = Button(rax, 'Save')
        method_radio = RadioButtons(rax2, ('Gauss', 'Lorent', 'Voigt'), active = 0)
        
        # Create text objects for err text output
        err_text = ax2.text(0.975, 0.83,'', horizontalalignment='right', 
                                 transform = ax2.transAxes)
        err_min_text = ax3.text(0.975, 0.83,'', horizontalalignment='right',
                                transform = ax3.transAxes)
        
        def on_select(xmin, xmax):
            peak_output = peak_fit(self.data, [xmin, xmax], func = self.function)
            (self.peak, self.err), self.coeffs = peak_output
            #self.calculate_errors()
            
            for idx, i in enumerate(range(1, self.steps + 1)):
        
                window = [self.peak - self.delta_q*i, self.peak + self.delta_q*i]
                p0 = p0_approx(self.data, window, self.function)
        
                try:
                    peak, stdev = peak_fit(self.data, window, p0, self.function)[0]
                    if peak < xmin or peak > xmax:
                        self.errors[idx] = np.nan
                    else:
                        self.errors[idx] = stdev / peak
                except RuntimeError:
                    self.errors[idx] = np.nan
                    
                self.windows[idx] = 2 * self.delta_q * i  
                
            err_nan = np.isfinite(self.errors)
            self.qmin = np.min(self.windows[err_nan])
            self.qmax = np.max(self.windows[err_nan])
            
            # Reset window, centre around peak
            self.window = self.qmin + 0.5 * (self.qmax - self.qmin)
            window = [self.peak - self.window/2, self.peak + self.window/2]
            indmin, indmax = np.searchsorted(self.q[0], window)
            indmax = min(len(self.q[0]) - 1, indmax)
            thisx = self.q[0][indmin:indmax]
            thisy = self.I[0,0,0][indmin:indmax]

            # Reset axes limits and set slider value
            ax2.set_xlim(thisx[0], thisx[-1])
            ax2.set_ylim(thisy.min(), thisy.max())
            ax3.set_xlim(self.qmin, self.qmax)
            ax3.set_ylim(np.nanmin(self.errors) / 1.75, 
                         np.nanmax(self.errors) * 1.75)
            window_slider.set_val(0.5)
            
            # Modify x and y data for line objects on ax2 and ax3
            errors_line.set_xdata(self.windows)
            errors_line.set_ydata(self.errors)
            fit_line.set_xdata(thisx)
            function = func_dict[self.function]
            fit_line.set_ydata(function(thisx, *self.coeffs))
            
            err_min = r'$e_{min} =$' + sci_notation(np.nanmin(self.errors), 1)
            err_min_text.set_text(err_min)
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes == ax3:
                sys.stdout.flush()
                self.window = event.xdata
                window = [self.peak - self.window/2, self.peak + self.window/2]
                peak_output = peak_fit(self.data, window, func = self.function)
                (self.peak, self.err), self.coeffs = peak_output

                indmin, indmax = np.searchsorted(self.q[0], window)
                indmax = min(len(self.q[0]) - 1, indmax)
                thisx = self.q[0][indmin:indmax]
                thisy = self.I[0,0,0][indmin:indmax]
                
                ax2.set_xlim(thisx[0], thisx[-1])
                ax2.set_ylim(thisy.min(), thisy.max())
                
                fit_line.set_xdata(thisx)
                function = func_dict[self.function]
                fit_line.set_ydata(function(thisx, *self.coeffs))
                window_line.set_xdata([self.window, self.window])
                window_line.set_ydata([0.000000001, 1])
                slid_val = (self.window - self.qmin) / (self.qmax - self.qmin)
                window_slider.set_val(slid_val)
                
                err = r'$e = $' + sci_notation(self.err / self.peak)
                err_text.set_text(err)
                fig.canvas.draw()
                
        def store_data(label):
            self.peak_dict[self.peak] = [self.window, self.function]

        store_button.on_clicked(store_data)

        def slider_update(val):
            self.window = self.qmin + window_slider.val * (self.qmax - self.qmin)
            window_slider.valtext.set_text('%0.2f' % self.window)
            window = [self.peak - self.window/2, self.peak + self.window/2]
            peak_output = peak_fit(self.data, window, func = self.function)
            (self.peak, self.err), self.coeffs = peak_output

            indmin, indmax = np.searchsorted(self.q[0], window)
            indmax = min(len(self.q[0]) - 1, indmax)
            self.thisx = self.q[0][indmin:indmax]
            thisy = self.I[0,0,0][indmin:indmax]
            
            ax2.set_xlim(self.thisx[0], self.thisx[-1])
            ax2.set_ylim(thisy.min(), thisy.max())
            
            fit_line.set_xdata(self.thisx)
            function = func_dict[self.function]
            fit_line.set_ydata(function(self.thisx, *self.coeffs))
            
            window_line.set_xdata([self.window, self.window])
            window_line.set_ydata([0.000000001, 1])
            err = r'$e = $' + sci_notation(self.err / self.peak)
            err_text.set_text(err)
            
            fig.canvas.draw_idle()
            
        window_slider.on_changed(slider_update)

        
        def fit_method(label):
            
            func_convert = {'Gauss': 'gaussian', 'Lorent': 'lorentzian',
                            'Voigt': 'psuedo_voigt'}
            
            self.function = func_convert[label]
            function = func_dict[self.function]
                       
            window = [self.peak - self.window/2, self.peak + self.window/2]
            indmin, indmax = np.searchsorted(self.q[0], window)
            indmax = min(len(self.q[0]) - 1, indmax)
            thisx = self.q[0][indmin:indmax]
            thisy = self.I[0,0,0][indmin:indmax]
            
            ax2.set_xlim(thisx[0], thisx[-1])
            ax2.set_ylim(thisy.min(), thisy.max())
            
            self.calculate_errors()
            errors_line.set_xdata(self.windows)
            errors_line.set_ydata(self.errors)
            err_min = r'$e_{min} =$' + sci_notation(np.nanmin(self.errors), 1)
            err_min_text.set_text(err_min)
            
            peak_output = peak_fit(self.data, window, func = self.function)
            (self.peak, self.err), self.coeffs = peak_output
            fit_line.set_xdata(thisx)
            fit_line.set_ydata(function(thisx, *self.coeffs))
            
            err = r'$e = $' + sci_notation(self.err / self.peak)
            err_text.set_text(err)
            fig.canvas.draw_idle()
        method_radio.on_clicked(fit_method)

        fig.canvas.mpl_connect('button_press_event', on_click)

        span = SpanSelector(ax, on_select, 'horizontal', useblit = True,
                            rectprops=dict(alpha = 0.5, facecolor = 'red'))
        cursor = Cursor(ax, useblit = True, color = 'gray', 
                        linewidth = 1, horizOn = False)        
        cursor2 = Cursor(ax3, useblit = True, color = 'gray', 
                         linewidth = 1, horizOn = False)
            
        return cursor, cursor2, span, store_button, window_slider, method_radio

    def calculate_errors(self):
        for idx, i in enumerate(range(1, self.steps + 1)):
    
            window = [self.peak - self.delta_q*i, self.peak + self.delta_q*i]
            p0 = p0_approx(self.data, window, self.function)
    
            try:
                peak, stdev = peak_fit(self.data, window, p0, self.function)[0]
                self.errors[idx] = stdev / peak
            except RuntimeError:
                self.errors[idx] = np.nan
                
            self.windows[idx] = 2 * self.delta_q * i     
        err_nan = np.isfinite(self.errors)
        self.qmin = np.min(self.windows[err_nan])
        self.qmax = np.max(self.windows[err_nan])

            
#if __name__ == "__main__":
#
#    q0 = Q0(r'N:/Work Data/ee12205-1/rawdata/50514.nxs')
#    q0.find_peaks()
            