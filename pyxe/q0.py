# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:31:03 2016

@author: casim
"""
import sys
import numpy as np
import h5py
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Cursor
from matplotlib.widgets import Button
import matplotlib.pyplot as plt

from pyxe.fitting_functions import gaussian#, lorentzian, psuedo_voigt
from pyxe.fitting_tools import peak_fit, p0_approx 

class Q0(object):
    
    def __init__(self, fname):
        self.f = h5py.File(fname)
        self.data = self.f['entry1/EDXD_elements/data'][:]
        self.q = self.f['entry1/EDXD_elements/edxd_q'][:]
        self.window = 0     
        
    def find_peaks(self, win_opt = True, delta_q = 0.01, steps = 200):
        fig = plt.figure(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.2)

        if win_opt:
            ax = fig.add_subplot(311, axisbg='#FFFFCC')
            ax2 = fig.add_subplot(312, axisbg='#FFFFCC')
            ax3 = fig.add_subplot(313, axisbg='#FFFFCC')
        else:
            ax = fig.add_subplot(211, axisbg='#FFFFCC')
            ax2 = fig.add_subplot(212, axisbg='#FFFFCC')
        
        line2, = ax2.plot(self.q[0], self.data[0, 0, 0], '-')
        
        ax.plot(self.q[0], self.data[0, 0, 0], '-')
        
        def onselect(xmin, xmax):
            data = [self.q[0], self.data[0,0,0]]
            (self.peak, err), self.coeffs = peak_fit(data, [xmin, xmax])
            
            if win_opt:
                
                windows = np.zeros((steps, 1))    
                errors = np.zeros((steps, 1))
                
                for idx, i in enumerate(range(1, steps + 1)):
            
                    window = [self.peak - delta_q * i, self.peak + delta_q * i]
                    p0 = p0_approx(data, window)
            
                    try:
                        peak, stdev = peak_fit(data, window, p0, 'gaussian')[0]
                        errors[idx] = stdev
                    except RuntimeError:
                        errors[idx] = np.nan
                        
                    windows[idx] = delta_q * i    
                ax3.plot(windows, errors)
                fig.canvas.draw()
                
                         
            indmin, indmax = np.searchsorted(self.q[0], (xmin, xmax))
            indmax = min(len(self.q[0]) - 1, indmax)
            thisx = self.q[0][indmin:indmax]
            thisy = self.data[0,0,0][indmin:indmax]
            
            ax2.set_xlim(thisx[0], thisx[-1])
            ax2.set_ylim(thisy.min(), thisy.max())
            
            ax2.plot(thisx, gaussian(thisx, *self.coeffs), 'r--')
            fig.canvas.draw()

        def on_button_press(event):
            if event.inaxes == ax3:
                print("Data coordinates:", event.xdata)
                sys.stdout.flush()
                self.window = event.xdata

                indmin, indmax = np.searchsorted(self.q[0], (self.peak - self.window, self.peak + self.window))
                indmax = min(len(self.q[0]) - 1, indmax)
                thisx = self.q[0][indmin:indmax]
                thisy = self.data[0,0,0][indmin:indmax]
                
                ax2.set_xlim(thisx[0], thisx[-1])
                ax2.set_ylim(thisy.min(), thisy.max())
                
                ax2.plot(thisx, gaussian(thisx, *self.coeffs), 'r--')
                fig.canvas.draw()

            
        
        span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                            rectprops=dict(alpha=0.5, facecolor='red'))
        cursor = Cursor(ax, useblit=True, color='gray', linewidth=1, horizOn=False)
        cursor2 = Cursor(ax3, useblit=True, color='gray', linewidth=1, horizOn=False)
        plt.show()

        def hzfunc(label):
            print(self.peak, self.window)

        rax = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(rax, 'Save')
        bnext.on_clicked(hzfunc)
        plt.show()

        if win_opt:
            
            fig.canvas.mpl_connect('button_press_event', on_button_press)
            
            return cursor, cursor2, span, bnext
        else:
            return cursor, span, bnext
            

            
if __name__ == "__main__":
    q0 = Q0(r'N:/Work Data/ee12205-1/rawdata/50514.nxs')
    q0.find_peaks()
            