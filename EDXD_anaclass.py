from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from peak_fitting import *
from fitting_optimization import *
from plotting import mesh_and_map
import seaborn as sns
sns.set()
sns.set_context(rc={'lines.markeredgewidth': 0.7})


class EDXD_store():
    
    def __init__(self, collection_name):
        self.name = collection_name
        self.collection = []
        
    def add_stage(self, EDXD_data):
        self.collection.append(EDXD_data) 
        
    def plot_lines(self, detector = [11]):
        for data in self.collection:
            data.plot_lines(detector)


class EDXD_analysis():
    """
    Takes in the ID of files that make up a single stage of EDXD data acquisition
    and allows for the processing, amalgamation and plotting of given data.
    """    
    
    def __init__(self, name, data_folder, fnums, slit_sizes, peak_window, q0, 
                 crack_tip = None):
        self.name = name   
        self.windows = peak_window
        order = np.argsort(slit_sizes)
        self.fnums = np.array(fnums)[order].flatten()
        self.slits = np.array(slit_sizes)[order].flatten()
        self.folder = data_folder
        self.fnames = ['%s%d.nxs' % (data_folder, fnum) for fnum in self.fnums]
        self.q0 = q0
        self.crack_tip = crack_tip
        
        # Initiate holders for peak and strain data
        self.x = {}; self.y = {}
        self.peaks = {}; self.peaks_std = {}
        self.strain = {}; self.strain_std = {}
        self.lines = {} 
        
    def analyze_xy(self, detector_x = 0, detector_y = 11, plot = False):
        
        self.peak_finder([self.detector_x, self.detector_y])
        self.strain_calc([self.detector_x, self.detector_y])
        self.stress_calc(self.detector_x, self.detector_y)
       
    def peak_finder(self, detectors = [0, 11], func = gaussian):
        """
        For a given detector (from 23 element array), fit peaks and merge files.
        Files are merged in order from smallest to largest slit size. Data points
        are removed if they lie within an area mapped with a finer slit size.
        Peaks can be fit with gaussian (default), psuedo-voigt or lorentzian
        functions.
        """
        
        for detector in detectors:

            for idx, slit_size in enumerate(np.unique(self.slits)):
               
                fnames = np.array(self.fnames)[self.slits == slit_size]  

                for order, fname in enumerate(fnames):
                    print('hi')
                    f = h5py.File(fname, 'r')
                    group = f['entry1']['EDXD_elements']
                    x, y = group['ss2_x'], group['ss2_y']
                    q, I = group['edxd_q'][detector], group['data'][:, :, detector]
    
                    peaks, stdevs = array_fit(q, I, self.q0, self.windows, func)         
                    
                    if idx == 0 and order == 0:
                        xi, yi, peaksi, stdevsi = x, y, peaks, stdevs
                            
                    elif idx == 0:
                        xi, yi = np.append(xi, x), np.append(yi, y)       
                        peaksi = np.append(peaksi, peaks)
                        stdevsi = np.append(stdevsi, stdevs)   
    
                    elif order == 0:
                        xi2, yi2, peaksi2, stdevsi2 = x, y, peaks, stdevs
    
                    else:
                        xi2, yi2 = np.append(xi2, x), np.append(yi2, y)   
                        peaksi2 = np.append(peaksi2, peaks)
                        stdevsi2 = np.append(stdevsi2, stdevs)
                            
                    if order == len(fnames) - 1 and idx > 0:
                        cond_x = np.logical_or(xi2 < (np.min(xi) - slit_size), xi2 > (np.max(xi) + slit_size))
                        cond_y = np.logical_or(yi2 < (np.min(yi) - slit_size), yi2 > (np.max(yi) + slit_size))
                        cond = np.logical_or(cond_x, cond_y)
                                                    
                        xi, yi = np.append(xi, xi2[cond]), np.append(yi, yi2[cond])       
                        peaksi = np.append(peaksi, peaksi2[cond])
                        stdevsi = np.append(stdevsi, stdevsi2[cond])
            
            self.peaks[detector] = peaksi
            self.peaks_std[detector] = stdevsi
            self.x[detector] = xi[:]
            self.y[detector] = yi[:]
                
    def strain_calc(self, detectors = [0, 11]):
        """
        Calculate strain from the evaluated peaks
        """
            
        for detector in detectors:
            self.strain[detector] = (self.q0 - self.peaks[detector])/ self.q0
        
        
        
    def stress_calc(self, detector_x = 0, detector_y = 11, 
                    E = 200 * 10**9, v = 0.3):
                        
        try:
            C = E / ((1 - 2 * v) * (1 + v))    
    
            self.stress_x = C * ((1 - v) * self.strain[detector_x] + v * self.strain[detector_y])
            
            self.stress_y = C * (v * self.strain[detector_x] + (1 - v) * self.strain[detector_y])
    
            data_store = np.zeros((np.size(self.stress_x), 6))         
            
            for idx, data in enumerate([self.x[detector_x], self.y[detector_x], 
                                       self.strain[detector_x], self.strain[detector_y],
                                       self.stress_x, self.stress_y]):
                data_store[:, idx] = data.flatten()
                
            self.df = pd.DataFrame(data_store, columns = ['x (mm)', 'y (mm)', 'Strain_x', 'Strain_y', 'Stress_x (Pa)', 'Stress_y (Pa)'])
            self.df.to_csv(r'./Analysis/' + self.name + '.csv')
        except NameError:
            print('You must first calculate strain for given detectors.')
        


            
            
    def strain_mapping(self, detector, lim = [-4.5e-3, 4.5e-3], resolution = 0.05, 
                       cmap = 'RdBu_r', interp = 'linear', plot = True):
        
        name = r'./Analysis/' + self.name + '_strain_' + str(detector)
        mesh_and_map(name, self.x[detector], self.y[detector], self.strain[detector], 
                     self.crack_tip, lim, resolution, cmap, interp, plot)

    def stress_mapping(self, lim = [-1200, 1200], resolution = 0.05, 
                       cmap = 'RdBu_r', interp = 'linear', plot = True):                           
                           
        for orientation, stress in zip(['x', 'y'],
                               [self.stress_x, self.stress_y]):
            name = r'./Analysis/' + self.name + '_stress_' +  orientation
            mesh_and_map(name, self.x[self.detector_x], self.y[self.detector_x], stress/10**6, 
                     self.crack_tip, lim, resolution, cmap, interp, plot)
            
        
    def extract_lines(self, detectors = None, points = 20, point_spacing = None):
        """
        Extracts a line through the array of strain data. Returns position and
        strain vectors. Plotting is optional.
        """
        if detectors == None:
            detectors = np.sort(list(self.strain.keys()))
        
        for detector in detectors:        
            
            if point_spacing != None:
                points = len(np.arange(np.min(self.x[detector]), np.max(self.x[detector]), point_spacing))

            xi = np.linspace(np.min(self.x[detector]), np.max(self.x[detector]), points) 
                
            yi = [self.crack_tip[1]]
            not_nan = np.isfinite(self.strain[detector])
            straini = griddata((self.x[detector][not_nan], self.y[detector][not_nan]), 
                                self.strain[detector][not_nan], (xi, yi), method = 'cubic')
       
            self.lines[detector] = -xi, straini                         
                                
    def plot_lines(self, detectors = [11]):
        """
        Add
        """
        for detector in detectors:   
            with sns.axes_style('darkgrid'):
                
                plt.plot(self.lines[detector][0], self.lines[detector][1], 'x-')
                plt.xlabel('Position relative to notch (mm)')
                plt.ylabel('Strain')
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
    def error_distribution(self, detectors = None, bins = 20):
        """
        Plots a histogram and PDF of the error in the strain data.
        """
        if detectors == None:
            detectors = np.sort(list(self.strain.keys()))
        for detector in detectors:
            std_sorted = np.sort(self.stdevs[detector][np.isfinite(self.stdevs[detector])])
            with sns.axes_style("darkgrid"):
                sns.distplot(std_sorted)
                plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                plt.xlabel('Error (strain)')
                plt.ylabel('Normalised intensity')
        plt.legend(detectors)