# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:09:42 2015

@author: casim
"""
from scipy.interpolate import griddata
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

sns.set()
sns.set_context(rc={'lines.markeredgewidth': 0.7})

def mapping_style(data_points, x, y, z, crack_tip, c_lim, x_lim = None, y_lim = None, 
                  cmap = 'RdBu_r', interp = 'linear', plot = True):  
    """
    Divergent mapping style for strain plotting
    """
    with sns.axes_style('ticks'):
            
        plt.rc('axes', linewidth = 0.8)
        plt.rc('xtick.major', width = 0.8, size = 4)
        plt.rc('ytick.major', width = 0.8, size = 4)
        
        plt.figure(figsize = (12.5, 6))
        plt.contourf(x, y, z, np.linspace(c_lim[0], c_lim[1], 21), 
                     cmap = cmap, interp = interp)
        plt.clim(c_lim)
        plt.colorbar(format = '%.1e')

        CS = plt.contour(x, y, z, np.linspace(c_lim[0], c_lim[1], 21), 
                    colors = 'k', alpha = 0.3, interp = interp, linewidths = 1)
        plt.clim(CS, c_lim)
        #plt.clabel(CS, CS.levels[::2], inline=1, fontsize=10)
        plt.plot(data_points[0], data_points[1], '+', color = '0.15' , alpha=0.5, markersize = 3)
        plt.axis('equal')
        
        if x_lim != None and y_lim != None:        
            plt.xlim(x_lim)
            plt.ylim(y_lim)
        
        if crack_tip != None:
            points = np.array([[-0.5, -0.1], [0, -0.1], [0, -0.025], 
                               [crack_tip[0], -0.025], [crack_tip[0], 0.025], 
                               [0, 0.025], [0, 0.1], [-0.5, 0.1]])
            points[:,1] = points[:,1] + crack_tip[1]
            crack = plt.Polygon(points, fc='w', lw = 0.8, ec = '0.5', zorder = 10)
            plt.gca().add_patch(crack)
        plt.axis('tight')


def mesh_and_map(name, x, y, z, crack_tip, lim, resolution = 0.05, 
            cmap = 'RdBu_r', interp = 'linear', plot = True):
    """
    Pretty contour plotting of x, y, strain/stress.
    Takes information about the scan for relevant naming/saving.
    """
    x = -x if (abs(np.min(x)) > abs(np.max(x))) else x  
    
    points_x = (np.max(x) - np.min(x)) / resolution
    points_y = (np.max(y) - np.min(y)) / resolution
    xi, yi = np.meshgrid(np.linspace(np.min(x), np.max(x), points_x), 
                         np.linspace(np.min(y), np.max(y), points_y))
    not_nan = np.isfinite(z)
    z_i = griddata((x[not_nan], y[not_nan]), z[not_nan], 
                       (xi, yi), method = interp)
    mapping_style((x, y), xi, yi, z_i, crack_tip, lim, 
                  cmap = cmap, interp = interp, plot = plot)
  
    #plt.title(name)
    plt.xlabel('Position relative to notch (mm)')
    plt.xlim([-0.5, 9.5]), plt.ylim([-3, 3])
    
    plt.savefig(name + '.png', dpi = 300)
    if plot == False:
        plt.close() 