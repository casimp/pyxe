# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:34:59 2015

@author: casim
"""

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable


def line_extract(X, Y, point, theta, npoints = 10):
    """
    Draw line through an x, y point cloud. Line is defined by a point and an 
    angle. The user defined point must lie within the point cloud.
    """
    if theta % np.pi/2 == 0:
        if theta % np.pi != 0:
            y = np.linspace(np.min(Y), np.max(Y), npoints)
            x = y * 0 + point[0]
        else:
            x = np.linspace(np.min(X), np.max(X), npoints)
            y = x * 0 + point[1]            
    else:
        m = np.tan(theta)
        c = point[1] - m * point[0]
        
        y_lim = [m * np.min(X) + c, m * np.max(X) + c]
        
        assert min(y_lim) < np.max(Y), "Line does not intersect data."
        assert max(y_lim) > np.min(Y), "Line does not intersect data."
        y_min = np.min(Y) if min(y_lim) < np.min(Y) else min(y_lim)
        y_max = np.max(Y) if max(y_lim) > np.max(Y) else max(y_lim)
            
        y = np.linspace(y_min, y_max, npoints)
        x = (y - c) / m

    return x, y
    
def meshgrid_res(d1, d2, spatial_resolution):
    """
    Takes data points and remeshes at a defined spatial resolution.
    """
    d1_points = np.ceil((np.max(d1) - np.min(d1)) / spatial_resolution) + 1
    d2_points = np.ceil((np.max(d2) - np.min(d2)) / spatial_resolution) + 1
    d1_ = np.linspace(np.min(d1), np.max(d1), d1_points)
    d2_ = np.linspace(np.min(d2), np.max(d2), d2_points)
    D1, D2 = np.meshgrid(d1_, d2_)
    return D1, D2
    
    
def plot_complex(x, y, X, Y, Z, cmap = 'jet', lvls = 11, figsize = (10, 10)):
    f, ax = plt.subplots(figsize = figsize)
    cf_back = ax.contourf(X, Y, Z, lvls, cmap = cmap)
    if type(lvls) != int:
        lvls_ = np.linspace(np.min(lvls), np.max(lvls), 192)
        ax.contourf(X, Y, Z, lvls_, cmap = cmap)
    else:
        ax.contourf(X, Y, Z, 192, cmap = cmap)
    c = ax.contour(X, Y, Z, lvls, colors = '0' , alpha=0.625)
    ax.plot(x, y, '+', color = '0.1' , alpha = 0.75, 
            markersize = 5, linestyle = 'None')
    ax.set_aspect('equal'); ax.autoscale(tight=True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "3%", pad="3%")
    cbar = plt.colorbar(cf_back, cax = cax)
    cbar.add_lines(c)
    return f, ax
    
def plot_simple(x, y, X, Y, Z, cmap = 'RdBu_r', lvls = 11):
    f, ax = plt.subplots()
    cf = ax.contourf(X, Y, Z, lvls, cmap = cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "3%", pad="3%")
    plt.colorbar(cf, cax = cax)

