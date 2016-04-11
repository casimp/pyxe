# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:34:59 2015

@author: casim
"""

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata


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


def az90(phi, idx):
    
    for i in [-np.pi/2, np.pi/2]:
        if phi[idx] < -np.pi:
            find_ind = np.isclose(phi, np.pi - phi[idx] + i)
        else:
            find_ind = np.isclose(phi, phi[idx] + i)
        if np.sum(find_ind) == 1:
            return np.argmax(find_ind)
    raise ValueError('No cake segment found perpendicular to given index.', 
                     'Number of cake segments must be divisable by 4.')
                     
def line_ext(positions, data, pnt, npnts, line_angle, method):
    """
    Not yet working function to take line from data
    """
    if len(positions) == 1:
        d1 = positions[0]
    else:
        d1, d2 = positions        
        d1_e, d2_e = line_extract(d1, d2, pnt, line_angle, npnts)
    
    not_nan = ~np.isnan(data)
    if len(positions) == 2:
        try:
            data = griddata((d1[not_nan], d2[not_nan]), data[not_nan], 
                                  (d1_e, d2_e), method = method)
            return [d1_e, d2_e], data
        except ValueError:
            pass
    else:
        return d1[not_nan], data[not_nan]
    
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
    
def mohrs_dec(func):
    def func_wrapper(*args):
        e_xx, e_yy, e_xy, e_1, e_2 = func(*args)
        R, mean = (e_1 - e_2) / 2, (e_1 + e_2) / 2

        fig = plt.figure(figsize = (7, 5))
        plt.axis('equal')
        ax = fig.add_subplot(1, 1, 1)
        circ = plt.Circle((mean, 0), radius=R, color='k', fill = False)
        ax.add_patch(circ)

        plt.xlim([mean - abs(2 * R), mean + abs(2 * R)])
        plt.plot([e_1, e_2], [0, 0], 'ko', markersize = 3)
        
        plt.plot(e_xx, e_xy, 'ko',label=r'$(\epsilon_{xx}$, $\epsilon_{xy})$')
        plt.plot(e_yy,-e_xy, 'wo',label=r'$(\epsilon_{yy}$, $-\epsilon_{xy})$')
        
        plt.legend(numpoints=1, frameon = False, handletextpad = 0.2)
        plt.plot([e_xx, e_yy], [e_xy, -e_xy], 'k-.')
        ax.annotate('  %s' % r'$\epsilon_{1}$', xy=(e_1, 0), 
                    textcoords='offset points', xytext=(e_1, 0))
        ax.annotate('  %s' % r'$\epsilon_{2}$', xy=(e_2, 0), 
                    textcoords='offset points', xytext=(e_2, 0))
        plt.xlabel('Normal strain')
        plt.ylabel('Shear strain')
    
    return func_wrapper
    

def plot_line(func):
    def func_wrapper(*args):
        pnt, axis, dims, data = func(*args)
        if len(dims) == 1:
                plt.plot(dims, data, '-*')
            
        else:
            d1, d2 = dims
                                    
            if d1[0] > d1[-1]:
                d1, d2, data = d1[::-1], d2[::-1], data[::-1]
                
            zero = ((pnt[0] - d1[0])**2 + (pnt[1] - d2[0])**2)**0.5
            scalar_ext = ((d1 - d1[0])**2 + (d2 - d2[0])**2)**0.5 - zero
    
            x = {'d1' : d1, 'd2' : d2, 'scalar' : scalar_ext}
            plt.plot(x[axis], data, '-x')
    return func_wrapper
    
def plot_complex(x, y, X, Y, Z, lvls = 11, figsize = (10, 10), ax = False, cbar = True, **kwargs):
    
    if not ax:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)

    cf_back = ax.contourf(X, Y, Z, lvls, **kwargs)
    ax.contour(X, Y, Z, levels = [0], colors=('k',),linestyles=('--',),linewidths=(3,))
    if type(lvls) != int:
        lvls_ = np.linspace(np.min(lvls), np.max(lvls), 192)
        ax.contourf(X, Y, Z, lvls_, **kwargs)
    else:
        ax.contourf(X, Y, Z, 192, **kwargs)
    c = ax.contour(X, Y, Z, lvls, colors = '0' , alpha=0.625)
    ax.plot(x, y, '+', color = '0.1' , alpha = 0.75, 
            markersize = 5, linestyle = 'None')
    ax.set_aspect('equal'); ax.autoscale(tight=True)

    divider = make_axes_locatable(ax)
    
    if cbar:  
        cax = divider.append_axes("right", "3%", pad="3%")
        cbar = plt.colorbar(cf_back, cax = cax)
        cbar.add_lines(c)
    return ax
    
def plot_simple(x, y, X, Y, Z, cmap = 'RdBu_r', lvls = 11):
    f, ax = plt.subplots()
    cf = ax.contourf(X, Y, Z, lvls, cmap = cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "3%", pad="3%")
    plt.colorbar(cf, cax = cax)

