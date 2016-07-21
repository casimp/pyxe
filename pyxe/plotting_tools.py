# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:34:59 2015

@author: casimp
"""
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable


def line_extract(X, Y, pnt, theta, res=0.05):
    x_valid = np.logical_and(pnt[0] >= np.min(X), pnt[1] <= np.max(X))
    y_valid = np.logical_and(pnt[0] >= np.min(Y), pnt[1] <= np.max(Y))
    error = "Specified point (pnt) doesn't lie within data limits."
    assert x_valid and y_valid, error

    if theta % np.pi / 2 == 0:
        if theta % np.pi != 0:
            npnts = 1 + (np.max(Y) - np.min(Y)) // res
            y = np.linspace(np.min(Y), np.max(Y), npnts)
            x = y * 0 + pnt[0]
            d = np.max(y) - np.min(y)
        else:
            npnts = 1 + (np.max(X) - np.min(X)) // res
            x = np.linspace(np.min(X), np.max(X), npnts)
            y = x * 0 + pnt[1]
            d = np.max(x) - np.min(x)
    else:
        m = np.tan(theta)
        c = pnt[1] - m * pnt[0]

        y_lim = [m * np.min(X) + c, m * np.max(X) + c]
        y_min = np.min(Y) if min(y_lim) < np.min(Y) else min(y_lim)
        y_max = np.max(Y) if max(y_lim) > np.max(Y) else max(y_lim)

        x_1, x_2 = (y_min - c) / m, (y_min - c) / m
        d = ((x_2 - x_1)**2 + (y_max - y_min)**2)**0.5
        npnts = 1 + d // res

        y = np.linspace(y_min, y_max, npnts)
        x = (y - c) / m

    return x, y, np.linspace(0, d, npnts)


def az90(phi, idx):

    for i in [-np.pi/2, np.pi/2]:
        if phi[idx] < -np.pi:
            find_ind = np.isclose(phi, np.pi - phi[idx] + i)
        else:
            find_ind = np.isclose(phi, phi[idx] + i)
        if np.sum(find_ind) == 1:
            return int(np.argmax(find_ind))
    raise ValueError('No cake segment found perpendicular to given index.',
                     'Number of cake segments must be divisable by 4.')


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


def plot_complex(x, y, X, Y, Z, lvls=11, figsize=(10, 10),
                 ax=False, cbar=True, **kwargs):
    
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    cf_back = ax.contourf(X, Y, Z, lvls, **kwargs)
    ax.contour(X, Y, Z, levels=[0], colors=('k',),
               linestyles=('--',), linewidths=(3,))
    if type(lvls) != int:
        lvls_ = np.linspace(np.min(lvls), np.max(lvls), 192)
        ax.contourf(X, Y, Z, lvls_, **kwargs)
    else:
        ax.contourf(X, Y, Z, 192, **kwargs)
    c = ax.contour(X, Y, Z, lvls, colors='0', alpha=0.625)
    ax.plot(x, y, '+', color='0.1', alpha=0.75,
            markersize=5, linestyle='None')
    ax.set_aspect('equal')
    ax.autoscale(tight=True)

    divider = make_axes_locatable(ax)
    
    if cbar:  
        cax = divider.append_axes("right", "3%", pad="3%")
        cbar = plt.colorbar(cf_back, cax=cax)
        cbar.add_lines(c)
    return ax
