# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:34:59 2015

@author: casimp
"""
import numpy as np
from numpy.polynomial.chebyshev import chebval
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyxe.fitting_tools import pawley_hkl, extract_parameters, array_fit_pawley
from scipy.optimize import curve_fit


def line_extract(X, Y, pnt, theta, res=0.05):
    """ Extracts line from 2d position array (according to point/angle).

    Args:
        X (ndarray): 1d/2d-array of positions
        Y (ndarray): 1d/2d-array of positions
        pnt (tuple): Define data point (index) else point (0, ) x ndim.
        theta (float): Angle (rad) though 2D array

    Returns:
        tuple: x, y, d - where (x, y) are vector co-ords and d is scalar pos
    """
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


def az90(phi, az_idx):
    """ Searches for and returns azithmuthal idx perp. to specified idx.

    Args:
        phi (ndarray): 1d array of azimuthal angles
        idx (int): Azimuthal slice index of chosen slice

    Returns:
        int: Azimuthal slice index
    """
    for i in [-np.pi/2, np.pi/2]:
        if phi[az_idx] < -np.pi:
            find_ind = np.isclose(phi, np.pi - phi[az_idx] + i)
        else:
            find_ind = np.isclose(phi, phi[az_idx] + i)
        if np.sum(find_ind) == 1:
            return int(np.argmax(find_ind))
    raise ValueError('No cake segment found perpendicular to given index.',
                     'Number of cake segments must be divisable by 4.')


def meshgrid_res(d1, d2, spatial_resolution):
    """ Takes flat data point arrays, re-meshes at a defined spatial resolution.

    Args:
        d1 (ndarray): Positions (x)
        d2 (ndarray): Positions (y)
        spatial_resolution (float): Point spacing

    Returns:
        tuple: Re-meshed 2d arrays (d1, d2)
    """
    d1_points = np.ceil((np.max(d1) - np.min(d1)) / spatial_resolution) + 1
    d2_points = np.ceil((np.max(d2) - np.min(d2)) / spatial_resolution) + 1
    d1_ = np.linspace(np.min(d1), np.max(d1), d1_points)
    d2_ = np.linspace(np.min(d2), np.max(d2), d2_points)
    return np.meshgrid(d1_, d2_)


def plot_complex(x_raw, y_raw, x, y, z, levels=11, limits=[None, None],
                 continuous=True, figsize=(10, 10),
                 ax=False, cbar=True, **kwargs):
    """ Plots 2D heat map of stress/strain fields.
    Args:
        x_raw (ndarray): Data acquisision points
        y_raw (ndarray): Data acquisision points
        x (ndarray): 2D x-position array (interpolated)
        y (ndarray): 2D y-position array (interpolated)
        z (ndarray): 2D stress/strain array
        lvls (int, ndarray): Number of contours to display (or defined levels)
        figsize (tuple): Figure size
        ax: Supply axis to plot on or (False) create new plot
        cbar (bool): Display colour bar
    """
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    if limits != [None, None]:
        z[z < limits[0]] = limits[0]
        z[z > limits[1]] = limits[1]

    cf_back = ax.contourf(x, y, z, levels, **kwargs)

    # Zero markings
    ax.contour(x, y, z, levels=[0], colors=('k',),
               linestyles=('--',), linewidths=(3,))

    # Continuous background of discrete colours
    if continuous:
        if not isinstance(levels, int):
            lvls_ = np.linspace(np.min(levels), np.max(levels), 192)
            ax.contourf(x, y, z, lvls_, **kwargs)
        else:
            ax.contourf(x, y, z, 192, **kwargs)

    # Contours
    c = ax.contour(x, y, z, levels, colors='0', alpha=0.625)

    # Acquisition points
    ax.plot(x_raw, y_raw, '+', color='0.1', alpha=0.75,
            markersize=5, linestyle='None')

    # Formatting
    ax.set_aspect('equal')
    ax.autoscale(tight=True)

    divider = make_axes_locatable(ax)

    if cbar:
        cax = divider.append_axes("right", "3%", pad="3%")
        cbar = plt.colorbar(cf_back, cax=cax)
        cbar.add_lines(c)
    return ax

def pawley_plot(q, I, detector, az_idx, ax, q_lim=None, func='gaussian'):
    """ Plots q against measured intensity overlaid with Pawley fit.

    Includes highlighting of anticipated Bragg peak locations and
    difference between measured intensity and Pawley fit.

    Args:
        q (ndarray): Reciprocal lattice
        I (ndarray): Intensity
        detector: pyxpb detector instance
        az_idx (int): Azimuthal slice index
        ax: Axis to apply plot to
    """

    background = chebval(q, detector._back[az_idx])
    if q_lim is None:
        q_lim = [np.min(q), np.max(q)]
    p0 = extract_parameters(detector, q_lim, np.nanmax(I))
    pawley = pawley_hkl(detector, background, func=func)
    coeff, var_mat = curve_fit(pawley, q, I, p0=p0)
    I_pawley = pawley(q, *coeff)

    # Plot raw data and Pawley fit to data
    ax.plot(q, I, 'o', markeredgecolor='0.3', markersize=4,
            markerfacecolor='none', label=r'$\mathregular{I_{obs}}$')
    ax.plot(q, I_pawley, 'r-', linewidth=0.75,
            label=r'$\mathregular{I_{calc}}$')

    # Plot Bragg positions - locate relative to max intensity
    ymin = -ax.get_ylim()[1] / 10
    materials = detector.materials
    for idx, mat in enumerate(materials):
        offset = (1 + idx) * ymin / 2
        for q0 in detector.q0[mat]:
            bragg_line = [offset + ymin / 8, offset - ymin / 8]
            ax.plot([q0, q0], bragg_line, 'g-', linewidth=2)
        # Use error bars to fudge vertical lines in legend
        ax.errorbar(0, 0, yerr=1, fmt='none', capsize=0, ecolor='g',
                    elinewidth=1.5, label=r'Bragg ({})'.format(mat))

    # Plot difference between raw and Pawley fit (shifted below Bragg)
    I_diff = I - I_pawley
    max_diff = np.max(I_diff)
    shifted_error = I - I_pawley + (idx + 2) * ymin / 2 - max_diff
    ax.plot(q, shifted_error, 'b-', linewidth=0.75,
            label=r'$\mathregular{I_{diff}}$')

    # Remove ticks below 0 intensity
    ylocs = ax.yaxis.get_majorticklocs()
    yticks = ax.yaxis.get_major_ticks()
    for idx, yloc in enumerate(ylocs):
        if yloc < 0:
            yticks[idx].set_visible(False)

    legend = ax.legend(numpoints=1)
    frame = legend.get_frame()
    frame.set_facecolor('w')
