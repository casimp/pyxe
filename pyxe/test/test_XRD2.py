# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:44:50 2016

@author: casim
"""

from pyxe.area_analysis import Area

azimuth_range = [-180, 0]
npt_azim = 23


poni = (741.577, 1053.137, 1027.562, 0.153, 41.314, 200, 200, 1.631371*10**-11)
base_folder = 'N:/Work Data/ee11080/Test15_CNTI6/'

q0_est = 2.528
win_width = 0.23

folder= base_folder + '0'
pos_file = folder + '/positions.csv'
bidge = Area(folder, pos_file, poni, q0_est, win_width, 
              pos_delimiter = ',', exclude = ['dark'], output = 'none', 
              error_limit = 2 * 10 ** -4, azimuth_range = azimuth_range,
              npt_azim = npt_azim)

bidge.strain_fit(10**-3)    
bidge.save_to_nxs('hi3')