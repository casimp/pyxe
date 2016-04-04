# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:02:41 2016

@author: casimp
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pyxe.area_analysis import Area
import re
import os
import pandas as pd

def spec_scrape(folder, save = False):
    """
    Runs through a .spc file (located in folder with associated .edf files)
    and extracts load, position, and slit size information.
    """
    spec_file = sorted([x for x in os.listdir(folder) if x[-4:] == '.spc'])
    error = 'Either zero or multiple .spc files have been found.'
    assert len(spec_file) == 1, error
    spec_file = spec_file[0]
    scan = spec_file[:-4]
    
    data_store = []
    with open(os.path.join(folder, spec_file), 'r') as f:
        lines = [line.rstrip('\n') for line in f][1:]
    
    search = r'-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?'
        
    for idx, line in enumerate(lines):
        if scan in line:
            x, y = [float(i) for i in re.findall(search, lines[idx + 11])[2:4]]
            slit_x = [float(i) for i in re.findall(search, lines[idx + 12])][7]
            slit_y = [float(i) for i in re.findall(search, lines[idx + 13])][1]
            scan_num = [float(i) for i in re.findall(search, lines[idx])][-1]
            load = [float(i) for i in re.findall(search, lines[idx + 23])][-3]
            data_store.append([int(scan_num), load, x, y, slit_x, slit_y]) 
    
    df = pd.DataFrame(data_store, columns=('Scan Number', 'Load (kN)', 
                      'x (mm)', 'y (mm)', 'slit_x (mm)', 'slit_y (mm)'))
    if save:
        pd.to_pickle(df, os.path.join(folder, '%s.pkl' % scan))
    return df
    