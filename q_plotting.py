# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:00:29 2015

@author: Chris
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py
import matplotlib.pyplot as plt
from fitting_optimization import q0_analysis

window = [2.75, 3.25]
fnum = 50514
data_folder = r'/Volumes/XLTRANSCEND/ee12205-1/rawdata/'

q0, stdev = q0_analysis(data_folder, 50514, window, detector = 11)
print('q0 = %.3f (std = %.5f)' % (q0, stdev))


fname = '%d.nxs' % fnum
f = h5py.File(data_folder + fname, 'r')
group = f['entry1']['EDXD_elements']

detector = 11

data_point = (0, 0)

q, I = group['edxd_q'][detector], group['data'][data_point[0], data_point[1], detector]

plt.plot(q, I)