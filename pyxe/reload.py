# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:40:07 2015

@author: casimp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pyxe.scrape import Scrape
from pyxe.plotting import StrainPlotting
from pyxe.strain_tools import StrainTools


class Reload(StrainTools, StrainPlotting):
    """
    Takes post-processed .nxs file from the I12 EDXD detector. File should have
    been created with the XRD_analysis tool and contain detector specific peaks 
    and associated strain.
    """
    def __init__(self, file):
        super(Reload, self).__init__(file)
        group = self.f['entry1']['EDXD_elements']
        try:        
            self.strain = group['strain'][:]
        except KeyError as e:
            e.args += ('Invalid .nxs file - no strain data found.',
                       'Run XRD_analysis tool.')
            raise
        self.dims = group['dims'][:]
        self.q0 = group['q0'][:]
        self.peak_windows = group['peak_windows'][:]
        self.peaks = group['peaks'][:]
        self.peaks_err = group['peaks_err'][:]    
        self.strain_err = group['strain_err'][:]
        self.strain_param = group['strain_param'][:]

