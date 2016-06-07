# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:25:14 2016

@author: casim
"""

from nose.tools import assert_equal
import numpy as np
import os
from mock import patch

from pyxe.edi12_analysis import EDI12
from pyxe.reload import Reload
from pyxe.merge import Merge


def test_load():
    """
    Check that the correct number of files is being loaded.
    """
    data = EDI2(r'50418.nxs')