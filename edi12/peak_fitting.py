# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 08:28:46 2015

@author: Chris Simpson
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
from scipy.optimize import curve_fit


def cos_(x, *p):
    """
    #   Amplitude                    : p[0]
    #   Angle (rad)                  : p[1]
    #   Mean                         : p[2]
    """
    return p[0]*np.cos(2 * (x + p[1])) + p[2]

def gaussian(x, *p):
    """
    Guassian curve fit for diffraction data.
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Standard deviation           : p[3]
    """
    return p[0] + p[1] * np.exp(- (x - p[2])**2 / (2. * p[3]**2))
    
    
def lorentzian(x, *p):
    """
    Loretnzian curve fit for diffraction data.    
    # A lorentzian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   HWHM / Standard deviation    : p[3]
    """
    return p[0] + p[1] / (1.0 + ((x - p[2]) / p[3])**2)


def psuedo_voigt(x, *p):
    """
    Psuedo-voigt curve fit for diffraction data.
    Linear combinationg of gaussian and lorentzian fit.     
    # A psuedo-voigt peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Standard deviation           : p[3]
    #   Linear combination fraction  : p[4]
    """
    return (1 - p[4]) * gaussian(x, *p) + p[4] * lorentzian(x, *p)
    


class FittingTests(unittest.TestCase):
    """
    Our basic test class
    """

    def test_gaussian(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        p0 = np.array([100, 1, 1, 1])
        x = np.linspace(p0[2] - 10 * p0[3], p0[2] + 10 * p0[3], 1000)
        I = gaussian(x, *p0)
        coeff, var_matrix = curve_fit(gaussian, x, I, p0)
        
        self.assertEqual(sum(p0), sum(coeff)) 
        
    def test_lorentzian(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        p0 = np.array([100, 1, 1, 1])
        x = np.linspace(p0[2] - 10 * p0[3], p0[2] + 10 * p0[3], 1000)
        I = lorentzian(x, *p0)
        coeff, var_matrix = curve_fit(lorentzian, x, I, p0)
        
        self.assertEqual(sum(p0), sum(coeff)) 
        
    def test_psuedo_voigt1(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        p0 = np.array([100, 1, 1, 1, 0.5])
        x = np.linspace(p0[2] - 10 * p0[3], p0[2] + 10 * p0[3], 1000)
        I = psuedo_voigt(x, *p0)
        coeff, var_matrix = curve_fit(psuedo_voigt, x, I, p0)
        
        self.assertEqual(sum(p0), sum(coeff))      
        
    def test_psuedo_voigt_gaussian(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        p0 = np.array([100, 1, 1, 1])
        pg = np.array([100, 1, 1, 1, 0])
        x = np.linspace(p0[2] - 10 * p0[3], p0[2] + 10 * p0[3], 1000)

        
        self.assertEqual(sum(gaussian(x, *p0)), sum(psuedo_voigt(x, *pg)))
        
    def test_psuedo_voigt_lorentzian(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        p0 = np.array([100, 1, 1, 1])
        pl = np.array([100, 1, 1, 1, 1])
        x = np.linspace(p0[2] - 10 * p0[3], p0[2] + 10 * p0[3], 1000)

        
        self.assertEqual(sum(lorentzian(x, *p0)), sum(psuedo_voigt(x, *pl)))


if __name__ == '__main__':
    unittest.main()