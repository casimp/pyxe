import unittest
from edi12.plotting import line_extract
#from edi12.fitting_optimization import p0_approx, peak_fit

import numpy as np
import matplotlib.pyplot as plt

class PlottingTest(unittest.TestCase):
    """Tests for `primes.py`."""

    def setUp(self):
        pass


    def test_extract_xy(self):
        """
        Testing that lines taken at 0/90 degrees are correct.
        """
        x = np.linspace(-5, 15, 20)
        y = np.linspace(-10, 10, 20)
        xx, yy = np.meshgrid(x, y)
        plt.figure()        
        plt.plot(xx.flatten(), yy.flatten(), 'b.')
        
        error = 'Extracted line extends beyond data points.'
        x, y = line_extract(xx, yy, (0, 0), 0)
        self.assertEqual(np.sum(y), 0, error)
        plt.plot(x, y, '-', label='x')  
            
        x, y = line_extract(xx, yy, (0, 0), np.pi/2)
        self.assertEqual(np.sum(y), 0, error)
        plt.plot(x, y, '-', label='y')  
        
        plt.legend()
        plt.xlim(-6, 16)
        plt.ylim(-11, 11)

        
    def test_extract_angles(self):
        """
        Test the line_extract method by rotating from negative to positive 2pi.
        Checking that line stays within values.
        """
        x = np.linspace(-5, 15, 20)
        y = np.linspace(-10, 10, 20)
        xx, yy = np.meshgrid(x, y)
        plt.figure()        
        plt.plot(xx.flatten(), yy.flatten(), 'b.')
        
        for theta in np.linspace(-2*np.pi, 2*np.pi, 21, endpoint=True):
            d1, d2 = line_extract(xx, yy, (0,0), theta)
            
            error = 'Extracted line extends beyond data points.'
            self.assertGreaterEqual(np.max(x), np.max(d1), error)
            self.assertGreaterEqual(np.min(d1), np.min(x), error)
            self.assertGreaterEqual(np.max(y), np.max(d2), error)
            self.assertGreaterEqual(np.min(d2), np.min(y), error)

            plt.plot(d1, d2, '-')
        
        plt.xlim(-6, 16)
        plt.ylim(-11, 11)
        
    def test_extract_outside(self):
        """
        Testing behaviour when centroid is outside of point array.
        """
        x = np.linspace(-5, 15, 20)
        y = np.linspace(-10, 10, 20)
        xx, yy = np.meshgrid(x, y)
        plt.figure()
        plt.plot(xx.flatten(), yy.flatten(), 'b.')
        
        for theta in np.linspace(-2*np.pi, 2*np.pi, 41, endpoint=True):
            d1, d2 = line_extract(xx, yy, (-7,-12), theta)
        
            plt.plot(d1, d2, '-')
        
        plt.xlim(-6, 16)
        plt.ylim(-11, 11)
        
if __name__ == '__main__':
    unittest.main()