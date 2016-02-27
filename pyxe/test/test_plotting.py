import unittest
from edi12.plotting import line_extract, meshgrid_res, plot_complex

import numpy as np
import matplotlib.pyplot as plt

class PlottingTest(unittest.TestCase):
    """Tests for plotting helped methods."""

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
        
        for theta in np.linspace(-2*np.pi, 2*np.pi, 21, endpoint=True):
            d1, d2 = line_extract(xx, yy, (0,0), theta)
            
            error = 'Extracted line extends beyond data points.'
            self.assertGreaterEqual(np.max(x), np.max(d1), error)
            self.assertGreaterEqual(np.min(d1), np.min(x), error)
            self.assertGreaterEqual(np.max(y), np.max(d2), error)
            self.assertGreaterEqual(np.min(d2), np.min(y), error)

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
        
    def test_meshgrid_res(self):
        """
        Confirm that the spacing for the meshrid_res function is implemented
        correctly.
        """
        d1 = np.linspace(0, 11, 12, endpoint = True)
        d2 = np.linspace(-10, 10, 5, endpoint = True)
        res = 0.1
        D1, D2 = meshgrid_res(d1, d2, res)
        spacing_1 = (np.max(D1[0,:]) - np.min(D1[0,:])) / (D1[0,:].size - 1)
        spacing_2 = (np.max(D2[:,0]) - np.min(D2[:,0])) / (D2[:,0].size - 1)
        self.assertEqual(spacing_1, res)
        self.assertEqual(spacing_2, res)
        
    def test_plot_complex(self):
        """
        Confirm that the spacing for the meshrid_res function is implemented
        correctly.
        """
        x = np.linspace(-5, 6.2, 35)
        y = np.linspace(-6, 4.4, 35)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
        plot_complex(xx, yy, xx, yy, z, lvls = 11, figsize = (7, 7))

if __name__ == '__main__':
    unittest.main()