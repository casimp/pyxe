import unittest
from edi12.peak_fitting import gaussian
from edi12.fitting_optimization import p0_approx, peak_fit

import numpy as np

class FittingTestCase(unittest.TestCase):
    """Tests for `primes.py`."""

    def setUp(self):
        pass
        
    def test_gaussian(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        p0 = np.array([100, 1, 0, 1])
        x = np.linspace(-20, 20, 100)
        I = gaussian(x, *p0)
        
        fit = peak_fit(np.array((x, I)), [-5, 5], p0 = p0, func = gaussian)
        np.testing.assert_array_equal(p0, fit[1])
        
    def test_gaussian2(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        p0 = np.array([100, 1, 0, 1])
        x = np.linspace(-20, 20, 100)
        I = gaussian(x, *p0)
        
        fit = peak_fit(np.array((x, I)), [-5, 5], func = gaussian)
        np.testing.assert_array_almost_equal(p0, fit[1], decimal = 5)
        
        
    def test_p0_approx(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        p0 = np.array([100, 1, 0, 1])
        x = np.linspace(-20, 20, 100)
        I = gaussian(x, *p0)
        
        p0_est = p0_approx(np.array((x, I)), [-5, 5], func = gaussian)
        np.testing.assert_array_almost_equal(p0, p0_est, decimal = 0)
        print(p0, p0_est)
        
        
if __name__ == '__main__':
    unittest.main()