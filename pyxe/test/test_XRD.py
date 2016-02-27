import unittest
from edi12.XRD_analysis import XRD_analysis

class XRDTestCase(unittest.TestCase):
    """Tests for `primes.py`."""

    def setUp(self):
        self.data = XRD_analysis('50418.nxs', 3.1, 0.25)
        
    def test_is_five_prime(self):
        """Is five successfully determined to be prime?"""
        XRD_analysis('50418.nxs', 3.1, 0.25)
        self.assertEqual(1, 2)

if __name__ == '__main__':
    unittest.main()