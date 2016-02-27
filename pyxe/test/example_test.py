import unittest
from edi12.XRD_analysis import XRD_analysis

            
class XRDTestCase(unittest.TestCase):
    """Tests for `primes.py`."""
    
    def test_analysis(self):
        file = '50414.nxs'
        XRD_analysis(file, 3.1, 0.25)
    


if __name__ == '__main__':
    unittest.main()