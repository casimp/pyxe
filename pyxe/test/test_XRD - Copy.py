import unittest
from pyxe.edi12_analysis import EDI12
#from pyxe.reload import Reload

class EDXDTestCase(unittest.TestCase):
    """Tests for edi12 analysis."""

    def test_analysis(self):
        self.data = EDI12('50418.nxs', [3.1, 4.4], 0.25)
        
    def test_dimensions(self):
        self.assertEqual(self.data.strain.shape, (13, 15, 24, 2))
        
    def test_save(self):
        """Test saving capabilities"""
        self.data.save_to_nxs('50418_md.nxs')
        
        
    def test_plot_detector(self):
        
        self.data.plot_detector([0])
        
    def test_plot_angle(self):
        self.data.plot_angle(0)
        
    

if __name__ == '__main__':
    unittest.main()