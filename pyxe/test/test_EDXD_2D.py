from pyxe.edi12_analysis import EDI12
from pyxe.reload import Reload
from pyxe.merge import Merge
import numpy as np
import os

def test_EDXD_2D():
    """
    Goes through most logical uses of the package.
    
    1) Loads raw NeXus file, analyzes and saves
    2) Reloads analysed data
    3) Loads, analyses seconds raw NeXus file - merges with reloaded data and saves
    4) Reloads merged data
    
    The plotting then covers most plotting use cases:
    
    a) Data taken from (1) - strain taken read from detector (q_idx = 0.)
    b) Data taken from (3) - stress calculated from two detectors (q_idx = 1). 
       Uses line prefs to extract and overlay line profile
    
    c) Data taken from (2) - strain calculated at arbitrary angle using tensor.
    d) Data taken from (4) - shear stress calculated from tensor.
       Uses line prefs to extract and overlay line profile. 
    """
    data_store = []

    with EDI12(r'50418.nxs', [3.1, 4.4], 0.25) as data1: 
        data1.save_to_nxs(r'50418_md.nxs')
        data_store.append(data1)
    
    with Reload(r'50418_md.nxs') as data_reload:
        data_store.append(data_reload)
        
    with EDI12(r'50414.nxs', [3.1, 4.4], 0.25) as data2:
        merge = Merge([data_reload, data2], order = [0,1], name = 'merge')
        data_store.append(merge)
        merge.save_to_nxs(r'merge.nxs')
    
    with Reload(r'merge.nxs') as merge_reload:
        data_store.append(merge_reload)
    

    data_store[0].plot_detector(detector=0, q_idx = 0, figsize = (6, 6))
   
    data_store[2].plot_detector(detector=0, q_idx = 1, figsize = (6, 6), 
                                line = True, pnt = (-2.6, 0), 
                                line_angle = np.pi/4, data_type = 'stress')
    
    data_store[1].plot_angle(np.pi/3, q_idx = 0, figsize = (6, 6))
    
    data_store[3].plot_angle(figsize = (6, 6), q_idx = 1, line = True, 
                             pnt = (-2.6, 0), line_angle = np.pi/4, 
                             data_type = 'stress', shear = True)

    
    os.remove(r'/test/merge.nxs')
    os.remove(r'/test/50418_md.nxs')
    
if __name__ == '__main__':
    test_EDXD_2D()