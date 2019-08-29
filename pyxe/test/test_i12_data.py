from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import os

from pyxe.merge import ordered_merge
from pyxe.energy_dispersive import EDI12
folder = os.path.join(os.path.split(os.path.dirname(__file__))[0], r'data')
fpath_1 = os.path.join(folder, r'50418.nxs')
fpath_2 = os.path.join(folder, r'50414.nxs')

def test_real():
    fine = EDI12(fpath_1)
    fine.crop1d(2, None, 6)
    coarse = EDI12(fpath_2)
    coarse.crop1d(5, -3, 6)
    for data in [fine, coarse]:
        data.peak_fit(3.1, 0.25)
        data.calculate_strain(3.1)
        data.material_parameters(E=200*10**9, v=0.3)
    merged = ordered_merge([fine, coarse], [0, 1], pad=0.2)
    merged.plot_slice('stress', phi=np.pi/3)

    return fine, coarse, merged

if __name__ == '__main__':
    f, c, m = test_real()
    m.add_material('Fe')
    plt.show()
