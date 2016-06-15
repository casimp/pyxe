import numpy as np
import matplotlib.pyplot as plt
import time

from pyxe.williams import sigma_xx, sigma_yy, sigma_xy, cart2pol
from pyxe.fitting_functions import strain_transformation, shear_transformation

x = np.linspace(-0.5, 1, 100)
y = np.linspace(-0.75, 0.75, 100)

X, Y = np.meshgrid(x, y)
r, theta = cart2pol(X, Y)

sig_xx = sigma_xx(20 * 10**6, r, theta)
sig_yy = sigma_yy(20 * 10**6, r, theta)
sig_xy = sigma_xy(20 * 10**6, r, theta)

sigma_array = np.nan * np.ones((y.size, x.size, 1, 23))

for idx, tt in enumerate(np.linspace(0, np.pi, 23)):
    sigma_array[:, :, 0, idx] = strain_transformation(tt, *(sig_xx, sig_yy, sig_xy))

plt.contourf(X, Y, sig_xx, 25)
plt.show()
print(sigma_array)
