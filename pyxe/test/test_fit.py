import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

phi = np.linspace(-np.pi, 0, 23)
e = np.array([1.87887752e-03,   1.78406970e-03,   1.55862944e-03,
                   1.06470727e-03,   9.93081739e-04,   9.17839411e-04,
                   5.30802495e-04,   4.04060523e-04,   3.21055961e-04,
                   -2.44380819e-04,   5.93497951e-05,  -5.42752739e-05,
                   8.41967803e-05,   4.31684202e-04,   2.75662528e-04,
                   5.80323933e-04,   1.38171805e-03,   1.09401698e-03,
                   1.64751349e-03,   2.25976412e-03,   1.64542547e-03,
                   1.80237684e-03,   1.87526610e-03])

# According to test
e_xx = 0.0018627234618083951
e_yy = -5.5377542242282226e-06
e_xy = -0.0002787013708738522


def strain_transformation(x, *p):
    """
    #   e_xx                  : p[0]
    #   e_yy                  : p[1]
    #   e_xy                  : p[2]
    """

    return (p[0] + p[1]) / 2 + np.cos(2 * x) * (p[0] - p[1]) / 2 + p[2] * np.sin(2 * x)


a, b = curve_fit(strain_transformation, phi, e, [0, 0, 0])

e_calc = strain_transformation(phi, *a)
plt.plot(phi, e, '*')
plt.plot(phi, e_calc, 'r-')
plt.show()