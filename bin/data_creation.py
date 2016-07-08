import numpy as np
import matplotlib.pyplot as plt
import time

from pyxe.williams import sigma_xx, sigma_yy, sigma_xy, cart2pol
from pyxe.fitting_functions import strain_transformation, shear_transformation

def plane_strain_s2e(sigma_xx, sigma_yy, sigma_xy, E, v, G=None):
    if G is None:
        G = E / (2 * (1 - v))
    e_xx = (1 / E) * (sigma_xx - v*sigma_yy)
    e_yy = (1 / E) * (sigma_yy - v*sigma_xx)
    e_xy = sigma_xy / G
    return e_xx, e_yy, e_xy


class StrainField(object):

    def __init__(self, x, y, K, E, v, G=None, state='plane strain'):
        self.K = K
        self.x = x
        self.y = y
        self.r, self.theta = cart2pol(x, y)
        self.sig_xx = sigma_xx(self.K, self.r, self.theta)
        self.sig_yy = sigma_yy(self.K, self.r, self.theta)
        self.sig_xy = sigma_xy(self.K, self.r, self.theta)
        sigma_comp = self.sig_xx, self.sig_yy, self.sig_xy
        stress2strain = plane_strain_s2e if state == 'plane strain' else None
        data = stress2strain(*sigma_comp, E, v, G)
        self.e_xx, self.e_yy, self.e_xy = data

    def extract_strain_map(self, phi=np.pi/2, shear=False):
        trans = strain_transformation if not shear else shear_transformation
        e = trans(phi, self.e_xx, self.e_yy, self.e_xy)
        return e

    def plot_strain_map(self, phi=np.pi/2, shear=False):
        e = self.extract_strain_map(phi, shear)
        plt.contourf(self.x, self.y, e, 21)
        plt.show()

    def extract_stress_map(self, phi=np.pi/2, shear=False):
        trans = strain_transformation if not shear else shear_transformation
        sig = trans(phi, self.sig_xx, self.sig_yy, self.sig_xy)
        return sig

    def plot_stress_map(self, phi=np.pi/2, shear=False):
        sig = self.extract_stress_map(phi, shear)
        plt.contourf(self.x, self.y, sig, 21)
        plt.show()

    def extract_strain_array(self, phi):
        """
        Add valus for phi
        :param phi:
        :return:
        """
        strain = np.nan * np.ones((self.x.shape + (1,) + phi.shape))

        for idx, tt in enumerate(phi):
            e_xx1 = strain_transformation(tt, self.e_xx, self.e_yy, self.e_xy)
            strain[:, :, 0, idx] = e_xx1

        return strain

def create_nxs_shell(x, y, phi):
    group = None
    ss2_x = x
    ss2_y = y
    ss2_x = None
    scan_command = [b'ss2_x', b'ss2_y']
    phi = phi
    q = 0
    I = 0

    # create nxs
    # h5py.save
    # load nxs and fill with data


def add_strain_field(data, K, E, v, G=None, state='plane strain'):
    crack_field = StrainField(data.ss2_x, data.ss2_y, K, E, v, G, state)
    data.strain = crack_field.extract_strain_array(data.phi)
    data.strain_err = np.zeros_like(data.strain)
    return crack_field


x = np.linspace(-0.5, 1, 100)
y = np.linspace(-0.75, 0.75, 100)
X, Y = np.meshgrid(x, y)

data = StrainField(X, Y, 20*10**6, 200*10**9, 0.3)
data.create_nxs(np.linspace(0, np.pi, 10))


#sigma_array = np.nan * np.ones((y.size, x.size, 1, n_phi))

#for idx, tt in enumerate(np.linspace(0, np.pi, n_phi)):
#    sigma_array[:, :, 0, idx] = strain_transformation(tt, *(sig_xx, sig_yy, sig_xy))

#e_xx, e_yy, e_xy = plane_strain_s2e(sig_xx, sig_yy, sig_xy, 200 * 10 **9, 0.3)

#strain_array = np.nan * np.ones((y.size, x.size, 1, n_phi))

#for idx, tt in enumerate(np.linspace(0, np.pi, n_phi)):
#    e_xx1 = strain_transformation(tt, *(e_xx, e_yy, e_xy))
#    strain_array[:, :, 0, idx] = e_xx1
#    plt.figure()
#    e_xx1[e_xx1>0.004]=0.004
#    e_xx1[e_xx1 < -0.001] = -0.001
#    plt.contourf(X, Y, e_xx1, np.linspace(-0.001, 0.004, 25))
#    plt.colorbar()
#    plt.contour(X, Y, e_xx1, np.linspace(-0.001, 0.004, 25), colors = 'k', linewidths=0.4, aplha=0.3)

#    plt.savefig(r'C:\Users\lbq76018\Documents\Python Scripts\pyxe_fake\%03d.png' % idx)
    #plt.show()

#plt.figure()
#c = plt.contourf(X, Y, sig_yy, 25)
#plt.colorbar()
#plt.figure()
#c = plt.contourf(X, Y, e_yy, 25)
#plt.colorbar()

#plt.show()
#print(sigma_array)
