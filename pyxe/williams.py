import numpy as np
import matplotlib.pyplot as plt


def sigma_xx(K, r, theta):
    sigma = (K / (2 * np.pi * r / 1000) ** 0.5) * np.cos(theta / 2) * (
             1 - np.sin(theta / 2) * np.sin(3 * theta / 2))

    return sigma


def sigma_yy(K, r, theta):
    sigma = (K / (2 * np.pi * r / 1000) ** 0.5) * np.cos(theta / 2) * (
             1 + np.sin(theta / 2) * np.sin(3 * theta / 2))

    return sigma


def sigma_xy(K, r, theta):
    sigma = (K / (2 * np.pi * r / 1000) ** 0.5) * np.cos(theta / 2) * np.sin(
             theta / 2) * np.sin(3 * theta / 2)
    sigma[theta < 0] = -sigma[theta < 0]
    return sigma


def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


if __name__ == "__main__":
    K_ = 20 * 10**6
    x_ = np.linspace(-0.75, 1.25, 201)
    y_ = np.linspace(-1, 1, 201)
    X, Y = np.meshgrid(x_, y_)
    r_, theta_ = cart2pol(X, Y)

    sig_xy = sigma_xy(K_, r_, theta_, 700*10**6)
    plt.contourf(X, Y, sig_xy, 25)
    plt.contour(X, Y, sig_xy, 25, colors='k', linewidth=0.5)
    plt.show()
