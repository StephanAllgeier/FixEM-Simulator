import numpy as np
import torch


def rbf_kernel_numpy(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))


def mmd_squared(X, Y, kernel=rbf_kernel_numpy):
    n = X.shape[0]
    m = Y.shape[0]

    mmd = 0.0

    for i in range(n):
        for j in range(n):
            mmd += kernel(X[i], X[j])

    mmd /= n * (n - 1)

    for i in range(n):
        for j in range(m):
            mmd -= 2 * kernel(X[i], Y[j]) / (n * m)

    for i in range(m):
        for j in range(m):
            mmd += kernel(Y[i], Y[j]) / (m * (m - 1))

    return mmd


def mix_rbf_mmd2_and_ratio(x, y, sigmas, wts=None):
    """
    Computes the mixed Radial Basis Function (RBF) MMD^2 and the MMD ratio.

    Parameters:
        - x: Tensor with samples from the first distribution.
        - y: Tensor with samples from the second distribution.
        - sigmas: Bandwidths for the RBF kernel (can be a single number or a list/tensor).
        - wts: Weights for individual RBF kernels (optional).

    Returns:
        - MMD^2 and MMD ratio.
    """
    if isinstance(sigmas, (int, float)):
        sigmas = [sigmas]

    Kxx, Kyy, Kxy = 0, 0, 0

    for sigma in sigmas:
        x_kernel = rbf_kernel(x, x, sigma)
        y_kernel = rbf_kernel(y, y, sigma)
        xy_kernel = rbf_kernel(x, y, sigma)

        Kxx += x_kernel.mean()
        Kyy += y_kernel.mean()
        Kxy += xy_kernel.mean()

    mmd2 = Kxx + Kyy - 2 * Kxy

    if wts is not None:
        mmd2 *= wts

    return mmd2.item(), Kxy.item() / (torch.sqrt(Kxx * Kyy) + 1e-8)


def rbf_kernel(x, y, sigma=1.0):
    """
    Berechnet den RBF-Kernel zwischen den Mengen x und y.

    :param x: Tensor mit Beispielen aus der ersten Menge
    :param y: Tensor mit Beispielen aus der zweiten Menge
    :param sigma: Bandbreite für den RBF-Kernel
    :return: RBF-Kernel-Matrix
    """
    distance = torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=2)
    kernel = torch.exp(-distance / (2 * sigma ** 2))
    return kernel
