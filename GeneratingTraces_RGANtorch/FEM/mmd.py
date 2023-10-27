import torch


def mix_rbf_mmd2_and_ratio(x, y, sigmas, wts=None):
    """
    Berechnet die gemischte RBF MMD^2 und den MMD-Verhältnis.

    :param x: Tensor mit Beispielen aus der ersten Verteilung
    :param y: Tensor mit Beispielen aus der zweiten Verteilung
    :param sigmas: Bandbreiten für den RBF-Kernel (kann eine einzelne Zahl oder eine Liste/Tensor sein)
    :param wts: Gewichtung der einzelnen RBF-Kernel (optional)
    :return: MMD^2 und MMD-Verhältnis
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