import numpy as np
from scipy.linalg import sqrtm


def fidelity(rho, sigma):
    rho = np.array(rho, dtype=complex)
    sigma = np.array(sigma, dtype=complex)
    return np.abs(np.trace(sqrtm(np.dot(sqrtm(rho), np.dot(sigma, sqrtm(rho)))))) ** 2


def purity(rho):
    rho = np.array(rho, dtype=complex)
    return np.trace(np.dot(rho, rho)).real
