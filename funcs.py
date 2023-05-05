import numpy as np


def model_potential(u0, eps, x):
    return (u0 + eps * np.power(x, -6)) / 4.637  # hbar^2/(2*mu*(1A^2)) = 4.637 cm-1


def morse_non_diag(x, d, alpha, x0):
    return d * (np.exp(-2 * alpha * (x - x0)) - 2 * np.exp(-alpha * (x - x0))) / 0.529 # bohr to angstrom


def morse_non_diag_diff(x, d, alpha, x0):
    return 2 * alpha * d * (-np.exp(-2 * alpha * (x - x0)) + np.exp(-alpha * (x - x0))) / 0.529 # bohr to angstrom


def k2(x, u0, eps, e, j):
    u = model_potential(u0, eps, x)
    return e - u - j * (j + 1) / np.power(x, 2)


def k_inf(u0, e):
    if e > u0:
        return np.sqrt(e - u0)
    else: 
        return 0


def model_force(eps, j, x):
    return ((-6 * eps * np.power(x, -7)) / 4.637) - 2 * j * (j + 1) * np.power(x, -3)  # hbar^2/(2*mu*(1A^2)) = 4.637 cm-1


