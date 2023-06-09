import numpy as np
from numba import njit

def model_potential(u0, eps, x):
    return (u0 + eps * np.power(x, -6)) / 4.637  # hbar^2/(2*mu*(1A^2)) = 4.637 cm-1


def morse_non_diag(x, d, alpha, x0):
    return d * (np.exp(-2 * alpha * (x - x0)) - 2 * np.exp(-alpha * (x - x0))) / 0.529 # bohr to angstrom


def morse_non_diag_diff(x, d, alpha, x0):
    return 2 * alpha * d * (-np.exp(-2 * alpha * (x - x0)) + np.exp(-alpha * (x - x0))) / 0.529 # bohr to angstrom


def k2(x, u, e, j):
    return e - u(x) - j * (j + 1) / np.power(x, 2)


def k_inf(u0, e):
    if e > u0:
        return np.sqrt(e - u0)
    else: 
        return 0


def model_force(eps, j, x):
    return ((-6 * eps * np.power(x, -7)) / 4.637) - 2 * j * (j + 1) * np.power(x, -3)  # hbar^2/(2*mu*(1A^2)) = 4.637 cm-1


def initial_value_wkb(x, u1, u2, f1, f2, e, j):
    k1p = f1(x) / (2. * np.sqrt(np.abs(k2(x, u1, e, j))))
    k2p = f2(x) / (2. * np.sqrt(np.abs(k2(x, u2, e, j))))
    k10 = np.sqrt(np.abs(k2(x, u1, e, j)))
    k20 = np.sqrt(np.abs(k2(x, u2, e, j)))
    psi1p = k10 - k1p / (2. * k10)
    psi2p = k20 - k2p / (2. * k20)
    return np.array([0., psi1p, 0., psi2p])
    

def initial_value_j(x, j):
    return np.array([1., j/x])
    

def ccequations(y, e, j, u01, u02, eps1, eps2, d, alpha, x0, x):
    f = np.zeros(len(y))
    f[0] = y[1]
    f[2] = y[3]
    i = morse_non_diag(x, d, alpha, x0)
    ip = morse_non_diag_diff(x, d, alpha, x0)
    f[1] = -k2(x, u01, eps1, e, j) * y[0] + ip * y[0] + 2. * i * y[1]
    f[3] = -k2(x, u02, eps2, e, j) * y[2] - ip * y[2] - 2. * i * y[3]
    return f


def elequations(y, e, j, u, x):
    f = np.zeros(len(y))
    f[0] = y[1]
    f[1] = -k2(x, u, e, j) * y[0]
    return f


def phase_calc(psi, dpsi, r, u, e, j):
    rm = np.max(r)
    k = np.sqrt(np.abs(k2(rm, u, e, j)))
    psi_rm = psi[np.argmax(r)]
    psi_p = dpsi[np.argmax(r)]
    A = (k * np.sin(k * rm) * psi_rm + np.cos(k * rm) * psi_p) / k
    B = (k * np.cos(k * rm) * psi_rm - np.sin(k * rm) * psi_p) / k
    return np.arctan(B / A)


def mul_initial_value_j(x, es, js):
    ejpairs = np.transpose([np.tile(js, len(es)), np.repeat(es, len(js))])
    f = np.ones(2*len(ejpairs))
    f[1::2] = ejpairs[:,0] / x
    return f

    
def mulelequations(y, es, js, u, x):
    ejpairs = np.transpose([np.tile(js, len(es)), np.repeat(es, len(js))])
    f = np.zeros(len(y))
    f[::2] = y[1::2]
    f[1::2] = -k2(x, u, ejpairs[:,1], ejpairs[:,0]) * y[::2]
    return f


def mul_sigma_calc(psi, dpsi, r, u, es, js):
    ejpairs = np.transpose([np.tile(js, len(es)), np.repeat(es, len(js))])
    rm = np.max(r)
    k = np.sqrt(ejpairs[:,1])
    psi_rm = psi[:, np.argmax(r)]
    psi_p = dpsi[:, np.argmax(r)]
    A = (k * np.sin(k * rm - (np.pi / 2) * ejpairs[:,0]) * psi_rm + np.cos(k * rm - (np.pi / 2) * ejpairs[:,0]) * psi_p) / k
    B = (k * np.cos(k * rm - (np.pi / 2) * ejpairs[:,0]) * psi_rm - np.sin(k * rm - (np.pi / 2) * ejpairs[:,0]) * psi_p) / k
    (2 * ejpairs[:,0] + 1) * (B**2 / (B**2 + A**2)) / ejpairs[:,1]
    part_sigma = (2 * ejpairs[:,0] + 1) * (B**2 / (B**2 + A**2)) / ejpairs[:,1]
    result = np.zeros(len(es))
    for i in range(len(ejpairs)):
        for j in range(len(es)):
            if ejpairs[i,1] == es[j]:
                result[j] += part_sigma[i]
    return result
        
        


