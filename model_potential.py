import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps


def model_potential(u0, eps, x):
    return (u0 + eps * np.power(x, -6)) / 4.637  # hbar^2/(2*mu*(1A^2)) = 4.637 cm-1


def morse_non_diag(x, d, alpha, x0):
    return d * (np.exp(-2 * alpha * (x - x0)) - 2 * np.exp(-alpha * (x - x0)))


def morse_non_diag_diff(x, d, alpha, x0):
    return 2 * alpha * d * (-np.exp(-2 * alpha * (x - x0)) + np.exp(-alpha * (x - x0)))


def k2(x, u0, eps, e, j):
    u = model_potential(u0, eps, x)
    return e - u - j * (j + 1) / np.power(x, 2)


def model_force(eps, j, x):
    return ((-6 * eps * np.power(x, -7)) / 4.637) - 2 * j * (j + 1) * np.power(x, -3)  # hbar^2/(2*mu*(1A^2)) = 4.637 cm-1


def initial_value(x, eps1, eps2, u01, u02, e, j):
    k1p = model_force(eps1, j, x) / (2. * np.sqrt(np.abs(k2(x, u01, eps1, e, j))))
    k2p = model_force(eps2, j, x) / (2. * np.sqrt(np.abs(k2(x, u02, eps2, e, j))))
    k10 = np.sqrt(np.abs(k2(x, u01, eps1, e, j)))
    k20 = np.sqrt(np.abs(k2(x, u02, eps2, e, j)))
    psi1p = k10 - k1p / (2. * k10)
    psi2p = k20 - k2p / (2. * k20)
    # return np.array([0., psi1p, 0., 0.])
    return np.array([0., psi1p, 0., psi2p])
    # return np.array([1. / psi1p, 1., 0, 1e-6])


def ccequations(y, e, j, u01, u02, eps1, eps2, d, alpha, x0, x):
    f = np.zeros(len(y))
    f[0] = y[1]
    f[2] = y[3]
    i = morse_non_diag(x, d, alpha, x0)
    ip = morse_non_diag_diff(x, d, alpha, x0)
    f[1] = -k2(x, u01, eps1, e, j) * y[0] + ip * y[0] + 2. * i * y[1]
    f[3] = -k2(x, u02, eps2, e, j) * y[2] - ip * y[2] - 2. * i * y[3]
    return f


def elequations(y, e, j, u01, u02, eps1, eps2, d, alpha, x0, x):
    f = np.zeros(len(y))
    f[0] = y[1]
    f[2] = y[3]
    f[1] = -k2(x, u01, eps1, e, j) * y[0]
    f[3] = -k2(x, u02, eps2, e, j) * y[2]
    return f


def phase_calc(psi, r, u0, eps, e, j):
    rm = np.max(r)
    k = np.sqrt(np.abs(k2(rm, u0, eps, e, j)))
    psi_rm = psi[np.argmax(r)]
    psi_p = (psi_rm - psi[np.argmax(r) - 1]) / (rm - r[np.argmax(r) - 1])
    A = (k * np.sin(k * rm) * psi_rm + np.cos(k * rm) * psi_p) / k
    B = (k * np.cos(k * rm) * psi_rm - np.sin(k * rm) * psi_p) / k
    return np.arctan(B / A)


r = np.arange(2, 20, 0.01)
eps1 = 3.5E6
eps2 = 2E6
u01 = 0
u02 = 547.
d = 0.05
alpha = 0.88
x0 = 3.5

u1 = model_potential(u01, eps1, r)
u2 = model_potential(u02, eps2, r)
# f = morse_non_diag(r, 0.05, 0.88, 3.5)
# fdp = morse_non_diag_diff(r, 0.05, 0.88, 3.5)

# plt.plot(r, u1)
# plt.plot(r, u2)
# plt.show()

# plt.plot(r, f)
# plt.show()

open('res_phase.dat', 'w').close()
with open('res_phase.dat', 'w') as res_file:
    res_file.write(f'E\tJ\td1\td2\n')
    for e in np.arange(5,205,5):
        for j in np.arange(0,31,1):
            sol_inelastic = solve_ivp(lambda t, y: ccequations(y, e, j, u01, u02, eps1, eps2, d, alpha, x0, t), (2., 20.),
                                    initial_value(2., eps1, eps2, u1[0], u2[0], e, j), t_eval=r)
            sol_elastic = solve_ivp(lambda t, y: elequations(y, e, j, u01, u02, eps1, eps2, d, alpha, x0, t), (2., 20.),
                                    initial_value(2., eps1, eps2, u1[0], u2[0], e, j), t_eval=r)
            psi_1_inel = sol_inelastic.y[0]
            psi_2_inel = sol_inelastic.y[2]
            psi_1_el = sol_elastic.y[0]
            psi_2_el = sol_elastic.y[2]
            psi_1_inel = psi_1_inel / np.max(psi_1_inel)
            psi_2_inel = psi_2_inel / np.max(psi_2_inel)
            psi_1_el = psi_1_el / np.max(psi_1_el)
            psi_2_el = psi_2_el / np.max(psi_1_el)
            phase1 = phase_calc(psi_1_el, r, u01, eps1, e, j)
            phase2 = phase_calc(psi_2_el, r, u02, eps2, e, j)
            res_file.write(f'{e}\t{j}\t{phase1}\t{phase2}\n')
            print(f'E={e}, J={j} done')
#       name1_inel = f'|1, inel, E={e}, J={j}>'
#       name2_inel = f'|2, inel, E={e}, J={j}>'
#       name1_el = f'|1, el, E={e}, J={j}>'
#       name2_el = f'|2, el, E={e}, J={j}>'
#       plt.plot(r, psi_1_inel+u01, label=name1_inel)
#       plt.plot(r, psi_2_inel+u02, label=name2_inel)
#       plt.plot(r, psi_1_el+u01, label=name1_el)
#       plt.plot(r, psi_2_el+u02, label=name2_el)
# plt.legend()
# plt.show()
        
    #     k = np.sqrt(np.abs(k2(r[-1], u02, eps2, e, 0)))
    #     sigma += np.pi*(2*j+1) * np.power(psi_2[-1] * np.sqrt(k) / np.cos(k * r[-1]), 2) / e
    # res_file.write(f'{e}\t{sigma}\n')

print('Done!')

