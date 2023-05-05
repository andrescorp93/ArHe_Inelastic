import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps
from funcs import *


def initial_value_wkb(x, eps1, eps2, u01, u02, e, j):
    k1p = model_force(eps1, j, x) / (2. * np.sqrt(np.abs(k2(x, u01, eps1, e, j))))
    k2p = model_force(eps2, j, x) / (2. * np.sqrt(np.abs(k2(x, u02, eps2, e, j))))
    k10 = np.sqrt(np.abs(k2(x, u01, eps1, e, j)))
    k20 = np.sqrt(np.abs(k2(x, u02, eps2, e, j)))
    psi1p = k10 - k1p / (2. * k10)
    psi2p = k20 - k2p / (2. * k20)
    # return np.array([0., psi1p, 0., 0.])
    # return np.array([0., psi1p, 0., psi2p])
    return np.array([0., psi1p, 0., psi2p])
    # return np.array([1. / psi1p, 1., 0, 1e-6])


def initial_value_j(x, j):
    # k1p = model_force(eps1, j, x) / (2. * np.sqrt(np.abs(k2(x, u01, eps1, e, j))))
    # k2p = model_force(eps2, j, x) / (2. * np.sqrt(np.abs(k2(x, u02, eps2, e, j))))
    # k10 = np.sqrt(np.abs(k2(x, u01, eps1, e, j)))
    # k20 = np.sqrt(np.abs(k2(x, u02, eps2, e, j)))
    # psi1p = k10 - k1p / (2. * k10)
    # psi2p = k20 - k2p / (2. * k20)
    # return np.array([0., psi1p, 0., 0.])
    # return np.array([0., psi1p, 0., psi2p])
    return np.array([1., j/x, 1., j/x])
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


r = np.arange(2, 20, 0.001)
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

# open('res_phase.dat', 'w').close()
# with open('res_phase.dat', 'w') as res_file:
#     res_file.write(f'E\tJ\td1\td2\n')
    # for e in np.arange(5,205,5):
    #     for j in np.arange(0,31,1):
en = np.arange(550., 1510., 10.)
# rm = np.zeros((len(en), 2, 2))
# km = np.zeros((len(en), 2, 2))
sm = np.zeros((len(en), 2, 2), dtype=np.complex128)
for i in range(len(en)):
    e = en[i]
    for j in np.array([0.]):
        sol_inelastic_1 = solve_ivp(lambda t, y: ccequations(y, e, j, u01, u02, eps1, eps2, d, alpha, x0, t), (2., 20.),
                                    initial_value_wkb(2., eps1, eps2, u1[0], u2[0], e, j), t_eval=r)
        # sol_inelastic_2 = solve_ivp(lambda t, y: ccequations(y, e, j, u01, u02, eps1, eps2, d, alpha, x0, t), (2., 20.),
        #                             initial_value_j(2., j), t_eval=r)
        # sol_elastic = solve_ivp(lambda t, y: elequations(y, e, j, u01, u02, eps1, eps2, d, alpha, x0, t), (2., 20.),
        #                         initial_value(2., eps1, eps2, u1[0], u2[0], e, j), t_eval=r)
        psi_1_inel_1 = sol_inelastic_1.y[0]
        psi_2_inel_1 = sol_inelastic_1.y[2]
        # psi_1_inel_2 = sol_inelastic_2.y[0]
        # psi_2_inel_2 = sol_inelastic_2.y[2]
        dpsi_1_inel_1 = sol_inelastic_1.y[1]
        dpsi_2_inel_1 = sol_inelastic_1.y[3]
        # dpsi_1_inel_2 = sol_inelastic_2.y[1]
        # dpsi_2_inel_2 = sol_inelastic_2.y[3]
        psi_1_inel_1 = psi_1_inel_1 / np.max(psi_1_inel_1)
        psi_2_inel_1 = psi_2_inel_1 / np.max(psi_1_inel_1)
        dpsi_1_inel_1 = dpsi_1_inel_1 / np.max(psi_1_inel_1)
        dpsi_2_inel_1 = dpsi_2_inel_1 / np.max(psi_1_inel_1)
        pm = np.array([[psi_1_inel_1[-1], psi_2_inel_1[-1]], [psi_1_inel_1[-5], psi_2_inel_1[-5]]])
        qm = np.array([[dpsi_1_inel_1[-1], dpsi_2_inel_1[-1]], [dpsi_1_inel_1[-5], dpsi_2_inel_1[-5]]]) # * r[-1]
        rm = pm.dot(np.linalg.inv(qm))
        km = np.diag(np.array([k_inf(u01, e), k_inf(u02, e)]))
        expm = np.diag(np.array([np.exp(-1j*k_inf(u01, e)*r[-1]), np.exp(-1j*k_inf(u02, e)*r[-1])]))
        rkm = rm.dot(km)
        lsm = expm.dot(np.sqrt(np.linalg.inv(km)))
        rsm = np.sqrt(km).dot(expm)
        csm = (np.eye(2)+1j*rkm).dot(np.linalg.inv(np.eye(2)-1j*rkm))
        sm[i] = np.power(-1, int(j)) * lsm.dot(csm.dot(rsm))
        # print(rm)
        print(sm[i].dot(sm[i].T.conjugate()))
            # psi_1_el = sol_elastic.y[0]
            # psi_2_el = sol_elastic.y[2]
            # psi_1_inel_1 = psi_1_inel_1 / np.max(psi_1_inel_1)
            # psi_2_inel_1 = psi_2_inel_1 / np.max(psi_2_inel_1)
            # psi_1_inel_2 = psi_1_inel_2 / np.max(psi_1_inel_2)
            # psi_2_inel_2 = psi_2_inel_2 / np.max(psi_2_inel_2)
            # psi_1_el = psi_1_el / np.max(psi_1_el)
            # psi_2_el = psi_2_el / np.max(psi_1_el)
            # phase1 = phase_calc(psi_1_el, r, u01, eps1, e, j)
            # phase2 = phase_calc(psi_2_el, r, u02, eps2, e, j)
            # res_file.write(f'{e}\t{j}\t{phase1}\t{phase2}\n')
            # print(f'E={e}, J={j} done')
            # name1_inel_1 = f'|1, inel_1, E={e}, J={j}>'
            # name2_inel_1 = f'|2, inel_1, E={e}, J={j}>'
            # name1_inel_2 = f'|1, inel_2, E={e}, J={j}>'
            # name2_inel_2 = f'|2, inel_2, E={e}, J={j}>'
#       name1_el = f'|1, el, E={e}, J={j}>'
#       name2_el = f'|2, el, E={e}, J={j}>'
            # plt.plot(r, psi_1_inel_1, label=name1_inel_1)
            # plt.plot(r, psi_2_inel_1 + 1, label=name2_inel_1)
            # plt.plot(r, psi_1_inel_2 + 2, label=name1_inel_2)
            # plt.plot(r, psi_2_inel_2 + 3, label=name2_inel_2)
#       plt.plot(r, psi_1_el+u01, label=name1_el)
#       plt.plot(r, psi_2_el+u02, label=name2_el)
# plt.legend()
# plt.show()
        
    #     k = np.sqrt(np.abs(k2(r[-1], u02, eps2, e, 0)))
    #     sigma += np.pi*(2*j+1) * np.power(psi_2[-1] * np.sqrt(k) / np.cos(k * r[-1]), 2) / e
    # res_file.write(f'{e}\t{sigma}\n')
# modsm = np.abs(sm)
# angsm = np.angle(sm)
# plt.plot(en, modsm[:,0,0], label='11')
# plt.plot(en, modsm[:,0,1], label='12')
# plt.plot(en, modsm[:,1,0], label='21')
# plt.plot(en, modsm[:,1,1], label='22')
# plt.legend()
# plt.show()
# plt.plot(en, angsm[:,0,0], label='11')
# plt.plot(en, angsm[:,0,1], label='12')
# plt.plot(en, angsm[:,1,0], label='21')
# plt.plot(en, angsm[:,1,1], label='22')
# plt.legend()
# plt.show()
# print(rm)
# print('Done!')

