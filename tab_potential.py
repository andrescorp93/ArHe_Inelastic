import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps
from scipy.interpolate import CubicSpline
from funcs import *

a = np.loadtxt('e1.txt')
rtab = a[:,0]
utab = a[:,1] / 4.637
# u = lambda x: np.interp(x, rtab, utab)
u = CubicSpline(rtab, utab)
r = np.arange(np.min(rtab), np.max(rtab), 0.005)

en = np.arange(50., 2050., 2.)
js = np.array([0., 20])
# en = np.array([10.])
phases = np.zeros(len(en))

# sm = np.zeros((len(en), 2, 2), dtype=np.complex128)
for j in js:
    for i in range(len(en)):
        e = en[i]
        sol_elastic = solve_ivp(lambda t, y: elequations(y, e, j, u, t), (np.min(rtab), np.max(rtab)),
                                initial_value_j(np.min(rtab), j), t_eval=r)
        psi = sol_elastic.y[0]
        dpsi = sol_elastic.y[1]
        phases[i] = phase_calc(psi, dpsi, r, u, e, j)
        # plt.plot(r, psi)
        # plt.plot(r, dpsi)
    plt.plot(en,np.sin(phases)**2, label=f'j={j}')

plt.legend()
plt.show()
        
