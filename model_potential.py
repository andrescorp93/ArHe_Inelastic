import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps
from funcs import *

r = np.arange(2, 11, 0.005)
eps1 = 3.5E6
eps2 = 2E6
u01 = 0
u02 = 547.
d = 0.05
alpha = 0.88
x0 = 3.5

u1 = model_potential(u01, eps1, r)
u2 = model_potential(u02, eps2, r)
u = lambda x: model_potential(u01, eps1, x)
en = np.arange(50., 2050., 2.)
js = np.array([0., 20])
# en = np.array([10.])
phases = np.zeros(len(en))

# sm = np.zeros((len(en), 2, 2), dtype=np.complex128)
for j in js:
    for i in range(len(en)):
        e = en[i]
        sol_elastic = solve_ivp(lambda t, y: elequations(y, e, j, u, t), (2., 20.),
                                initial_value_j(2., j), t_eval=r)
        psi = sol_elastic.y[0]
        dpsi = sol_elastic.y[1]
        phases[i] = phase_calc(psi, dpsi, r, u, e, j)
        # plt.plot(r, psi)
        # plt.plot(r, dpsi)
    plt.plot(en,np.sin(phases)**2, label=f'j={j}')

plt.legend()
plt.show()
        
