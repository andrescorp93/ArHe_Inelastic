import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps
from scipy.interpolate import CubicSpline
from funcs import *

a = np.loadtxt('e1.txt')
rtab = a[:,0]
utab = a[:,1] / 4.637
u1 = lambda x: np.interp(x, rtab, utab)
u2 = CubicSpline(rtab, utab)
r = np.arange(np.min(rtab), np.max(rtab), 0.005)

# en = np.arange(50., 2050., 2.)
js = np.arange(0,51,1)
en = np.arange(5., 1505., 5.)
# phases_lin = np.zeros(len(en))
# phases_cub = np.zeros(len(en))
ejpairs = np.transpose([np.tile(en, len(js)), np.repeat(js, len(en))])
# sm = np.zeros((len(en), 2, 2), dtype=np.complex128)
sol_elastic_cub = solve_ivp(lambda t, y: mulelequations(y, en, js, u2, t), (np.min(rtab), np.max(rtab)),
                                    mul_initial_value_j(np.min(rtab), en, js), t_eval=r)
psi_cub = sol_elastic_cub.y[::2]
dpsi_cub = sol_elastic_cub.y[1::2]

print(mul_phase_calc(psi_cub, dpsi_cub, r, u2, en, js))

# plt.legend()
# plt.show()
        
