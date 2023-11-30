import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps
from scipy.interpolate import CubicSpline
from funcs import *

a = np.loadtxt('e1.txt', skiprows=1)
rtab = a[:,0]
utab = a[:,1] / 4.637
u1 = lambda x: np.interp(x, rtab, utab)
u2 = CubicSpline(rtab, utab-utab[-1])
r = np.arange(np.min(rtab), np.max(rtab), 0.005)

logemax = np.log10(utab[0]-utab[-1])
es = np.power(10, np.arange(0, logemax, 0.02))

result = np.zeros((len(es), 2))
result[:,0] = es * 4.637 * 1.42879
for i in range(len(es)):    
    en = np.array([es[i]])
    jmax = np.round(np.sqrt(es[i])*np.max(rtab)/2)
    js = np.arange(0,int(jmax),1)
    ejpairs = np.transpose([np.tile(js, len(en)), np.repeat(en, len(js))])
    sol_elastic_cub = solve_ivp(lambda t, y: mulelequations(y, en, js, u2, t), (np.min(rtab), np.max(rtab)),
                                        mul_initial_value_j(np.min(rtab), en, js), t_eval=r)
    psi_cub = sol_elastic_cub.y[::2]
    dpsi_cub = sol_elastic_cub.y[1::2]
    
    result[i,1] = 4 * np.pi * mul_sigma_calc(psi_cub, dpsi_cub, r, u2, en, js)[0] * 1e-16
    print(f'{es[i] * 4.637}\t{result[i,1]}')

np.savetxt('sigmas.txt', result)

plt.plot(result[:,0]/1.42879, result[:,1])
plt.show()
        
