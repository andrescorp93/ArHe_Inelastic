import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps
from scipy.interpolate import CubicSpline
from funcs import *

print(open('scurves.txt', 'r').readline().split())
a = np.loadtxt('scurves.txt', skiprows=1)
rtab = a[:,0]
utab = a[:,1:] / 4.637
u = [CubicSpline(rtab, utab[:,i]-utab[-1,i]) for i in range(len(utab[0]))]
r = np.arange(np.min(rtab), np.max(rtab), 0.005)

total = np.zeros((np.max(np.array([len(np.power(10, np.arange(0, np.log10(utab[0,n]-utab[-1,n]), 0.02))) for n in range(len(utab[0]))])),2*len(u)))

for n in range(len(utab[0])):
    logemax = np.log10(utab[0,n]-utab[-1,n])
    es = np.power(10, np.arange(0, logemax, 0.02))

    for i in range(len(es)):    
        en = np.array([es[i]])
        jmax = np.round(3*np.sqrt(es[i])*np.max(rtab)/4)
        js = np.arange(0,int(jmax),1)
        ejpairs = np.transpose([np.tile(js, len(en)), np.repeat(en, len(js))])
        sol_elastic_cub = solve_ivp(lambda t, y: mulelequations(y, en, js, u[n], t), (np.min(rtab), np.max(rtab)),
                                            mul_initial_value_j(np.min(rtab), en, js), t_eval=r)
        psi_cub = sol_elastic_cub.y[::2]
        dpsi_cub = sol_elastic_cub.y[1::2]

        total[i][2*n] = en * 4.637 * 1.42879
        total[i][2*n+1] = 4 * np.pi * mul_sigma_calc(psi_cub, dpsi_cub, r, u[n], en, js)[0] * 1e-16
        # print(f'{es[i] * 4.637}\t{result[i,1]}')
    
    

# print(total)
np.savetxt(f'sigmas_s.txt', total, fmt='%.6e', delimiter='\t')
    # plt.plot(np.log10(result[:,0]/1.42879), np.log10(result[:,1]))

# plt.xlabel('log10(E)')
# plt.ylabel('log10(sigma)')
# plt.show()
