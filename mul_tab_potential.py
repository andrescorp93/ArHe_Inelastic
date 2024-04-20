import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps
from scipy.interpolate import CubicSpline
from funcs import *

dir = 'H_test'
headers = open(f'{dir}/{dir}_diag.txt', 'r').readline().split()[1:]
a = np.loadtxt(f'{dir}/{dir}_diag.txt', skiprows=1)
rtab = a[:,0]
utab = a[:,1:] / 4.637
u = [CubicSpline(rtab, utab[:,i]-utab[-1,i]) for i in range(len(utab[0]))]
r = np.arange(np.min(rtab), np.max(rtab), 0.005)

total = np.zeros((np.max(np.array([len(np.power(10, np.arange(0, np.log10(1.15*utab[0,n]-utab[-1,n]), 0.02))) for n in range(len(utab[0]))])),2*len(u)))
headerfinal = ''
for n in range(len(utab[0])):
    headerfinal += f'State{headers[n][1:]}\tSigma{headers[n][1:]}\t'
    logemax = np.log10(1.15*utab[0,n]-utab[-1,n])
    es = np.power(10, np.arange(0, logemax, 0.02))
    for i in range(len(es)):
        en = np.array([es[i]])
        jmax = np.round(3*np.sqrt(es[i])*np.max(rtab)/4)
        js = np.arange(0,int(jmax),1)
        psi, dpsi = norm_sol(r, en, js, u[n])
        total[i][2*n] = (en+utab[-1,n]) * 4.637
        total[i][2*n+1] = 4 * np.pi * mul_sigma_calc(psi, dpsi, r, en, js)[0] * 1e-16
        # sl = mul_smatrix_calc(psi_cub, dpsi_cub, r, en, js)
        print(f'{total[i][2*n]}\t{total[i][2*n+1]}')

np.savetxt(f'{dir}/sigmas_el_high.txt', total, fmt='%.6e', delimiter='\t', header=headerfinal, comments='')
