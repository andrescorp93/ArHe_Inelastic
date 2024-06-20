import numpy as np
import matplotlib.pyplot as plt
from funcs import *
from scipy.integrate import simps, trapezoid

cm1toK = 1.43841 # hc/kB
velcoeff = 3534.19 # sqrt(3kB/mu)

def maxwell(theta, T):
    return 2*np.sqrt(theta/T)*np.exp(-theta/T) / (np.sqrt(np.pi)*T)

dir = 'H_0_plus_p'
r, hls, ddrls = load_matrices(dir)
emin = hls[-1,0,0]
n = len(hls[0])

thres = np.loadtxt(f'{dir}/{dir}_diag.txt', skiprows=1)[-1,1:]

sig_mat = np.loadtxt(f'{dir}/sigmas_total.txt', skiprows=1)

e = sig_mat[:,0]
sigmas = np.zeros((len(e),n,n))
for i in range(n):
    for j in range(n):
        sigmas[:,i,j] = sig_mat[:,i*n+j+1]

thetas = np.zeros((len(e),n))
for i in range(n):
    thetas[:,i] = np.array([max(0, e[j]-thres[i]) * cm1toK for j in range(len(e))])
vels = velcoeff*np.sqrt(thetas)

headerfinal = 'T, K\t'
for i in range(n):
    for j in range(n):
        if i != j:
            headerfinal += f'k_{i}{j}, cm3/s\t'

temperatures = np.arange(300,1100,50)
coeffs = np.zeros((len(temperatures),n,n))
for t in range(len(temperatures)):
    for i in range(n):
        for j in range(n):
            if i != j:
                # coeffs[t,i,j] = simps(maxwell(thetas[:,i], temperatures[t])*vels[:,i]*sigmas[:,i,j],thetas[:,i])
                coeffs[t,i,j] = trapezoid(maxwell(thetas[:,i], temperatures[t])*vels[:,i]*sigmas[:,i,j],thetas[:,i])

tofile = np.zeros((len(temperatures), n*(n-1)+1))
tofile[:,0] = temperatures
m = 0
for i in range(n):
    for j in range(n):
        if i != j:
            tofile[:,m+1] = coeffs[:,i,j]
            m += 1

np.savetxt(f'{dir}/rate_const.txt', tofile, fmt='%.6e', delimiter='\t', header=headerfinal, comments='')
# for i in range(n):
#     for j in range(i+1, n):
#         plt.plot(1/temperatures, coeffs[:,i,j], label=f'{i}->{j}')
#         plt.plot(1/temperatures, coeffs[:,j,i], label=f'{i}<-{j}')

# # for i in range(n):
# #     for j in range(n):
# #         if i != j:
# #             plt.scatter(temperatures, coeffs[:,i,j], label=f'{i}->{j}')
# plt.legend()
# plt.semilogy()
# plt.show()
