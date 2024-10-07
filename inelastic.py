import numpy as np
import matplotlib.pyplot as plt
from funcs import *
from scipy.integrate import simps
import cycler


def get_state_curve(h, i, x):
    return np.real(np.diag(h(x))[i])


plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 15
plt.rcParams['image.cmap'] = 'Paired'
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
                                                          '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'])
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['savefig.format'] = 'eps'
plt.rcParams['lines.linewidth'] = 3

dir = 'H_test'
r, hls, ddrls = load_matrices(dir)
emin = hls[-1,0,0]
n = len(hls[0])
shift = np.array([emin*np.eye(len(hls[0]), dtype=complex)]*len(r))
vd = (hls-shift)/4.637

hfunc, ddrfunc = matrix_funcs(r, vd, ddrls)
emax = np.abs(np.array(hfunc(r[0])[-1,-1]))
# es = np.power(10, np.arange(0, np.log10(emax), 0.01))
# es = np.power(10, np.array([np.log10(emin), np.log10(emax)]))
es = np.array([emax/3])
x = np.arange(np.min(r), np.max(r)+0.005, 0.005)
sigmas = np.zeros((len(es), n, n))
sigmadiff = np.zeros((len(es), n+1))
sigmadiff[:,0] = es * 4.637 + np.real(hls[-1,0,0])
headerfinal = 'E, cm-1\t'
for i in range(n):
    for j in range(n):
        headerfinal += f'Sigma_{i}{j}, cm2\t'
for k in range(len(es)):
    e = es[k]
    lmax = int(np.round(np.sqrt(e)*np.max(r[-1]))*0.8)
    q = np.abs(np.sqrt(e-np.diag(hfunc(r[-1]))))
    js = np.arange(0,int(lmax),1)
    psis = {}
    dpsis = {}
    nt = np.sum(np.diag(vd[-1]) < e)
    for i in range(nt):
        en = np.array([e-np.real(vd[-1,i,i])])
        psi, dpsi = norm_sol(x, en, js, lambda z: get_state_curve(hfunc, i, z)-vd[-1,i,i])
        # sigmas[k, i, i] = 4e-16 * np.pi * mul_sigma_calc(psi, dpsi, x, en, js)[0]
        psis[i] = psi
        dpsis[i] = dpsi
        # sigmadiff[k,i+1] = 4e-16 * np.pi * np.sum((js[:-1]+1)*np.abs(np.sin(np.diff(s_phase_calc(psi, dpsi, x, en, js))))**2)/en[0]
    print(e * 4.637 + np.real(hls[-1,0,0]))
    if nt > 1:
        als = np.zeros((lmax, nt, nt), dtype=complex)
        addr = np.zeros((lmax, nt, nt), dtype=complex)
        for i in range(nt):
            for j in range(nt):
                if i != j:
                    als[:,i,j] = np.array([simps(-hfunc(x)[:,i,j]*np.conjugate(psis[i][l])*psis[j][l],x) for l in range(lmax)])
                    addr[:,i,j] = np.array([simps(ddrfunc(x)[:,i,j]*(np.conjugate(psis[i][l])*dpsis[j][l] - np.conjugate(dpsis[i][l])*psis[j][l]),x) for l in range(lmax)])
                    # sigmas[k, i, j] = 4e-16 * np.pi *np.sum(np.array([(2*l+1)*np.abs(als[l,i,j]+addr[l,i,j])**2 for l in range(lmax)]))/(q[j]**2)
        for i in range(nt):
            for j in range(i+1, nt):
                    if any([np.abs(c) >= 1e-6 for c in np.real(als[:,i,j])]):
                        plt.plot(js, np.real(als[:,i,j]), 'o-', label='$\Re A_{2p_9 \\rightarrow 2p_8}$')
                    if any([np.abs(c) >= 1e-6 for c in np.imag(als[:,i,j])]):
                        plt.plot(js, np.imag(als[:,i,j]), 'o-', label='$\Im A_{2p_9 \\rightarrow 2p_8}$')
                    if any([np.abs(c) >= 1e-6 for c in np.real(addr[:,i,j])]):
                        plt.plot(js, np.real(addr[:,i,j]), 'o-', label='$\Re B_{2p_9 \\rightarrow 2p_8}$')
                    if any([np.abs(c) >= 1e-6 for c in np.imag(addr[:,i,j])]):
                        plt.plot(js, np.imag(addr[:,i,j]), 'o-', label='$\Im B_{2p_9 \\rightarrow 2p_8}$')
        plt.xlabel('$L$')
        plt.legend()
        plt.savefig(f'images/Amps_{int(np.round(e * 4.637 + np.real(hls[-1,0,0])))}.eps')
        plt.close()
    # print(sigmadiff[k])
# tofile = np.zeros((len(es), n*n+1))
# tofile[:,0] = es * 4.637 + np.real(hls[-1,0,0])
# for i in range(n):
#     for j in range(n):
#         tofile[:,i*n+j+1] = sigmas[:,i,j]
# np.savetxt(f'{dir}/sigmas_total_new.txt', tofile, fmt='%.6e', delimiter='\t', header=headerfinal, comments='')
# np.savetxt(f'{dir}/sigmas_diff.txt', sigmadiff, fmt='%.6e', delimiter='\t', header=headerfinal, comments='')
