import numpy as np
import matplotlib.pyplot as plt
from funcs import *
from scipy.integrate import solve_ivp, simps
from scipy.special import riccati_jn, riccati_yn
from scipy.linalg import expm
from numba import jit


def get_state_curve(h, i, x):
    return np.real(np.diag(h(x))[i])


dir = 'H_1_p'
r, hls, ddrls = load_matrices(dir)
emin = hls[-1,0,0]
n = len(hls[0])
shift = np.array([emin*np.eye(len(hls[0]), dtype=complex)]*len(r))
vd = (hls-shift)/4.637

hfunc, ddrfunc = matrix_funcs(r, vd, ddrls)
es = np.array([1])
for i in range(n):
    emax = np.abs(np.array(hfunc(r[0])[i,i]))
    emin = np.abs(np.array(hfunc(r[-1])[i,i]))
    es = np.concatenate((es, np.power(10, np.arange(np.log10(emin if emin!=0 else 1), np.log10(emax), 0.01))))
es = np.unique(np.round(np.sort(es)))
x = np.arange(np.min(r), np.max(r)+0.005, 0.005)
sigmas = np.zeros((len(es), n, n))
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
    s0 = np.zeros((lmax, nt, nt), dtype=complex)
    s = np.zeros((lmax, nt, nt), dtype=complex)
    for i in range(nt):
        en = np.array([e-np.real(vd[-1,i,i])])
        psi, dpsi = norm_sol(x, en, js, lambda z: get_state_curve(hfunc, i, z)-vd[-1,i,i])
        sigmas[k, i, i] = 4e-16 * np.pi * mul_sigma_calc(psi, dpsi, x, en, js)[0]
        psis[i] = psi
        dpsis[i] = dpsi
        s0[:,i,i] = np.exp(1j * s_phase_calc(psi, dpsi, x, en, js))
    d = np.zeros((lmax, nt, nt), dtype=complex)
    if nt > 1:
        for i in range(nt):
            for j in range(nt):
                if i != j:
                    als = np.array([simps(-hfunc(x)[:,i,j]*np.conjugate(psis[i][l])*psis[j][l],x) for l in range(lmax)])
                    addr = np.array([simps(ddrfunc(x)[:,i,j]*(np.conjugate(psis[i][l])*dpsis[j][l] - np.conjugate(dpsis[i][l])*psis[j][l]),x) for l in range(lmax)])
                    d[:,i,j] = 2*(als+addr)
        for l in range(lmax):
            s[l] = np.dot(s0[l], np.dot(expm(1j*d[l]), s0[l]))
        for i in range(nt):
            for j in range(nt):
                sigmas[k, i, j] = 1e-16 * np.pi * np.sum(np.array([(2*l+1)*np.abs(s[l,i,j]-float(i==j))**2 for l in range(lmax)]))/(q[j]**2)
    print(e * 4.637 + np.real(hls[-1,0,0]))
    print(sigmas[k])

tofile = np.zeros((len(es), n*n+1))
tofile[:,0] = es * 4.637 + np.real(hls[-1,0,0])
for i in range(n):
    for j in range(n):
        tofile[:,i*n+j+1] = sigmas[:,i,j]

np.savetxt(f'{dir}/sigmas_total_exp.txt', tofile, fmt='%.6e', delimiter='\t', header=headerfinal, comments='')