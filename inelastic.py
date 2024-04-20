import numpy as np
import matplotlib.pyplot as plt
from funcs import *
from scipy.integrate import solve_ivp, simps
from scipy.special import riccati_jn, riccati_yn
from numba import jit

@jit(nopython=True)
def extract_block_diag(a, n, k=0):
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("Only 2-D arrays handled")
    if not (n > 0):
        raise ValueError("Must have n >= 0")

    if k > 0:
        a = a[:,n*k:] 
    else:
        a = a[-n*k:]

    n_blocks = min(a.shape[0]//n, a.shape[1]//n)

    new_shape = (n_blocks, n, n)
    new_strides = (n*a.strides[0] + n*a.strides[1],
                   a.strides[0], a.strides[1])

    return np.lib.stride_tricks.as_strided(a, new_shape, new_strides)


def inelq(e, l, h, ddr, psi, x):
    f = np.zeros(len(psi), dtype=complex)
    f[:(len(psi)//2)] = psi[(len(psi)//2):]
    h1 = (e - l*(l+1) / x**2) * np.eye(len(psi)//2)
    h2 = h(x) - ddr(x,1)
    hu = -h1 - h2
    hp = -2*ddr(x)
    f[(len(psi)//2):] = (np.matmul(hu, psi[:(len(psi)//2)].transpose()) + np.matmul(hp, psi[(len(psi)//2):].transpose())).transpose()
    return f


def get_state_curve(h, i, x):
    return np.real(np.diag(h(x))[i])


def mat_power(m, p):
    lam, v = np.linalg.eig(m)
    f = np.diag(np.power(lam, p))
    return np.dot(v, np.dot(f, np.linalg.inv(v)))


dir = 'H_test_3'
r, hls, ddrls = load_matrices(dir)
emin = hls[-1,0,0]
n = len(hls[0])
shift = np.array([emin*np.eye(len(hls[0]), dtype=complex)]*len(r))
vd = (hls-shift)/4.637

hfunc, ddrfunc = matrix_funcs(r, vd, ddrls)
emax = np.abs(np.array(hfunc(r[0])[-1,-1]))
es = np.power(10, np.arange(0, np.log10(1.5*emax), 0.01))
# es = np.power(10, np.array([np.log10(emin), np.log10(emax)]))
# es = np.power(10, np.array([(np.log10(emin)+np.log10(emax))/2]))
x = np.arange(np.min(r), np.max(r), 0.005)
sigmas = np.zeros((len(es), n, n))
headerfinal = 'E, cm-1\t'
for i in range(n):
    for j in range(n):
        headerfinal += f'Sigma_{i}{j}, cm2\t'
for k in range(len(es)):
    e = es[k]
    lmax = int(np.round(3*np.sqrt(e)*np.max(r[-1])/4))
    q = np.abs(np.sqrt(e-np.diag(hfunc(r[-1]))))
    js = np.arange(0,int(lmax),1)
    psis = {}
    dpsis = {}
    nt = np.sum(np.diag(vd[-1]) < e)
    for i in range(nt):
        en = np.array([e-np.real(vd[-1,i,i])])
        psi, dpsi = norm_sol(x, en, js, lambda z: get_state_curve(hfunc, i, z)-vd[-1,i,i])
        sigmas[k, i, i] = 4e-16 * np.pi * mul_sigma_calc(psi, dpsi, x, en, js)[0]
        psis[i] = psi
        dpsis[i] = dpsi
    if nt > 1:
        for i in range(nt):
            for j in range(nt):
                if i != j:
                    als = np.array([simps(-hfunc(x)[:,i,j]*np.conjugate(psis[i][l])*psis[j][l],x) for l in range(lmax)])
                    addr = np.array([simps(ddrfunc(x)[:,i,j]*(np.conjugate(psis[i][l])*dpsis[j][l] - np.conjugate(dpsis[i][l])*psis[j][l]),x) for l in range(lmax)])
                    sigmas[k, i, j] = 4e-16 * np.pi *np.sum(np.array([(2*l+1)*np.abs(als[l]+addr[l])**2 for l in range(lmax)]))/(q[i]*(q[j]**3))
    print(e * 4.637 + np.real(hls[-1,0,0]))
    print(sigmas[k])
tofile = np.zeros((len(es), n*n+1))
tofile[:,0] = es * 4.637 + np.real(hls[-1,0,0])
for i in range(n):
    for j in range(n):
        tofile[:,i*n+j+1] = sigmas[:,i,j]

np.savetxt(f'{dir}/sigmas_total.txt', tofile, fmt='%.6e', delimiter='\t', header=headerfinal, comments='')

print(np.diag(hls[-1]))