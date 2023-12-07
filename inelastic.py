import numpy as np
import matplotlib.pyplot as plt
from funcs import *
from scipy.integrate import solve_ivp, simps
from scipy.special import ivp, kvp


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


def sph_in(n, x, m=0):
    return np.sqrt(x) * ivp(n+(1/2), x, m)


def sph_kn(n, x, m=0):
    return np.sqrt(x) * kvp(n+(1/2), x, m)


def inelq(e, l, h, ddr, psi, x):
    f = np.zeros(len(psi), dtype=complex)
    f[:(len(psi)//2)] = psi[(len(psi)//2):]
    h1 = (e - l*(l+1) / x**2) * np.eye(len(psi)//2)
    h2 = h(x) - ddr(x,1)
    hu = -h1 - h2
    hp = -2*ddr(x)
    f[(len(psi)//2):] = (np.dot(hu, psi[:(len(psi)//2)].transpose()) + np.dot(hp, psi[(len(psi)//2):].transpose())).transpose()
    return f


def yeqsol(e, l, h, rmin, rmax, nstep=1000):
    n = len(h(rmin))
    q = np.abs(np.sqrt(e-np.diag(h(rmax))))
    # ethres = np.diag(h(rmax))
    jm = np.diag(np.array([sph_jn(l,q[i]*rmax)/np.sqrt(q[i]) for i in range(n)]))
    nm = np.diag(np.array([sph_yn(l,q[i]*rmax)/np.sqrt(q[i]) for i in range(n)]))
    jpm = np.diag(np.array([sph_jn(l,q[i]*rmax,True)*np.sqrt(q[i]) for i in range(n)]))
    npm = np.diag(np.array([sph_yn(l,q[i]*rmax,True)*np.sqrt(q[i]) for i in range(n)]))
    y = np.zeros((nstep+1,n,n), dtype=complex)
    u = np.zeros((nstep+1,n,n), dtype=complex)
    w = np.zeros(nstep+1, dtype=complex)
    v = lambda r: (e - l*(l+1) / r**2) * np.eye(n)-h(r)
    w[0] = 1
    w[1::2] = 4
    w[2::2] = 2
    w[-1] = 1
    x = np.linspace(rmin, rmax, nstep+1)
    dx = np.diff(x)[0]
    y[0] = 1e20 * np.eye(n, dtype=complex)
    for i in range(1,len(x),2):
        u[i-1] = v(x[i-1])
        u[i] = np.dot(np.linalg.inv(np.eye(n, dtype=complex)-(dx**2)*v(x[i])/6), v(x[i]))
    for i in range(nstep):
        y[i+1] = np.dot(np.linalg.inv(np.eye(n, dtype=complex)+dx*y[i]), y[i]) - (dx*w[i]*u[i]/3)
    left = np.dot(y[-1], nm) - npm
    right = np.dot(y[-1], jm) - jpm
    k = -np.dot(np.linalg.inv(left), right)
    s = np.dot(np.linalg.inv(np.eye(n, dtype=complex)+1j*k),np.eye(n, dtype=complex)-1j*k)
    return s


def yeqsolfull(e, h, rmin, rmax, nstep=1000):
    n = len(h(rmin))
    q = np.abs(np.sqrt(e-np.diag(h(rmax))))
    lmax = int(np.round(3*np.sqrt(e)*rmax/4))
    lm = np.eye(lmax)
    hm = np.eye(n)
    qjs = np.transpose([np.kron(np.arange(lmax), np.ones(n)), np.kron(np.ones(lmax), q)])
    vcenter = lambda x: np.kron(np.diag(np.arange(lmax)*(np.arange(lmax)+1)/(x**2)), hm)
    vpot = lambda x: np.kron(lm, hfunc(x))
    # ethres = np.diag(h(rmax))
    jm = np.diag(np.array([sph_jn(int(qjs[i,0]),qjs[i,1]*rmax)/np.sqrt(qjs[i,1]) for i in range(len(qjs))]))
    nm = np.diag(np.array([sph_yn(int(qjs[i,0]),qjs[i,1]*rmax)/np.sqrt(qjs[i,1]) for i in range(len(qjs))]))
    jpm = np.diag(np.array([sph_jn(int(qjs[i,0]),qjs[i,1]*rmax,True)*np.sqrt(qjs[i,1]) for i in range(len(qjs))]))
    npm = np.diag(np.array([sph_yn(int(qjs[i,0]),qjs[i,1]*rmax,True)*np.sqrt(qjs[i,1]) for i in range(len(qjs))]))
    y = np.zeros((nstep+1,n*lmax,n*lmax), dtype=complex)
    u = np.zeros((nstep+1,n*lmax,n*lmax), dtype=complex)
    w = np.zeros(nstep+1, dtype=complex)
    v = lambda r: e*np.eye(n*lmax) - vcenter(r) - vpot(r)
    w[0] = 1
    w[1::2] = 4
    w[2::2] = 2
    w[-1] = 1
    x = np.linspace(rmin, rmax, nstep+1)
    dx = np.diff(x)[0]
    y[0] = 1e20 * np.eye(n*lmax, dtype=complex)
    for i in range(1,len(x),2):
        u[i-1] = v(x[i-1])
        u[i] = np.dot(np.linalg.inv(np.eye(n*lmax, dtype=complex)-(dx**2)*v(x[i])/6), v(x[i]))
    for i in range(nstep):
        y[i+1] = np.dot(np.linalg.inv(np.eye(n*lmax, dtype=complex)+dx*y[i]), y[i]) - (dx*w[i]*u[i]/3)
    left = np.dot(y[-1], nm) - npm
    right = np.dot(y[-1], jm) - jpm
    k = -np.dot(np.linalg.inv(left), right)
    sbig = np.dot(np.linalg.inv(np.eye(n*lmax, dtype=complex)+1j*k),np.eye(n*lmax, dtype=complex)-1j*k)
    return sbig


def get_state_curve(h, i, x):
    return np.diag(h(x))[i]


dir = 'H_2'
r, hls, ddrls = load_matrices(dir)
emin = hls[-1,0,0]
n = len(hls[0])
shift = np.array([emin*np.eye(len(hls[0]), dtype=complex)]*len(r))
vd = (hls-shift)/4.637

hfunc, ddrfunc = matrix_funcs(r, vd, ddrls)
emin = np.ceil(np.abs(hfunc(r[-1])[-1,-1]))
emax = np.abs(np.array(hfunc(r[0])[-1,-1]))
es = np.power(10, np.arange(np.log10(emin), np.log10(emax), 0.2))
sigmas = np.zeros((len(es), n, n))
total = np.zeros((len(es), n*n+1))
total[:,0] = es
for i in range(len(es)):
    e = es[i]
    q2inv = np.linalg.inv(np.diag(np.diag(np.abs(e-hfunc(r[-1])))))

    sbig = yeqsolfull(e, hfunc, r[0], r[-1], nstep=400)
    s = extract_block_diag(sbig, n)
    sigmas[i] = np.dot(np.sum(np.array([(2*i+1)*(np.abs(s[i]-np.eye(n, dtype=complex)))**2 for i in range(len(s))]),axis=0), q2inv)*np.pi*1e-16

    print(np.diag(np.abs(e-hfunc(r[-1]))) * 4.637 * 1.42879)
    print(sigmas[i])

for i in range(n):
    for j in range(n):
        total[:,n*i+j+1] = sigmas[:,i,j]

np.savetxt(f'{dir}/sigmas_inel.txt', total, fmt='%.6e', delimiter='\t')