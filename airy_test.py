from funcs import load_matrices, matrix_funcs
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy, airye, riccati_jn, riccati_yn, iv, kv
import os
from airy_props import *

# dirs = ['H_test_3_im']
dir = 'H_test_3_im'
r, hls, ddrls = load_matrices(dir)
print(dir)
emin = hls[-1,0,0]
n = len(hls[0])
shift = np.array([emin*np.eye(len(hls[0]), dtype=complex)]*len(r))
vd = (hls-shift)/4.637
hfunc, ddrfunc = matrix_funcs(r, vd, ddrls)
emax = np.abs(np.array(hfunc(r[0])[-1,-1]))
w = lambda rs, l, e: np.array([e*np.eye(n)-(hfunc(r)+l*(l*1)/(r**2)*np.eye(n)) for r in rs])
wp = lambda rs, l: np.array([-(hfunc(r,1)-2*l*(l*1)/(r**3)*np.eye(n)) for r in rs])
es = np.array([1])
smats = {}
for i in range(n):
    emax = np.abs(np.array(hfunc(r[0])[i,i]))
    emin = np.abs(np.array(hfunc(r[-1])[i,i]))
    es = np.concatenate((es, np.power(10, np.arange(np.log10(emin if emin!=0 else 1), np.log10(emax), 0.01))))
es = np.unique(np.round(np.sort(es)))
sigmas = np.zeros((len(es), n, n))
ecm = es*4.637+np.real(hls[-1,0,0])
for ei in range(len(es)):
    e = es[ei]
    lmax = int(np.round(np.sqrt(e)*np.max(r[-1]))*0.8)
    ks = np.sqrt(np.abs(np.diag(e*np.eye(n)-hfunc(r[-1]))))
    yf = np.zeros((lmax, n, n), dtype=complex)
    nt = np.sum(np.diag(vd[-1]) < e)
    for l in range(lmax):
        yf[l] = airy_imb_prop(r, w(r[:-1]+np.diff(r)/2, l, e), wp(r[:-1]+np.diff(r)/2, l), n)
    ytot = np.array([(yf[l]+np.matrix(yf[l]).H)/2 for l in range(lmax)])
    jmat = np.zeros((lmax, n, n))
    nmat = np.zeros((lmax, n, n))
    jpmat = np.zeros((lmax, n, n))
    npmat = np.zeros((lmax, n, n))
    for j in range(n):
        if j <= nt:
            prej, prejp = riccati_jn(lmax-1, np.real(ks[j])*r[-1])
            pren, prenp = riccati_yn(lmax-1, np.real(ks[j])*r[-1])
            jmat[:,j,j] = prej/np.sqrt(np.real(ks[j]))
            nmat[:,j,j] = pren/np.sqrt(np.real(ks[j]))
            jpmat[:,j,j] = np.sqrt(np.real(ks[j])) * prejp
            npmat[:,j,j] = np.sqrt(np.real(ks[j])) * prenp
        else:
            jmat[:,j,j], prejp = riccati_in(lmax-1, np.real(ks[j])*r[-1])
            nmat[:,j,j], prenp = riccati_kn(lmax-1, np.real(ks[j])*r[-1])
            jpmat[:,j,j] = ks[j] * prejp
            npmat[:,j,j] = ks[j] * prenp
    rmat = np.array([-np.matrix(npmat[l] - ytot[l] @ nmat[l]).I @ np.matrix(jpmat[l] - ytot[l] @ jmat[l]) for l in range(lmax)])
    smat = np.array([np.matrix(np.eye(nt) + 1j*rmat[l,:nt,:nt]).I @ np.matrix(np.eye(nt) - 1j*rmat[l,:nt,:nt]) if not np.all(np.isnan(rmat[l,:nt,:nt])) else np.eye(nt) for l in range(lmax)])
    print(smat[0])
    smats[f'E = {ecm[ei]}'] = smat


