import numpy as np
from scipy.special import airy, airye, iv, kv


def riccati_in(n, x):
    f = np.array([np.sqrt(x)*iv(l+(1./2), x) for l in range(n+1)])
    fp = np.array([(x*(iv(l-(1./2), x)+iv(l+(3./2), x))+iv(l+(1./2), x))/(2*np.sqrt(x)) for l in range(n+1)])
    return f, fp


def riccati_kn(n, x):
    f = np.array([np.sqrt(x)*kv(l+(1./2), x) for l in range(n+1)])
    fp = np.array([(-x*(kv(l-(1./2), x)+kv(l+(3./2), x))+kv(l+(1./2), x))/(2*np.sqrt(x)) for l in range(n+1)])
    return f, fp


def airy_imb_prop(r, warr, wparr, n):
    k2 = np.zeros((len(warr), n))
    t = np.zeros((len(warr), n, n), dtype=complex)
    wpt = np.zeros((len(warr), n, n), dtype=complex)
    for i in range(len(warr)):
        k2[i], t[i] = np.linalg.eigh(warr[i])
        wpt[i] = np.matrix(t[i]).H @ (np.matrix(wparr[i]) @ np.matrix(t[i]))
    p = np.array([np.matrix(t[i]).H @ np.matrix(t[i-1]) if i > 0 else np.matrix(t[0]) for i in range(len(warr))])
    c1 = np.zeros((len(warr), n, n), dtype=complex)
    c2 = np.zeros((len(warr), n, n), dtype=complex)
    c3 = np.zeros((len(warr), n, n), dtype=complex)
    c4 = np.zeros((len(warr), n, n), dtype=complex)
    for i in range(len(warr)):
        alpha = np.array([-(wpt[i,j,j])**(1./3.) if np.allclose(-(wpt[i,j,j])**(1./3.), np.real(-(wpt[i,j,j])**(1./3.)))
                          else -(wpt[i,j,j])**(1./3.)*np.exp(-1j*np.angle(-(wpt[i,j,j])**(1./3.))) for j in range(n)])
        beta = k2[i]/np.diag(wpt[i])
        rhob = np.diff(r)[i]/2
        x = np.array([[alpha[j]*(-rhob+beta[j]), alpha[j]*(rhob+beta[j])] for j in range(n)])
        for j in range(n):
            if np.any(np.real(x[j]) > 100):
                dksi = np.diff((2./3)*(x[j]**(3./2)))[0]
                aie, aiep, bie, biep = airye(np.real(x[j]))
                c1[i,j,j] = np.pi*(np.exp(-dksi)*aie[1]*biep[0]-np.exp(dksi)*bie[1]*aiep[0])
                c2[i,j,j] = np.pi*(np.exp(dksi)*bie[1]*aie[0]-np.exp(-dksi)*aie[1]*bie[0])/alpha[j]
                c3[i,j,j] = np.pi*alpha[j]*(np.exp(-dksi)*aiep[1]*biep[0]-np.exp(dksi)*biep[1]*aiep[0])
                c4[i,j,j] = np.pi*(np.exp(dksi)*biep[1]*aie[0]-np.exp(-dksi)*aiep[1]*bie[0])
            else:
                ai, aip, bi, bip = airy(np.real(x[j]))
                c1[i,j,j] = np.pi*(ai[1]*bip[0]-bi[1]*aip[0]) #/wr[j]
                c2[i,j,j] = np.pi*(bi[1]*ai[0]-ai[1]*bi[0])/alpha[j]
                c3[i,j,j] = np.pi*alpha[j]*(aip[1]*bip[0]-bip[1]*aip[0]) #/wr[j]
                c4[i,j,j] = np.pi*(bip[1]*ai[0]-aip[1]*bi[0]) #/wr[j]
    d1 = np.array([np.matrix(c2[i]).I @ c1[i] for i in range(len(warr))])
    d2 = np.array([np.matrix(c2[i]).I for i in range(len(warr))])
    d3 = np.array([c4[i] @ np.matrix(c2[i]).I @ c1[i] - c3[i] for i in range(len(warr))])
    d4 = np.array([c4[i] @ np.matrix(c2[i]).I for i in range(len(warr))])
    y = np.zeros((len(r), n, n), dtype=complex)
    y[0] = 1e308*np.eye(n, dtype=complex)
    for i in range(len(warr)):
        prey = d4[i] - d3[i] @ np.matrix(y[i]+d1[i]).I @ d2[i]
        y[i+1] = np.matrix(p[i]) @ (prey @ np.matrix(p[i]).H)
    return np.matrix(t[-1]).H @ (y[-1] @ np.matrix(t[-1]))