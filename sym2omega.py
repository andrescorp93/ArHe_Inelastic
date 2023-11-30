import os
import numpy as np
import matplotlib.pyplot as plt
from funcs import *

so_outs = [filename for filename in os.listdir() if filename.find('so')!=-1 and filename.find('out')!=-1]
so_outs.sort()

rs = []
e0 = []
hso = []
rehso = []
imhso = []
multablesym = [[1,2,3,4],[2,1,3,4],[3,4,1,2],[4,3,2,1]]


in_states = [{'n': 1, 'sym': 1, 'lam': 0, 's': 1, 'sz': 1}, {'n': 2, 'sym': 1, 'lam': 0, 's': 1, 'sz': 1}, {'n': 3, 'sym': 1, 'lam': 2, 's': 1, 'sz': 1},
 {'n': 4, 'sym': 1, 'lam': 0, 's': 1, 'sz': 1}, {'n': 1, 'sym': 1, 'lam': 0, 's': 1, 'sz': 0}, {'n': 2, 'sym': 1, 'lam': 0, 's': 1, 'sz': 0},
 {'n': 3, 'sym': 1, 'lam': 2, 's': 1, 'sz': 0}, {'n': 4, 'sym': 1, 'lam': 0, 's': 1, 'sz': 0}, {'n': 1, 'sym': 1, 'lam': 0, 's': 1, 'sz':-1},
 {'n': 2, 'sym': 1, 'lam': 0, 's': 1, 'sz':-1}, {'n': 3, 'sym': 1, 'lam': 2, 's': 1, 'sz':-1}, {'n': 4, 'sym': 1, 'lam': 0, 's': 1, 'sz':-1},
 {'n': 1, 'sym': 2, 'lam': 1, 's': 1, 'sz': 1}, {'n': 2, 'sym': 2, 'lam': 1, 's': 1, 'sz': 1}, {'n': 3, 'sym': 2, 'lam': 1, 's': 1, 'sz': 1},
 {'n': 1, 'sym': 2, 'lam': 1, 's': 1, 'sz': 0}, {'n': 2, 'sym': 2, 'lam': 1, 's': 1, 'sz': 0}, {'n': 3, 'sym': 2, 'lam': 1, 's': 1, 'sz': 0},
 {'n': 1, 'sym': 2, 'lam': 1, 's': 1, 'sz':-1}, {'n': 2, 'sym': 2, 'lam': 1, 's': 1, 'sz':-1}, {'n': 3, 'sym': 2, 'lam': 1, 's': 1, 'sz':-1},
 {'n': 1, 'sym': 3, 'lam': 1, 's': 1, 'sz': 1}, {'n': 2, 'sym': 3, 'lam': 1, 's': 1, 'sz': 1}, {'n': 3, 'sym': 3, 'lam': 1, 's': 1, 'sz': 1},
 {'n': 1, 'sym': 3, 'lam': 1, 's': 1, 'sz': 0}, {'n': 2, 'sym': 3, 'lam': 1, 's': 1, 'sz': 0}, {'n': 3, 'sym': 3, 'lam': 1, 's': 1, 'sz': 0},
 {'n': 1, 'sym': 3, 'lam': 1, 's': 1, 'sz':-1}, {'n': 2, 'sym': 3, 'lam': 1, 's': 1, 'sz':-1}, {'n': 3, 'sym': 3, 'lam': 1, 's': 1, 'sz':-1},
 {'n': 1, 'sym': 4, 'lam': 2, 's': 1, 'sz': 1}, {'n': 2, 'sym': 4, 'lam': 0, 's': 1, 'sz': 1}, {'n': 1, 'sym': 4, 'lam': 2, 's': 1, 'sz': 0},
 {'n': 2, 'sym': 4, 'lam': 0, 's': 1, 'sz': 0}, {'n': 1, 'sym': 4, 'lam': 2, 's': 1, 'sz':-1}, {'n': 2, 'sym': 4, 'lam': 0, 's': 1, 'sz':-1},
 {'n': 1, 'sym': 1, 'lam': 0, 's': 0, 'sz': 0}, {'n': 2, 'sym': 1, 'lam': 0, 's': 0, 'sz': 0}, {'n': 3, 'sym': 1, 'lam': 2, 's': 0, 'sz': 0},
 {'n': 4, 'sym': 1, 'lam': 0, 's': 0, 'sz': 0}, {'n': 5, 'sym': 1, 'lam': 0, 's': 0, 'sz': 0}, {'n': 1, 'sym': 2, 'lam': 1, 's': 0, 'sz': 0},
 {'n': 2, 'sym': 2, 'lam': 1, 's': 0, 'sz': 0}, {'n': 3, 'sym': 2, 'lam': 1, 's': 0, 'sz': 0}, {'n': 1, 'sym': 3, 'lam': 1, 's': 0, 'sz': 0},
 {'n': 2, 'sym': 3, 'lam': 1, 's': 0, 'sz': 0}, {'n': 3, 'sym': 3, 'lam': 1, 's': 0, 'sz': 0}, {'n': 1, 'sym': 4, 'lam': 2, 's': 0, 'sz': 0},
 {'n': 2, 'sym': 4, 'lam': 0, 's': 0, 'sz': 0}]

out_states = [{'n': 1, 'sign': +1, 'lam': 0, 's': 1, 'sz': 1}, {'n': 2, 'sign': +1, 'lam': 0, 's': 1, 'sz': 1}, {'n': 3, 'sign': +1, 'lam': 2, 's': 1, 'sz': 1},
 {'n': 4, 'sign': +1, 'lam': 0, 's': 1, 'sz': 1}, {'n': 1, 'sign': +1, 'lam': 0, 's': 1, 'sz': 0}, {'n': 2, 'sign': +1, 'lam': 0, 's': 1, 'sz': 0},
 {'n': 3, 'sign': +1, 'lam': 2, 's': 1, 'sz': 0}, {'n': 4, 'sign': +1, 'lam': 0, 's': 1, 'sz': 0}, {'n': 1, 'sign': +1, 'lam': 0, 's': 1, 'sz':-1},
 {'n': 2, 'sign': +1, 'lam': 0, 's': 1, 'sz':-1}, {'n': 3, 'sign': +1, 'lam': 2, 's': 1, 'sz':-1}, {'n': 4, 'sign': +1, 'lam': 0, 's': 1, 'sz':-1},
 {'n': 1, 'sign': +1, 'lam': 1, 's': 1, 'sz': 1}, {'n': 2, 'sign': +1, 'lam': 1, 's': 1, 'sz': 1}, {'n': 3, 'sign': +1, 'lam': 1, 's': 1, 'sz': 1},
 {'n': 1, 'sign': +1, 'lam': 1, 's': 1, 'sz': 0}, {'n': 2, 'sign': +1, 'lam': 1, 's': 1, 'sz': 0}, {'n': 3, 'sign': +1, 'lam': 1, 's': 1, 'sz': 0},
 {'n': 1, 'sign': +1, 'lam': 1, 's': 1, 'sz':-1}, {'n': 2, 'sign': +1, 'lam': 1, 's': 1, 'sz':-1}, {'n': 3, 'sign': +1, 'lam': 1, 's': 1, 'sz':-1},
 {'n': 1, 'sign': -1, 'lam': 1, 's': 1, 'sz': 1}, {'n': 2, 'sign': -1, 'lam': 1, 's': 1, 'sz': 1}, {'n': 3, 'sign': -1, 'lam': 1, 's': 1, 'sz': 1},
 {'n': 1, 'sign': -1, 'lam': 1, 's': 1, 'sz': 0}, {'n': 2, 'sign': -1, 'lam': 1, 's': 1, 'sz': 0}, {'n': 3, 'sign': -1, 'lam': 1, 's': 1, 'sz': 0},
 {'n': 1, 'sign': -1, 'lam': 1, 's': 1, 'sz':-1}, {'n': 2, 'sign': -1, 'lam': 1, 's': 1, 'sz':-1}, {'n': 3, 'sign': -1, 'lam': 1, 's': 1, 'sz':-1},
 {'n': 1, 'sign': -1, 'lam': 2, 's': 1, 'sz': 1}, {'n': 2, 'sign': -1, 'lam': 0, 's': 1, 'sz': 1}, {'n': 1, 'sign': -1, 'lam': 2, 's': 1, 'sz': 0},
 {'n': 2, 'sign': -1, 'lam': 0, 's': 1, 'sz': 0}, {'n': 1, 'sign': -1, 'lam': 2, 's': 1, 'sz':-1}, {'n': 2, 'sign': -1, 'lam': 0, 's': 1, 'sz':-1},
 {'n': 1, 'sign': +1, 'lam': 0, 's': 0, 'sz': 0}, {'n': 2, 'sign': +1, 'lam': 0, 's': 0, 'sz': 0}, {'n': 3, 'sign': +1, 'lam': 2, 's': 0, 'sz': 0},
 {'n': 4, 'sign': +1, 'lam': 0, 's': 0, 'sz': 0}, {'n': 5, 'sign': +1, 'lam': 0, 's': 0, 'sz': 0}, {'n': 1, 'sign': +1, 'lam': 1, 's': 0, 'sz': 0},
 {'n': 2, 'sign': +1, 'lam': 1, 's': 0, 'sz': 0}, {'n': 3, 'sign': +1, 'lam': 1, 's': 0, 'sz': 0}, {'n': 1, 'sign': -1, 'lam': 1, 's': 0, 'sz': 0},
 {'n': 2, 'sign': -1, 'lam': 1, 's': 0, 'sz': 0}, {'n': 3, 'sign': -1, 'lam': 1, 's': 0, 'sz': 0}, {'n': 1, 'sign': -1, 'lam': 2, 's': 0, 'sz': 0},
 {'n': 2, 'sign': -1, 'lam': 0, 's': 0, 'sz': 0}]

u = trans_matrix(in_states, out_states)

for f in so_outs:
    txt = open(f).readlines()
    for i in range(len(txt)):
        if txt[i].find('r=[')!=-1:
            rs = [*rs, *txt[i][4:-2].split(',')]
        if txt[i].find('Spin-Orbit Matrix (CM-1)')!=-1:
            e0.append(float(txt[i-7].split()[-1]))
            result = []
            txtm = [s for s in txt[i+3:i+751] if len(s.split())>0]
            for j in range(len(txtm)):
                if txtm[j].find('State')!=-1:
                    part = txtm[j+1:j+99]
                    if len(result) == 0:
                        result = [s[19:-1] for s in part]
                    else:
                        for k in range(len(part)):
                            result[k] += part[k][19:-1]
            hso.append(result)

r = np.array([float(s) for s in rs])
e0 = np.array(e0)
shift = (e0-e0[-1])*219474.63
h0 = np.array([np.eye(49)*s for s in shift])

for i in range(len(r)):
    re_hso = []
    im_hso = []
    for j in range(0, len(hso[i]), 2):
        re_hso.append([float(s) for s in hso[i][j].split()])
        im_hso.append([float(s) for s in hso[i][j+1].split()])
    rehso.append(np.array(re_hso))
    imhso.append(np.array(im_hso))

rehso = np.array(rehso) + h0
imhso = np.array(imhso)

rehso = flip_at_point(rehso, 15, 2, 3)
rehso = correct_diag_points(rehso, 68, 2, 3)
rehso = flip_at_point(rehso, 15, 6, 7)
rehso = correct_diag_points(rehso, 68, 6, 7)
rehso = flip_at_point(rehso, 15, 10, 11)
rehso = correct_diag_points(rehso, 68, 10, 11)
rehso = flip_at_point(rehso, 14, 38, 39)
rehso = correct_diag_point(rehso, 14, 38, 39)
rehso = correct_diag_points(rehso, 71, 38, 39)


imhso = flip_at_point(imhso, 15, 2, 3)
imhso = flip_at_point(imhso, 15, 6, 7)
imhso = flip_at_point(imhso, 15, 10, 11)
imhso = flip_at_point(imhso, 14, 38, 39)
       
for i in range(49):
    for j in range(49):
        if i != j:
            if any([np.abs(c) >= 1e-2 for c in rehso[:,i,j]]):
                if rehso[-1,i,j] != 0:
                    rehso[:,i,j] = np.sign(rehso[-1,i,j])*np.abs(rehso[:,i,j])
                else:
                    rehso[:,i,j] = np.sign(rehso[0,i,j])*np.abs(rehso[:,i,j])
            if any([np.abs(c) >= 1e-2 for c in imhso[:,i,j]]):
                if imhso[-1,i,j] != 0:
                    imhso[:,i,j] = np.sign(imhso[-1,i,j])*np.abs(imhso[:,i,j])
                else:
                    imhso[:,i,j] = np.sign(imhso[0,i,j])*np.abs(imhso[:,i,j])

hsom = rehso+1j*imhso
hsom = correct_delta(in_states, hsom)
hsomomega = hsom.copy()

ddr_outs = os.listdir('repaired_NACME')
ddr_read = {}
for f in ddr_outs:
    data = np.loadtxt(f'repaired_NACME/{f}', skiprows=1).transpose()
    header = open(f'repaired_NACME/{f}').readlines()[0].split()
    for i in range(len(header)):
        ddr_read[header[i]] = data[i]

r = ddr_read['R']
ddr = np.zeros((len(r),49,49), dtype=complex)

for k in ddr_read.keys():
    if k != 'R':
        ids = [int(i) for i in k[1:-1].split('|')[::2]]
        ddr[:, ids[0], ids[1]] = ddr_read[k]
        ddr[:, ids[1], ids[0]] = -ddr_read[k]

ddromega = ddr.copy()

for i in range(len(hsomomega)):
    hsomomega[i] = np.dot(np.matrix(u).H, np.dot(hsom[i], u))
    ddromega[i] = np.dot(np.matrix(u).H, np.dot(ddr[i], u))

omegastates = {omega: [i for i in range(len(out_states)) if omega == get_lambda(out_states[i])] for omega in range(4)}

hsomomegaparsed = {omega: np.zeros((len(r),len(omegastates[omega]),len(omegastates[omega])), dtype=complex) for omega in range(4)}

ddromegaparsed = {omega: np.zeros((len(r),len(omegastates[omega]),len(omegastates[omega])), dtype=complex) for omega in range(4)}

for omega in range(4):
    s = omegastates[omega]
    for i in range(len(s)):
        for j in range(len(s)):
            hsomomegaparsed[omega][:,i,j] = hsomomega[:,s[i],s[j]]
            ddromegaparsed[omega][:,i,j] = ddromega[:,s[i],s[j]]

    hls = np.zeros((len(r),len(s),len(s)), dtype=complex)
    ddrls = np.zeros((len(r),len(s),len(s)), dtype=complex)

    e, v = np.linalg.eigh(hsomomegaparsed[omega][-1])

    for i in range(len(r)):
        hls[i] = np.dot(np.matrix(v).H, np.dot(hsomomegaparsed[omega][i], v))
        ddrls[i] = np.dot(np.matrix(v).H, np.dot(ddromegaparsed[omega][i], v))

    if omega != 0:
        get_matrix(r, hls, f'{omega}')
        get_nondiag(r, ddrls, f'ddR_{omega}', eps=1e-6, form='%.6f')
    else:
        sp = [j for j in range(len(hls[-1])) if any([np.abs(c) >= 1e-2 for c in hls[:,0,j]])]
        sm = [j for j in range(len(hls[-1])) if any([np.abs(c) >= 1e-2 for c in hls[:,1,j]])]
        hlsplus = np.zeros((len(r),len(sp),len(sp)), dtype=complex)
        ddrlsplus = np.zeros((len(r),len(sp),len(sp)), dtype=complex)
        hlsminus = np.zeros((len(r),len(sm),len(sm)), dtype=complex)
        ddrlsminus = np.zeros((len(r),len(sm),len(sm)), dtype=complex)
        for i in range(len(sp)):
            for j in range(len(sp)):
                hlsplus[:,i,j] = hls[:,sp[i],sp[j]]
                ddrlsplus[:,i,j] = ddrls[:,sp[i],sp[j]]
        for i in range(len(sm)):
            for j in range(len(sm)):
                hlsminus[:,i,j] = hls[:,sm[i],sm[j]]
                ddrlsminus[:,i,j] = ddrls[:,sm[i],sm[j]]
        
        get_matrix(r, hlsplus, '0_plus')
        get_matrix(r, hlsminus, '0_minus')
        get_nondiag(r, ddrlsplus, 'ddR_0_plus', eps=1e-6, form='%.6f')
        get_nondiag(r, ddrlsminus, 'ddR_0_minus', eps=1e-6, form='%.6f')