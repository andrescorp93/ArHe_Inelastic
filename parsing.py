import os
import re
import numpy as np
import matplotlib.pyplot as plt


def transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


so_outs = [filename for filename in os.listdir() if filename.find('so')!=-1]
nacme_outs = [filename for filename in os.listdir() if filename.find('nacme')!=-1]

rs = []
els = []
psils = []
repsils = []
impsils = []
dmxs = []
dmys = []
dmzs = []
redmx = []
redmy = []
redmz = []
imdmx = []
imdmy = []
imdmz = []

in_states_a1 = ['1.2.1', '2.2.1', '3.2.1', '1.3.1', '2.3.1', '3.3.1', '1.4.1',
                '2.4.1', '1.1.0', '2.1.0', '3.1.0', '4.1.0', '5.1.0']
in_states_b1 = ['1.1.1', '2.1.1', '3.1.1', '4.1.1', '1.3.1', '2.3.1',
                '3.3.1', '1.4.1', '2.4.1', '1.2.0', '2.2.0', '3.2.0']
in_states_b2 = ['1.1.1', '2.1.1', '3.1.1', '4.1.1', '1.2.1', '2.2.1',
                '3.2.1', '1.4.1', '2.4.1', '1.3.0', '2.3.0', '3.3.0']
in_states_a2 = ['1.1.1', '2.1.1', '3.1.1', '4.1.1', '1.2.1', '2.2.1',
                '3.2.1', '1.3.1', '2.3.1', '3.3.1', '1.4.0', '2.4.0']
ls_states = [str(i+1) for i in range(49)]

for f in so_outs:
    txt = open(f).readlines()
    for i in range(len(txt)):
        if txt[i].find('r=[')!=-1:
            rs = [*rs, *txt[i][4:-2].split(',')]
        if txt[i].find('Summary of SO results')!=-1:
            els.append([float(s.split()[2]) for s in txt[i+8:i+60] if len(s.split())>0])
        if txt[i].find('Spin-orbit eigenvectors   (columnwise ')!=-1:
            result = []
            txtm = [s for s in txt[i+5:i+1060] if len(s.split())>0]
            for j in range(len(txtm)):
                if txtm[j].find('Total')!=-1:
                    part = txtm[j+2:j+100]
                    if len(result) == 0:
                        result = [s[30:-1] for s in part]
                    else:
                        for k in range(len(part)):
                            result[k] += part[k][30:-1]
            psils.append(result)
        if txt[i].find('Property matrix of the DMX operator in a.u.')!=-1:
            result = []
            txtm = [s for s in txt[i:i+1052] if len(s.split())>0]
            for j in range(len(txtm)):
                if txtm[j].find('Total')!=-1:
                    part = txtm[j+2:j+100]
                    if len(result) == 0:
                        result = [s[15:-1] for s in part]
                    else:
                        for k in range(len(result)):
                            result[k] += ' ' + part[k][15:-1]
            dmxs.append(result)
        if txt[i].find('Property matrix of the DMY operator in a.u.')!=-1:
            result = []
            txtm = [s for s in txt[i:i+1052] if len(s.split())>0]
            for j in range(len(txtm)):
                if txtm[j].find('Total')!=-1:
                    part = txtm[j+2:j+100]
                    if len(result) == 0:
                        result = [s[15:-1] for s in part]
                    else:
                        for k in range(len(result)):
                            result[k] += ' ' + part[k][15:-1]
            dmys.append(result)
        if txt[i].find('Property matrix of the DMZ operator in a.u.')!=-1:
            result = []
            txtm = [s for s in txt[i:i+1052] if len(s.split())>0]
            for j in range(len(txtm)):
                if txtm[j].find('Total')!=-1:
                    part = txtm[j+2:j+100]
                    if len(result) == 0:
                        result = [s[15:-1] for s in part]
                    else:
                        for k in range(len(result)):
                            result[k] += ' ' + part[k][15:-1]
            dmzs.append(result)

r = np.array([float(s) for s in rs])
els = np.array(els)

for i in range(len(r)):
    re_psi = []
    im_psi = []
    dmx_re_psi = []
    dmx_im_psi = []
    dmy_re_psi = []
    dmy_im_psi = []
    dmz_re_psi = []
    dmz_im_psi = []
    for j in range(0, len(psils[i]), 2):
        re_psi.append([float(s) for s in psils[i][j].split()])
        im_psi.append([float(s) for s in psils[i][j+1].split()])
        dmx_re_psi.append([float(s) for s in dmxs[i][j].split()])
        dmx_im_psi.append([float(s) for s in dmxs[i][j+1].split()])
        dmy_re_psi.append([float(s) for s in dmys[i][j].split()])
        dmy_im_psi.append([float(s) for s in dmys[i][j+1].split()])
        dmz_re_psi.append([float(s) for s in dmzs[i][j].split()])
        dmz_im_psi.append([float(s) for s in dmzs[i][j+1].split()])
    repsils.append(np.array(re_psi))
    impsils.append(np.array(im_psi))
    redmx.append(np.array(dmx_re_psi))
    imdmx.append(np.array(dmx_im_psi))
    redmy.append(np.array(dmy_re_psi))
    imdmy.append(np.array(dmy_im_psi))
    redmz.append(np.array(dmz_re_psi))
    imdmz.append(np.array(dmz_im_psi))

repsils = np.array(repsils)
impsils = np.array(impsils)
redmx = np.array(redmx)
imdmx = np.array(imdmx)
redmy = np.array(redmy)
imdmy = np.array(imdmy)
redmz = np.array(redmz)
imdmz = np.array(imdmz)
psi = repsils+1j*impsils
dmx = redmx + 1j*imdmx
dmy = redmy + 1j*imdmy
dmz = redmz + 1j*imdmz

for i in range(49):
    for j in range(49):
        psi[:,i,j] = np.exp(1j*np.abs(np.angle(psi[-1,i,j])))*np.abs(psi[:,i,j])

nacmes = {}

for f in nacme_outs:
    txt = open(f).readlines()
    t = f[11:-4].split('_')
    tail = t[0] + '.' + ('0' if t[1] == 'sing' else '1')
    head = ['|'.join([i + '.' + tail for i in s[-2:]]) for s in txt[2].split()[1:]]
    a = []
    for s in txt[3:]:
        a.append([float(n) for n in s.split()[1:]])
    a = np.array(a).T
    for i in range(len(head)):
        nacmes[head[i]] = a[i]

psi_a1 = psi[:,0:13,0:13]
psi_b1 = psi[:,13:25,13:25]
psi_b2 = psi[:,25:37,25:37]
psi_a2 = psi[:,37:49,37:49]

# for i in range(13):
#     for j in range(13):
#         if all([np.abs(np.real(c)) >= 1e-8 for c in psi_a1[:,i,j]]) or all([np.abs(np.imag(c)) >= 1e-8 for c in psi_a1[:,i,j]]):
#             plt.plot(r,np.abs(psi_a1[:,i,j]), label=f'|<{j+1}|{in_states_a1[i]}>|')
#     plt.legend()
#     plt.show()

# for i in range(13):
#     for j in range(13):
#         if all([np.abs(np.imag(c)) >= 1e-8 for c in psi_a1[:,i,j]]):
#             plt.plot(r,np.abs(np.imag(psi_a1[:,i,j])), label=f'Im<{j+1}|{in_states_a1[i]}>')

els = (els - np.min(els[:,0]))*219474.63
for i in range(len(els[0])):
    fname = f'e{i+1}.txt'
    header = f'r\te{i+1}'
    res = np.zeros((len(r),2))
    res[:,0] = r
    res[:,1] = els[:,i]
    np.savetxt(fname, res, header=header)

