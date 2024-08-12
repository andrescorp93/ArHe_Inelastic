import os
import numpy as np
import matplotlib.pyplot as plt
from funcs import *

dirs = [n for n in os.listdir() if (os.path.isdir(n) and (n[-2:]=='_s' or n[-2:]=='_p'))]

weights_states = {'1s5': 5, '1s4': 3, '1s3': 1, '1s2': 3,
                  '2p10': 3, '2p9': 7, '2p8': 5, '2p7': 3,
                  '2p6': 5, '2p5': 1, '2p4': 3, '2p3': 5,
                  '2p2': 3, '2p1': 1}

weights_omegas = {'0-': 1, '0+': 1, '1': 2, '2': 2}

labels = {'s': {'0-': ['1s5', '1s3'], '0+': ['1s4', '1s2'], '1': ['1s5', '1s4', '1s2']},
          'p': {'0-': ['2p10', '2p9', '2p7', '2p4', '2p2'],
                '0+': ['2p8', '2p6', '2p5', '2p3', '2p1'],
                '1': ['2p10', '2p9', '2p8', '2p7', '2p6', '2p4', '2p3', '2p2'],
                '2': ['2p9', '2p8', '2p6', '2p3']}}

constants = {}

for d in dirs:
    group = d[-1]
    omega = d[2:-2] if d[2] != '0' else ('0+' if d[2:5]=='0_p' else '0-')
    states = labels[group][omega]
    data = np.loadtxt(f'{d}/rate_const_new.txt', skiprows=1).transpose()
    if 'T' not in constants.keys():
        constants['T'] = data[0]
    n = len(states)
    col = 1
    for i in range(n):
        for j in range(n):
            if i != j:
                proc_name = states[i] + '->' + states[j]
                pij = weights_omegas[omega] / weights_states[states[i]]
                if proc_name not in constants.keys():
                    constants[proc_name] = pij * data[col]
                else:
                    constants[proc_name] += pij * data[col]
                col +=1

constants_p = {'T': constants['T']}
constants_s = {'T': constants['T']}

for k in constants.keys():
    if k != 'T':
        if k.find('s') != -1:
            constants_s[k] = constants[k]
        if k.find('p') != -1:
            constants_p[k] = constants[k]
            # print(k[:k.find('->')], k[k.find('->')+2:])

s_labels = ['1s5', '1s4', '1s3', '1s2']
header_s = 'T, K\t'
out_data_s = np.zeros((len(constants_s['T']), len(constants_s.keys())))
out_data_s[:,0] = constants_s['T']
col = 1
for i in range(len(s_labels)):
    for j in range(len(s_labels)):
        if i != j:
            k = f'{s_labels[i]}->{s_labels[j]}'
            if k in constants_s.keys():
                header_s += f'k({k}), cm3/s\t'
                out_data_s[:,col] = constants_s[k]
                col += 1

np.savetxt(f'rate_const_s.txt', out_data_s, fmt='%.6e', delimiter='\t', header=header_s, comments='')

p_labels = ['2p10', '2p9', '2p8', '2p7', '2p6', '2p5', '2p4', '2p3', '2p2', '2p1']
header_p = 'T, K\t'
out_data_p = np.zeros((len(constants_s['T']), len(constants_p.keys())))
out_data_p[:,0] = constants_p['T']
col = 1
for i in range(len(p_labels)):
    for j in range(len(p_labels)):
        if i != j:
            k = f'{p_labels[i]}->{p_labels[j]}'
            if k in constants_p.keys():
                header_p += f'k({k}), cm3/s\t'
                out_data_p[:,col] = constants_p[k]
                col += 1

np.savetxt(f'rate_const_p.txt', out_data_p, fmt='%.6e', delimiter='\t', header=header_p, comments='')

plt.style.use('seaborn')
for k in constants_s.keys():
    if k != 'T':
        plt.scatter(1000/constants_s['T'], constants_s[k], label=k)
plt.semilogy()
plt.legend()
plt.show()
