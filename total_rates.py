import os
import numpy as np
import matplotlib.pyplot as plt
from funcs import *
import cycler

dirs = [n for n in os.listdir() if (os.path.isdir(n) and ((n[-2:]=='_s' and int(n[2]) < 2) or (n[-2:]=='_p' and int(n[2]) < 3)))]

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

colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
          '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 15
plt.rcParams['image.cmap'] = 'Paired'
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', colors)
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['savefig.format'] = 'eps'
plt.rcParams['lines.linewidth'] = 3

colorid = 1
for k in constants_s.keys():
    if k != 'T':
        states = k.split('-')
        states[1] = states[1][1:]
        if (int(states[0][2:]) > 3) and (int(states[1][2:]) > 3) and (int(states[1][2:]) > int(states[0][2:])):
            krev = f'{states[1]}->{states[0]}'
            sign = f'${{{states[1][:2]}_{{{states[1][2:]}}} \\rightarrow {states[0][:2]}_{{{states[0][2:]}}}}}$'
            signrev = f'${{{states[0][:2]}_{{{states[0][2:]}}} \\rightarrow {states[1][:2]}_{{{states[1][2:]}}}}}$'
            plt.plot(constants_s['T'], constants_s[krev], label=sign, color=colors[colorid], linestyle='-')
            plt.plot(constants_s['T'], constants_s[k], label=signrev, color=colors[colorid], linestyle='--')
            colorid += 2
plt.semilogy()
plt.legend()
plt.xlabel('T, K')
plt.ylabel('k, cm${}^{3}$/s')
plt.savefig(f'images/Rate_const_s.eps')
plt.close()

colorid = 1

for k in constants_p.keys():
    if k != 'T':
        states = k.split('-')
        states[1] = states[1][1:]
        if (int(states[0][2:]) > 7) and (int(states[1][2:]) > 7) and (int(states[1][2:]) > int(states[0][2:])):
            krev = f'{states[1]}->{states[0]}'
            sign = f'${{{states[1][:2]}_{{{states[1][2:]}}} \\rightarrow {states[0][:2]}_{{{states[0][2:]}}}}}$'
            signrev = f'${{{states[0][:2]}_{{{states[0][2:]}}} \\rightarrow {states[1][:2]}_{{{states[1][2:]}}}}}$'
            plt.plot(constants_s['T'], constants_p[krev], label=sign, color=colors[colorid], linestyle='-')
            plt.plot(constants_s['T'], constants_p[k], label=signrev, color=colors[colorid], linestyle='--')
            colorid += 2
plt.semilogy()
plt.legend()
plt.xlabel('T, K')
plt.ylabel('k, cm${}^{3}$/s')
plt.savefig(f'images/Rate_const_p.eps')
plt.close()
