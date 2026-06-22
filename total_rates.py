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

exp_temp = np.array([375, 415, 470, 525, 600])

# experiments = {'Han':{'2p8->2p9':{'T': 298, 'k': 4.5e-11, 'errork': 0.3*1.1e-11, 'marker': 'v'},
#                          '2p8->2p10':{'T': 298, 'k': 4.0e-12, 'errork': 0.3*4.0e-12, 'marker': 'v'},
#                          '2p9->2p8':{'T': 298, 'k': 1.5e-11, 'errork': 0.3*1.5e-11, 'marker': '^'},
#                          '2p9->2p10':{'T': 298, 'k': 1.6e-11, 'errork': 0.3*1.6e-11, 'marker': 'v'}},
#                'Kuramshin':{'2p8->2p9':{'T': exp_temp, 'k': 6.9e-11*np.ones(len(exp_temp)), 'errork': 1.9e-11*np.ones(len(exp_temp)), 'marker': 's'},
#                             '2p9->2p10':{'T': exp_temp, 'k': 7e-12*np.ones(len(exp_temp)), 'errork': 2e-12*np.ones(len(exp_temp)), 'errort': 10, 'marker': 's'},
#                             '2p8->2p10':{'T': exp_temp, 'k': 1.1e-11*np.ones(len(exp_temp)), 'errork': 0.3e-11*np.ones(len(exp_temp)), 'errort': 10, 'marker': 's'}},
#                'Ivanov':{'1s5->1s4':{'T': 300, 'k': 2.1e-15, 'errork': 2e-16, 'marker': 'o'}}}

experiments = {'Kuramshin':{'2p8->2p9':{'T': exp_temp, 'k': np.array([4.83e-11, 3.98e-11, 6.84e-11, 3.99e-11, 5.03e-11]), 'errork': 1.9e-11*np.ones(len(exp_temp)), 'marker': 's'},
                            '2p9->2p10':{'T': exp_temp, 'k': np.array([6.25e-12, 4.50e-12, 7.68e-12, 5.45e-12, 7.32e-12]), 'errork': 2e-12*np.ones(len(exp_temp)), 'errort': 10, 'marker': 's'},
                            '2p8->2p10':{'T': exp_temp, 'k': np.array([1.02e-11, 8.33e-12, 8.91e-12, 1.65e-11, 1.37e-11]), 'errork': 0.3e-11*np.ones(len(exp_temp)), 'errort': 10, 'marker': 's'}}}

constants = {}

for d in dirs:
    group = d[-1]
    omega = d[2:-2] if d[2] != '0' else ('0+' if d[2:5]=='0_p' else '0-')
    states = labels[group][omega]
    data = np.loadtxt(f'{d}/rate_const_airy_redetailed.txt', skiprows=1).transpose()
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

np.savetxt(f'rate_const_s_airy_redetailed.txt', out_data_s, fmt='%.6e', delimiter='\t', header=header_s, comments='')

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

np.savetxt(f'rate_const_p_airy_redetailed.txt', out_data_p, fmt='%.6e', delimiter='\t', header=header_p, comments='')

out_data_p_edwba = np.loadtxt(f'rate_const_p_new.txt', skiprows=1)
p_airy_header = open('rate_const_p_airy_redetailed.txt').readlines()[0].split('\t')[:-1]
constants_p_edwba = {'T': out_data_p_edwba[:,0]}

for i in range(len(p_airy_header)):
    if p_airy_header[i].find('T') == -1:
        constants_p_edwba[p_airy_header[i][2:-8]] = out_data_p_edwba[:,i]


colors = ['#a6cee3', '#1f78b4', '#fb9a99', '#e31a1c', '#b2df8a', '#33a02c',
          '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 15
plt.rcParams['image.cmap'] = 'Paired'
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', colors)
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['lines.linewidth'] = 3

colorid = 1
# for k in constants_s.keys():
#     if k != 'T':
#         states = k.split('-')
#         states[1] = states[1][1:]
#         if (int(states[0][2:]) > 3) and (int(states[1][2:]) > 3) and (int(states[1][2:]) > int(states[0][2:])):
#             krev = f'{states[1]}->{states[0]}'
#             sign = f'${{{states[1][:2]}_{{{states[1][2:]}}} \\rightarrow {states[0][:2]}_{{{states[0][2:]}}}}}$'
#             signrev = f'${{{states[0][:2]}_{{{states[0][2:]}}} \\rightarrow {states[1][:2]}_{{{states[1][2:]}}}}}$'
#             for author in experiments.keys():
#                 for trans in experiments[author]:
#                     if trans == k:
#                         print(trans)
#                         res = experiments[author][k]
#                         plt.scatter(res['T'], res['k'], label=f'{author}, {sign}', color=colors[colorid], marker=res['marker'])
#                     elif trans == krev:
#                         res = experiments[author][krev]
#                         plt.scatter(res['T'], res['k'], label=f'{author}, {sign}', color=colors[colorid], marker=res['marker'])
#             plt.plot(constants_s['T'], constants_s[krev], label=sign, color=colors[colorid], linestyle='-')
#             plt.plot(constants_s['T'], constants_s[k], label=signrev, color=colors[colorid], linestyle='--')
#             colorid += 2
# plt.semilogy()
# plt.legend()
# plt.xlabel('T, K')
# plt.ylabel('k, cm${}^{3}$/s')
# plt.show()
# plt.savefig(f'images/Figure_7_rev.pdf')
# plt.close()

plt.rcParams['figure.figsize'] = [11, 7]
colorid = 1

for k in constants_p.keys():
    # if (k.find('2p8->2p9') == -1 and k.find('2p9->2p8') == -1) and k != 'T':
    if k != 'T':
        print(k)
        states = k.split('-')
        states[1] = states[1][1:]
        if (int(states[0][2:]) > 7) and (int(states[1][2:]) > 7) and (int(states[1][2:]) > int(states[0][2:])):
            krev = f'{states[1]}->{states[0]}'
            sign = f'${{{states[1][:2]}_{{{states[1][2:]}}} \\rightarrow {states[0][:2]}_{{{states[0][2:]}}}}}$'
            signrev = f'${{{states[0][:2]}_{{{states[0][2:]}}} \\rightarrow {states[1][:2]}_{{{states[1][2:]}}}}}$'
            for author in experiments.keys():
                for trans in experiments[author]:
                    # if trans == k and (trans != '2p8->2p9' or trans != '2p9->2p8'):
                    if trans == k:
                        res = experiments[author][k]
                        states_exp = k.split('-')
                        states_exp[1] = states_exp[1][1:]
                        sign_exp = f'${{{states_exp[0][:2]}_{{{states_exp[0][2:]}}} \\rightarrow {states_exp[1][:2]}_{{{states_exp[1][2:]}}}}}$'
                        plt.errorbar(res['T'], res['k'], yerr = res['errork'], fmt='.', label=f'{author}, {sign_exp}', color=colors[colorid], marker=res['marker'])
                    # elif trans == krev  and (trans != '2p8->2p9' or trans != '2p9->2p8'):
                    elif trans == krev:
                        res = experiments[author][krev]
                        states_exp = krev.split('-')
                        states_exp[1] = states_exp[1][1:]
                        sign_exp = f'${{{states_exp[0][:2]}_{{{states_exp[0][2:]}}} \\rightarrow {states_exp[1][:2]}_{{{states_exp[1][2:]}}}}}$'
                        plt.errorbar(res['T'], res['k'], yerr = res['errork'], fmt='.', label=f'{author}, {sign_exp}', color=colors[colorid], marker=res['marker'])
            # plt.plot(constants_s['T'], constants_p[krev], label=sign, color=colors[colorid], linestyle='-')
            plt.plot(constants_s['T'], constants_p[k], label=f'{signrev}, CC approach', color=colors[colorid], linestyle='-')
            # plt.plot(constants_s['T'], constants_p_edwba[k], label=f'{signrev}, EDWBA', color=colors[colorid], linestyle='--')
            colorid += 2
plt.semilogy()
plt.legend(ncol=3, columnspacing=0.6)
# plt.legend()
plt.xlabel('T, K')
plt.ylabel('k, cm${}^{3}$/s')
plt.ylim(1e-12, 1e-10)
plt.show()
# plt.savefig(f'images/Figure_7_Airy.png')
# plt.close()
