import numpy as np
from funcs import *

dirs = [n for n in os.listdir() if (os.path.isdir(n) and (n[-2:]=='_s' or n[-2:]=='_p'))]

nist_levels = {'1s5': 93143.7600,
               '1s4': 93750.5978,
               '1s3': 94553.6652,
               '1s2': 95399.8276,
               '2p10': 104102.0990,
               '2p9': 105462.7596,
               '2p8': 105617.2700,
               '2p7': 106087.2598,
               '2p6': 106237.5518, 
               '2p5': 107054.2720, 
               '2p4': 107131.7086, 
               '2p3': 107289.7001,
               '2p2': 107496.4166,
               '2p1': 108722.6194}

elabelarr = {}
for dir in dirs:
    omega = dir[2:-2] if dir[2] != '0' else ('0+' if dir[2:5]=='0_p' else '0-')
    r, hls, ddrls = load_matrices(dir)
    n = len(hls[0])
    group = dir[-1]
    
    for i in range(n):
        if labels[group][omega][i] not in elabelarr.keys():
            if int(omega[0]) > 0:
                elabelarr[labels[group][omega][i]] = [np.real(hls[-1,i,i]), np.real(hls[-1,i,i])]
            else:
                elabelarr[labels[group][omega][i]] = [np.real(hls[-1,i,i])]
        else:
            if int(omega[0]) > 0:
                elabelarr[labels[group][omega][i]].extend([np.real(hls[-1,i,i]), np.real(hls[-1,i,i])])
            else:
                elabelarr[labels[group][omega][i]].append(np.real(hls[-1,i,i]))
calcs = {}
errors = []
for k, v in elabelarr.items():
    calcs[k] = sum(v)/len(v)
    errors.append(np.abs((sum(v)/len(v))-nist_levels[k]))
print(sum(errors)/len(errors))
print(calcs)
