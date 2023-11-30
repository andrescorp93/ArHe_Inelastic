import numpy as np
import os
import matplotlib.pyplot as plt

multablesym = [[1,2,3,4],[2,1,3,4],[3,4,1,2],[4,3,2,1]]
spinsym = [{'s': 0, 'spinsym': 0}, {'s': 1, 'spinsym': 1}, {'s': 1, 'spinsym': -1}, {'s': 1, 'spinsym': 0}]


def gen_sym_basis(states):
    unsorted_states = [s.copy() for s in states]
    for k in unsorted_states:
        k['spinsym'] = k['sz']
        del k['sz']
    outstates = []
    for totrep in [1,2,3,4]:
        sub = []
        for k in unsorted_states:
            spinrep = spinsym.index({'s': k['s'], 'spinsym': k['spinsym']})
            spatrep = k['spatsym'] - 1
            if multablesym[spatrep][spinrep] == totrep:
                sub.append(k)
        outstates = [*outstates, *sub]
    return unsorted_states


def gen_transform(instates):
    states = [s.copy() for s in instates]
    outstates = gen_sym_basis(states)
    result = np.matrix(np.zeros((len(states), len(states))))
    for i in range(len(states)):
        n = outstates[i]
        # nsym = multablesym[n['spatsym'] - 1][spinsym.index({'s': n['s'], 'spinsym': n['spinsym']})]
        # print(nsym)
        for j in range(len(states)):
            m = states[j]
            if n['n'] == m['n'] and n['spatsym'] == m['spatsym'] and n['s'] == m['s']:
                if n['spinsym'] == 0 and m['sz'] == 0:
                    if n['s'] == 0:
                        result[j,i] = 1
                    elif n['s'] == 1:
                        result[j,i] = 1
                elif n['spinsym'] == 1 and m['sz'] == 1:
                    result[j,i] = 1/np.sqrt(2)
                elif n['spinsym'] == 1 and m['sz'] == -1:
                    result[j,i] = 1/np.sqrt(2)
                elif n['spinsym'] == -1 and m['sz'] == 1:
                    result[j,i] = 1/np.sqrt(2)
                elif n['spinsym'] == -1 and m['sz'] == -1:
                    result[j,i] = -1/np.sqrt(2)
    return result
   

readres = {}

in_states = [{'n': 1, 'spatsym': 1, 's': 1, 'sz': 1}, {'n': 2, 'spatsym': 1, 's': 1, 'sz': 1}, {'n': 3, 'spatsym': 1, 's': 1, 'sz': 1},
 {'n': 4, 'spatsym': 1, 's': 1, 'sz': 1}, {'n': 1, 'spatsym': 1, 's': 1, 'sz': 0}, {'n': 2, 'spatsym': 1, 's': 1, 'sz': 0},
 {'n': 3, 'spatsym': 1, 's': 1, 'sz': 0}, {'n': 4, 'spatsym': 1, 's': 1, 'sz': 0}, {'n': 1, 'spatsym': 1, 's': 1, 'sz':-1},
 {'n': 2, 'spatsym': 1, 's': 1, 'sz':-1}, {'n': 3, 'spatsym': 1, 's': 1, 'sz':-1}, {'n': 4, 'spatsym': 1, 's': 1, 'sz':-1},
 {'n': 1, 'spatsym': 2, 's': 1, 'sz': 1}, {'n': 2, 'spatsym': 2, 's': 1, 'sz': 1}, {'n': 3, 'spatsym': 2, 's': 1, 'sz': 1},
 {'n': 1, 'spatsym': 2, 's': 1, 'sz': 0}, {'n': 2, 'spatsym': 2, 's': 1, 'sz': 0}, {'n': 3, 'spatsym': 2, 's': 1, 'sz': 0},
 {'n': 1, 'spatsym': 2, 's': 1, 'sz':-1}, {'n': 2, 'spatsym': 2, 's': 1, 'sz':-1}, {'n': 3, 'spatsym': 2, 's': 1, 'sz':-1},
 {'n': 1, 'spatsym': 3, 's': 1, 'sz': 1}, {'n': 2, 'spatsym': 3, 's': 1, 'sz': 1}, {'n': 3, 'spatsym': 3, 's': 1, 'sz': 1},
 {'n': 1, 'spatsym': 3, 's': 1, 'sz': 0}, {'n': 2, 'spatsym': 3, 's': 1, 'sz': 0}, {'n': 3, 'spatsym': 3, 's': 1, 'sz': 0},
 {'n': 1, 'spatsym': 3, 's': 1, 'sz':-1}, {'n': 2, 'spatsym': 3, 's': 1, 'sz':-1}, {'n': 3, 'spatsym': 3, 's': 1, 'sz':-1},
 {'n': 1, 'spatsym': 4, 's': 1, 'sz': 1}, {'n': 2, 'spatsym': 4, 's': 1, 'sz': 1}, {'n': 1, 'spatsym': 4, 's': 1, 'sz': 0},
 {'n': 2, 'spatsym': 4, 's': 1, 'sz': 0}, {'n': 1, 'spatsym': 4, 's': 1, 'sz':-1}, {'n': 2, 'spatsym': 4, 's': 1, 'sz':-1},
 {'n': 1, 'spatsym': 1, 's': 0, 'sz': 0}, {'n': 2, 'spatsym': 1, 's': 0, 'sz': 0}, {'n': 3, 'spatsym': 1, 's': 0, 'sz': 0},
 {'n': 4, 'spatsym': 1, 's': 0, 'sz': 0}, {'n': 5, 'spatsym': 1, 's': 0, 'sz': 0}, {'n': 1, 'spatsym': 2, 's': 0, 'sz': 0},
 {'n': 2, 'spatsym': 2, 's': 0, 'sz': 0}, {'n': 3, 'spatsym': 2, 's': 0, 'sz': 0}, {'n': 1, 'spatsym': 3, 's': 0, 'sz': 0},
 {'n': 2, 'spatsym': 3, 's': 0, 'sz': 0}, {'n': 3, 'spatsym': 3, 's': 0, 'sz': 0}, {'n': 1, 'spatsym': 4, 's': 0, 'sz': 0},
 {'n': 2, 'spatsym': 4, 's': 0, 'sz': 0}]

new_states = gen_sym_basis(in_states)
# print(new_states)
ot = gen_transform(in_states)
np.savetxt('trans.txt', ot, fmt='%.5f')
# print(np.dot(ot.T, ot))
# print(np.allclose(np.dot(ot.T, ot), np.eye(len(ot))))

for f in os.listdir('repaired_HLS_v2'):
    filename = 'repaired_HLS_v2/' + f
    headers = open(filename).readlines()[0].split()
    if f.find('Im')!=-1:
        data = 1j * np.loadtxt(filename, skiprows=1).T
    else:
        data = np.loadtxt(filename, skiprows=1).T
    for i in range(len(headers)):
        readres[headers[i]] = data[i]

r = readres['r']
hsom = np.zeros((len(r),49,49), dtype=complex)
for k in readres.keys():
    if k != 'r':
        ids = [int(s) for s in k[1:-1].split('|')[::2]]
        if ids[0] == ids[1]:
            hsom[:, ids[0], ids[1]] = readres[k]
        else:
            hsom[:, ids[0], ids[1]] = readres[k]
            hsom[:, ids[1], ids[0]] = readres[k].conjugate()
nonzerosin = 0
for i in range(49):
    for j in range(i+1, 49):
        if all([np.abs(np.real(c)) >= 1e-2 for c in hsom[:,i,j]]) or all([np.abs(np.imag(c)) >= 1e-2 for c in hsom[:,i,j]]):
            nonzerosin += 1


hso_sym = np.zeros((len(r), 49, 49), dtype=complex)

for i in range(len(hsom)):
    hso_sym[i] = np.array(ot.T.dot(np.matrix(hsom[i])).dot(ot))

# np.savetxt('hso_large.txt', hso_sym[-1], fmt='%.3f')

psis = np.zeros((len(r), 49, 49), dtype=complex)
es = np.zeros((len(r), 49))

for i in range(len(hso_sym)):
    es[i], psis[i] = np.linalg.eigh(hso_sym[i])

# nonzerosout = 0
# for i in range(49):
#     ist = new_states[i]
#     istsym = multablesym[ist['spatsym'] - 1][spinsym.index({'s': ist['s'], 'spinsym': ist['spinsym']})]
#     for j in range(i+1, 49):
#         jst = new_states[j]
#         jstsym = multablesym[jst['spatsym'] - 1][spinsym.index({'s': jst['s'], 'spinsym': jst['spinsym']})]
#         if (all([np.abs(np.real(c)) >= 1e-3 for c in hso_sym[:,i,j]]) or all([np.abs(np.imag(c)) >= 1e-3 for c in hso_sym[:,i,j]])):
#             nonzerosout += 1
#             print(f'<{i}|Hso|{j}>!=0; repconv={istsym==jstsym}')
# print(nonzerosin)
# print(nonzerosout)
            # print(f'<{i}|Hso|{j}>!=0; repconv={istsym==jstsym}')
        #     plt.plot(r, np.real(hso_sym[:,i,j]), label=f'Re<{i}|Hso|{j}>')
        # elif all([np.abs(np.imag(c)) >= 1e-3 for c in hso_sym[:,i,j]]):
        #     plt.plot(r, np.imag(hso_sym[:,i,j]), label=f'Im<{i}|Hso|{j}>')
            
# plt.legend()
# plt.show()

for i in range(49):
    plt.plot(r, np.abs(es[:,i]))
plt.show()
