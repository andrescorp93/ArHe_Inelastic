import numpy as np
import os
from scipy.special import spherical_jn as sph_jn, spherical_yn as sph_yn
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

state_keys = ['n', 'lam', 's', 'sz']

labels = {'s': {'0-': ['1s5', '1s3'],
                '0+': ['1s4', '1s2'],
                '1': ['1s5', '1s4', '1s2'],
                '2': ['1s5']},
          'p': {'0-': ['2p10', '2p9', '2p7', '2p4', '2p2'],
                '0+': ['2p8', '2p6', '2p5', '2p3', '2p1'],
                '1': ['2p10', '2p9', '2p8', '2p7', '2p6', '2p4', '2p3', '2p2'],
                '2': ['2p9', '2p8', '2p6', '2p3'],
                '3': ['2p9']}}

weights_states = {'1s5': 5, '1s4': 3, '1s3': 1, '1s2': 3,
                  '2p10': 3, '2p9': 7, '2p8': 5, '2p7': 3,
                  '2p6': 5, '2p5': 1, '2p4': 3, '2p3': 5,
                  '2p2': 3, '2p1': 1}

nist_energies = {'1s5':  93143.7600, '1s4':  93750.5978, '1s3':  94553.6652, '1s2':  95399.8276,
                  '2p10':  104102.0990, '2p9':  105462.7596, '2p8':  105617.2700, '2p7':  106087.2598,
                  '2p6':  106237.5518, '2p5':  107054.2720, '2p4':  107131.7086, '2p3':  107289.7001,
                  '2p2':  107496.4166, '2p1':  108722.6194}

weights_omegas = {'0-': 1, '0+': 1, '1': 2, '2': 2}

colors = {'1s5': '#a6cee3',
          '1s4': '#1f78b4',
          '1s3': '#b2df8a',
          '1s2': '#33a02c'}

colorp = {'2p10': '#a6cee3',
          '2p9': '#1f78b4',
          '2p8': '#b2df8a',
          '2p7': '#33a02c',
          '2p6': '#fb9a99',
          '2p5': '#e31a1c',
          '2p4': '#fdbf6f',
          '2p3': '#ff7f00',
          '2p2': '#cab2d6',
          '2p1': '#6a3d9a'}

lines = {'0+':'-',
         '0-':'--',
         '1':'-.',
         '2':':',
         '3':(5, (10, 3)),
         'nist': (0, (1, 1))}

spatchom = [mlines.Line2D([], [], color='black', linestyle=lines[k], label=f'$\Omega = {k}$') for k in lines.keys() if k != 'nist' and int(k[0]) < 3]
spatchom.append(mlines.Line2D([], [], color='black', linestyle=lines['nist'], label='NIST'))
spatchst = [mlines.Line2D([], [], color=colors[k], linestyle='-', label=f'${k[:2]}_{{{k[2:]}}}$') for k in colors.keys() if k != 'nist' and int(k[2:]) > 3]
ppatchom = [mlines.Line2D([], [], color='black', linestyle=lines[k], label=f'$\Omega = {k}$') for k in lines.keys() if k != 'nist']
ppatchom.append(mlines.Line2D([], [], color='black', linestyle=lines['nist'], label='NIST'))
ppatchst = [mlines.Line2D([], [], color=colorp[k], linestyle='-', label=f'${k[:2]}_{{{k[2:]}}}$') for k in colorp.keys() if k != 'nist' and int(k[2:]) > 7]


def model_potential(u0, eps, x):
    return (u0 + eps * np.power(x, -6)) / 4.637  # hbar^2/(2*mu*(1A^2)) = 4.637 cm-1


def morse_non_diag(x, d, alpha, x0):
    return d * (np.exp(-2 * alpha * (x - x0)) - 2 * np.exp(-alpha * (x - x0))) / 0.529 # bohr to angstrom


def morse_non_diag_diff(x, d, alpha, x0):
    return 2 * alpha * d * (-np.exp(-2 * alpha * (x - x0)) + np.exp(-alpha * (x - x0))) / 0.529 # bohr to angstrom


def k2(x, u, e, j):
    return e - u(x) - j * (j + 1) / np.power(x, 2)


def k_inf(u0, e):
    if e > u0:
        return np.sqrt(e - u0)
    else: 
        return 0


def model_force(eps, j, x):
    return ((-6 * eps * np.power(x, -7)) / 4.637) - 2 * j * (j + 1) * np.power(x, -3)  # hbar^2/(2*mu*(1A^2)) = 4.637 cm-1


def initial_value_wkb(x, u1, u2, f1, f2, e, j):
    k1p = f1(x) / (2. * np.sqrt(np.abs(k2(x, u1, e, j))))
    k2p = f2(x) / (2. * np.sqrt(np.abs(k2(x, u2, e, j))))
    k10 = np.sqrt(np.abs(k2(x, u1, e, j)))
    k20 = np.sqrt(np.abs(k2(x, u2, e, j)))
    psi1p = k10 - k1p / (2. * k10)
    psi2p = k20 - k2p / (2. * k20)
    return np.array([0., psi1p, 0., psi2p])
    

def initial_value_j(x, j):
    return np.array([1., j/x])
    

def ccequations(y, e, j, u01, u02, eps1, eps2, d, alpha, x0, x):
    f = np.zeros(len(y))
    f[0] = y[1]
    f[2] = y[3]
    i = morse_non_diag(x, d, alpha, x0)
    ip = morse_non_diag_diff(x, d, alpha, x0)
    f[1] = -k2(x, u01, eps1, e, j) * y[0] + ip * y[0] + 2. * i * y[1]
    f[3] = -k2(x, u02, eps2, e, j) * y[2] - ip * y[2] - 2. * i * y[3]
    return f


def elequations(y, e, j, u, x):
    f = np.zeros(len(y))
    f[0] = y[1]
    f[1] = -(e - u(x) - j * (j + 1) / np.power(x, 2)) * y[0]
    return f


def phase_calc(psi, dpsi, r, u, e, j):
    rm = np.max(r)
    k = np.sqrt(np.abs(k2(rm, u, e, j)))
    psi_rm = psi[np.argmax(r)]
    psi_p = dpsi[np.argmax(r)]
    A = (k * np.sin(k * rm) * psi_rm + np.cos(k * rm) * psi_p) / k
    B = (k * np.cos(k * rm) * psi_rm - np.sin(k * rm) * psi_p) / k
    return np.arctan(B / A)


def mul_initial_value_j(x, es, js):
    ejpairs = np.transpose([np.tile(js, len(es)), np.repeat(es, len(js))])
    f = np.ones(2*len(ejpairs))*x
    f[1::2] = ejpairs[:,0]
    return np.array(f, dtype=complex)

    
def mulelequations(y, es, js, u, x):
    ejpairs = np.transpose([np.tile(js, len(es)), np.repeat(es, len(js))])
    f = np.zeros(len(y), dtype=complex)
    f[::2] = y[1::2]
    k2 = (ejpairs[:,1] - u(x) - ejpairs[:,0] * (ejpairs[:,0] + 1) / np.power(x, 2))
    f[1::2] = -k2 * y[::2]
    return f


def mul_sigma_calc(psi, dpsi, r, es, js):
    ejpairs = np.transpose([np.tile(js, len(es)), np.repeat(es, len(js))])
    rm = np.max(r)
    k = np.sqrt(ejpairs[:,1])
    psi_rm = psi[:, np.argmax(r)]
    psi_p = dpsi[:, np.argmax(r)]
    jl = np.array([sph_jn(int(j), np.sqrt(ks)*rm) for j,ks in ejpairs])
    yl = np.array([sph_yn(int(j), np.sqrt(ks)*rm) for j,ks in ejpairs])
    djl = np.array([sph_jn(int(j), np.sqrt(ks)*rm, derivative=True) for j,ks in ejpairs])
    dyl = np.array([sph_yn(int(j), np.sqrt(ks)*rm, derivative=True) for j,ks in ejpairs])
    g = psi_p/psi_rm
    B = (g * rm - 1) * jl - k * rm * djl
    A = (g * rm - 1) * yl - k * rm * dyl
    part_sigma = np.abs((2 * ejpairs[:,0] + 1) * (B**2 / (A**2 + B**2)) / ejpairs[:,1])
    result = np.zeros(len(es))
    for i in range(len(ejpairs)):
        for j in range(len(es)):
            if ejpairs[i,1] == es[j]:
                result[j] += part_sigma[i]
    return result


def s_phase_calc(psi, dpsi, r, es, js):
    ejpairs = np.transpose([np.tile(js, len(es)), np.repeat(es, len(js))])
    rm = np.max(r)
    k = np.sqrt(ejpairs[:,1])
    psi_rm = psi[:, np.argmax(r)]
    psi_p = dpsi[:, np.argmax(r)]
    jl = np.array([sph_jn(int(j), np.sqrt(ks)*rm) for j,ks in ejpairs])
    yl = np.array([sph_yn(int(j), np.sqrt(ks)*rm) for j,ks in ejpairs])
    djl = np.array([sph_jn(int(j), np.sqrt(ks)*rm, derivative=True) for j,ks in ejpairs])
    dyl = np.array([sph_yn(int(j), np.sqrt(ks)*rm, derivative=True) for j,ks in ejpairs])
    g = psi_p/psi_rm
    B = (g * rm - 1) * jl - k * rm * djl
    A = (g * rm - 1) * yl - k * rm * dyl
    return np.arctan(B/A)

        
def norm_sol(r, es, js, u):
    ejpairs = np.transpose([np.tile(js, len(es)), np.repeat(es, len(js))])
    h = np.diff(r)[-1]
    k = np.sqrt(ejpairs[:,1])
    sol_elastic = solve_ivp(lambda t, y: mulelequations(y, es, js, u, t), (np.min(r), np.max(r)),
                                            mul_initial_value_j(np.min(r), es, js), t_eval=r)
    psi = sol_elastic.y[::2]
    dpsi = sol_elastic.y[1::2]
    m = int(np.round(2*np.pi/(h*k[0])))
    cl = np.array([np.max(psi[i,-m:]) for i in range(len(psi))])
    u = np.array([psi[i]/(np.sqrt(k[i])*cl[i]) for i in range(len(cl))])
    du = np.array([dpsi[i]/(np.sqrt(k[i])*cl[i]) for i in range(len(cl))])
    return u, du
        
        
def correct_diag_point(matrix, p, m, n):
    tm = matrix[p,m,m].copy()
    tn = matrix[p,n,n].copy()
    matrix[p,m,m], matrix[p,n,n] = tn, tm
    return matrix


def correct_diag_points(matrix, p, m, n):
    tm = matrix[:p,m,m].copy()
    tn = matrix[:p,n,n].copy()
    matrix[:p,m,m], matrix[:p,n,n] = tn, tm
    return matrix


def transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


def flip_at_point(matrix, p, m, n):
    thor = matrix[:p,m,:].copy()
    matrix[:p,m,:], matrix[:p,n,:] = matrix[:p,n,:].copy(), thor
    tver = matrix[:p,:,m].copy()
    matrix[:p,:,m], matrix[:p,:,n] = matrix[:p,:,n].copy(), tver
    return matrix


def state_to_str(state):
    start = str(state['n']) + '.' +  str(state['sym']) + '.' +  str(state['s'])
    finish = state['sz']
    return f'{start},sz={finish}'


def get_lambda(state):
    return state['sign']*state['lam']+state['sz']


def trans_matrix(basin, basout):
    u = np.eye(len(basin),dtype=complex)
    for i in range(len(basin)):
        for j in range(len(basout)):
            if basin[i]['lam'] < 2:
                if all([basin[i][k]==basout[j][k] for k in state_keys]):
                    if basin[i]['lam'] != 0:
                        if basin[i]['lam'] == 1 and basin[i]['sym'] == 2:
                            u[i,j] = 1j/np.sqrt(2)
                        elif basin[i]['lam'] == 1 and basin[i]['sym'] == 3:
                            u[i,j] = -basout[j]['sign']/np.sqrt(2)
            else:
                if all([basin[i][k]==basout[j][k] for k in state_keys[1:]]):
                    if basin[i]['lam'] == 2 and basin[i]['sym'] == 1:
                        u[i,j] = 1j/np.sqrt(2)
                    elif basin[i]['lam'] == 2 and basin[i]['sym'] == 4:
                        u[i,j] = -basout[j]['sign']/np.sqrt(2)        
    return u


def correct_delta(basin, matrix):
    for i in range(len(basin)):
        for j in range(len(basin)):
            if all([basin[i][k]==basin[j][k] for k in state_keys[1:]]) and basin[i]['lam'] == 2:
                mean = (matrix[:,i,i] + matrix[:,j,j]) / 2
                matrix[:,i,i], matrix[:,j,j] = mean.copy(), mean.copy()
    return matrix


def get_diagonal(r, matrix, name, form='%.2f'):
    result = [r]
    header = 'R'
    for i in range(len(matrix[-1])):
        result.append(np.real(matrix[:,i,i]))
        header += ' ' + f'E{i}'
    np.savetxt(f'H_{name}_diag.txt', np.array(result).T, fmt=form, header=header, comments='')


def get_nondiag(r, matrix, name, eps=1., form='%.2f'):
    result_re = [r]
    header_re = 'R'
    n = 0
    for i in range(len(matrix[-1])):
        for j in range(i+1, len(matrix[-1])):
            if any([np.abs(np.real(c)) >= eps for c in matrix[:,i,j]]):
                result_re.append(np.real(matrix[:,i,j]))
                header_re += ' ' + f'Re<{i}|V|{j}>'
                n += 1
    if n != 0:
        np.savetxt(f'Re_{name}.txt', np.array(result_re).T, fmt=form, header=header_re, comments='')

    result_im = [r]
    header_im = 'R'
    n = 0
    for i in range(len(matrix[-1])):
        for j in range(i+1, len(matrix[-1])):
            if any([np.abs(np.imag(c)) >= eps for c in matrix[:,i,j]]):
                result_im.append(np.imag(matrix[:,i,j]))
                header_im += ' ' + f'Im<{i}|V|{j}>'
                n += 1
    if n != 0:
        np.savetxt(f'Im_{name}.txt', np.array(result_im).T, fmt=form, header=header_im, comments='')


def get_matrix(r, matrix, name, eps=1., form='%.2f'):
    get_diagonal(r, matrix, name, form=form)
    if len(matrix[-1]) > 1:
        get_nondiag(r, matrix, f'V_{name}', eps, form=form)


def get_single(array, sub):
    for f in array:
        if f.find(sub) != -1:
            return f


def load_matrices(dir):
    diagf = get_single(os.listdir(dir), 'diag')
    nondiagf = [f for f in os.listdir(dir) if f.find('V') != -1]
    ddrf = [f for f in os.listdir(dir) if f.find('ddR') != -1]

    ids = open(f'{dir}/{diagf}').readlines()[0].split()
    ediags = np.loadtxt(f'{dir}/{diagf}', skiprows=1)

    r = ediags[:,0]
    es = ediags[:,1:].transpose()
    hls = np.zeros((len(r), len(ids)-1, len(ids)-1), dtype=complex)
    ddrls = np.zeros((len(r), len(ids)-1, len(ids)-1), dtype=complex)
    for i in range(len(es)):
        hls[:,i,i] = es[i].copy()

    for f in nondiagf:
        nondiags = np.loadtxt(f'{dir}/{f}', skiprows=1).transpose()[1:]
        ids = open(f'{dir}/{f}').readlines()[0].split()[1:]
        for i in range(len(ids)):
            k = [int(j) for j in ids[i][3:-1].split('|')[::2]]
            if f.find('Re') != -1:
                hls[:, k[0], k[1]] += nondiags[i]
                hls[:, k[1], k[0]] += nondiags[i]
            elif f.find('Im') != -1:
                hls[:, k[0], k[1]] += 1j*nondiags[i]
                hls[:, k[1], k[0]] += -1j*nondiags[i]

    for f in ddrf:
        nondiags = np.loadtxt(f'{dir}/{f}', skiprows=1).transpose()[1:]
        ids = open(f'{dir}/{f}').readlines()[0].split()[1:]
        for i in range(len(ids)):
            k = [int(j) for j in ids[i][3:-1].split('|')[::2]]
            if f.find('Re') != -1:
                ddrls[:, k[0], k[1]] += nondiags[i]
                ddrls[:, k[1], k[0]] += -nondiags[i]
            elif f.find('Im') != -1:
                ddrls[:, k[0], k[1]] += 1j*nondiags[i]
                ddrls[:, k[1], k[0]] += -1j*nondiags[i]
    
    return r, hls, ddrls


def matrix_funcs(r, hls, ddrls):
    return CubicSpline(r, hls), CubicSpline(r, ddrls)


def energy_plot(dirgroup, palette):
    plotstates = []
    for dir in dirgroup:
        r, hls, ddrls = load_matrices(dir)
        n = len(hls[0])
        group = dir[-1]
        omega = dir[2:-2] if dir[2] != '0' else ('0+' if dir[2:5]=='0_p' else '0-')
        signs = labels[group][omega]
        for i in range(n):
            if (signs[i][1] == 'p' and int(signs[i][2:]) > 7) or (signs[i][1] == 's' and int(signs[i][2:]) > 3):
                plotstates.append(signs[i])
                plt.plot(r, np.real(hls[:,i,i]), ls=lines[omega], color=palette[signs[i]])
    for s in set(plotstates):
        plt.plot([np.min(r), np.max(r)], [nist_energies[s], nist_energies[s]], ls=lines['nist'], color=palette[s])
    plt.grid(visible=False)
    if group == 'p':
        legend1 = plt.legend(handles=ppatchst, loc=1)
        plt.legend(handles=ppatchom, loc=9)
        plt.gca().add_artist(legend1)
    elif group == 's':
        legend1 = plt.legend(handles=spatchst, loc=1)
        plt.legend(handles=spatchom, loc=9)
        plt.gca().add_artist(legend1)
    plt.xlabel('R, \AA')
    plt.ylabel('E, cm${}^{-1}$')
    # plt.show()
    plt.savefig(f'images/Energies_{group}.pdf')
    plt.close()


def so_plot(dirgroup):
    for dir in dirgroup:
        omega = dir[2:-2] if dir[2] != '0' else ('0+' if dir[2:5]=='0_p' else '0-')
        r, hls, ddrls = load_matrices(dir)
        n = len(hls[0])
        group = dir[-1]

        signs = labels[group][omega]
        for i in range(n):
            for j in range(i+1,n):
                if (group == 'p' and int(signs[i][2:]) > 7 and int(signs[j][2:]) > 7) or (signs[i][1] == 's' and int(signs[i][2:]) > 3 and int(signs[j][2:]) > 3):
                    signre = f'$\Re V_{{{signs[i][:2]}_{{{signs[i][2:]}}}, {signs[j][:2]}_{{{signs[j][2:]}}}}}$'
                    signim = f'$\Im V_{{{signs[i][:2]}_{{{signs[i][2:]}}}, {signs[j][:2]}_{{{signs[j][2:]}}}}}$'
                    if any([np.abs(c) >= 1e-2 for c in np.real(hls[:,i,j])]):
                        plt.plot(r, np.real(hls[:,i,j]), label=f'{signre}, $\Omega={omega}$', ls=lines[omega])
                    if any([np.abs(c) >= 1e-2 for c in np.imag(hls[:,i,j])]):
                        plt.plot(r, np.imag(hls[:,i,j]), label=f'{signim}, $\Omega={omega}$', ls=lines[omega])
    plt.xlabel('R, \AA')
    plt.ylabel('V, cm${}^{-1}$')
    plt.legend()
    plt.savefig(f'images/Spin-orbit_{group}.eps')
    plt.close()


def ddr_plot(dirgroup):
    for dir in dirgroup:
        omega = dir[2:-2] if dir[2] != '0' else ('0+' if dir[2:5]=='0_p' else '0-')
        r, hls, ddrls = load_matrices(dir)
        n = len(hls[0])
        
        group = dir[-1]
        signs = labels[group][omega]
        for i in range(n):
            for j in range(i+1,n):
                if (group == 'p' and int(signs[i][2:]) > 7 and int(signs[j][2:]) > 7) or (signs[i][1] == 's' and int(signs[i][2:]) > 3 and int(signs[j][2:]) > 3):
                    signre = f'$\Re m_{{{signs[i][:2]}_{{{signs[i][2:]}}}, {signs[j][:2]}_{{{signs[j][2:]}}}}}$'
                    signim = f'$\Im m_{{{signs[i][:2]}_{{{signs[i][2:]}}}, {signs[j][:2]}_{{{signs[j][2:]}}}}}$'
                    if any([np.abs(c) >= 1e-2 for c in np.real(ddrls[:,i,j])]):
                        plt.plot(r, np.real(ddrls[:,i,j]), label=f'{signre}, $\Omega={omega}$', ls=lines[omega])
                    if any([np.abs(c) >= 1e-2 for c in np.imag(ddrls[:,i,j])]):
                        plt.plot(r, np.imag(ddrls[:,i,j]), label=f'{signim}, $\Omega={omega}$', ls=lines[omega])
    plt.xlabel('R, \AA')
    plt.ylabel('m, \AA ${}^{-1}$')
    plt.legend()
    plt.savefig(f'images/Nondiag_{group}.eps')
    plt.close()


def elastic_plot(dirgroup, palette):
    emax = []
    emin = []
    for dir in dirgroup:
        r, hls, ddrls = load_matrices(dir)
        n = len(hls[0])
        sig_mat = np.loadtxt(f'{dir}/sigmas_total_new.txt', skiprows=1)
        e = sig_mat[:,0]
        emax.append(np.max(e))
        emin.append(np.min(e))
        sigmas = np.zeros((len(e),n,n))
        for i in range(n):
            for j in range(n):
                sigmas[:,i,j] = sig_mat[:,i*n+j+1]
        group = dir[-1]
        omega = dir[2:-2] if dir[2] != '0' else ('0+' if dir[2:5]=='0_p' else '0-')
        signs = labels[group][omega]
        for i in range(n):
            if (signs[i][1] == 'p' and int(signs[i][2:]) > 7) or (signs[i][1] == 's' and int(signs[i][2:]) > 3):
                eplot = np.linspace(np.min(e[np.argwhere(sigmas[:,i,i] != 0)]), np.max(e[np.argwhere(sigmas[:,i,i] != 0)]), 600)
                spl = make_interp_spline(e[np.argwhere(sigmas[:,i,i] != 0)].transpose()[0], np.log10(sigmas[np.argwhere(sigmas[:,i,i] != 0),i,i].transpose()[0]))
                sigmaplot = np.power(10, spl(eplot))
                plt.plot(eplot-np.real(hls[-1,i,i]), sigmaplot*1.e14, ls=lines[omega], color=palette[signs[i]])
    # plt.xlim(-100, np.min(np.array(emax))-np.min(np.array(emin))+300)
    plt.grid(visible=False)
    plt.xlabel('$E_{col}$, cm${}^{-1}$')
    plt.ylabel('$\sigma_{t}$, $10^{-14}$ cm${}^{2}$')
    if group == 'p':
        legend1 = plt.legend(handles=ppatchst, loc=1)
        plt.legend(handles=ppatchom, loc=9)
        plt.gca().add_artist(legend1)
    elif group == 's':
        legend1 = plt.legend(handles=spatchst, loc=1)
        plt.legend(handles=spatchom, loc=9)
        plt.gca().add_artist(legend1)
    plt.savefig(f'images/Elastic_{group}.eps')
    plt.close()


def diffuse_plot(dirgroup, palette):
    emax = []
    emin = []
    for dir in dirgroup:
        r, hls, ddrls = load_matrices(dir)
        n = len(hls[0])
        sig_mat = np.loadtxt(f'{dir}/sigmas_diff.txt', skiprows=1)
        e = sig_mat[:,0]
        emax.append(np.max(e))
        emin.append(np.min(e))
        sigmas = np.zeros((len(e),n))
        for i in range(n):
            sigmas[:,i] = sig_mat[:,i+1]
        group = dir[-1]
        omega = dir[2:-2] if dir[2] != '0' else ('0+' if dir[2:5]=='0_p' else '0-')
        signs = labels[group][omega]
        for i in range(n):
            if (signs[i][1] == 'p' and int(signs[i][2:]) > 7) or (signs[i][1] == 's' and int(signs[i][2:]) > 3):
                eplot = np.linspace(np.min(e[np.argwhere(sigmas[:,i] != 0)]), np.max(e[np.argwhere(sigmas[:,i] != 0)]), 600)
                spl = make_interp_spline(e[np.argwhere(sigmas[:,i] != 0)].transpose()[0], np.log10(sigmas[np.argwhere(sigmas[:,i] != 0),i].transpose()[0]))
                sigmaplot = np.power(10, spl(eplot))
                plt.plot(eplot-np.real(hls[-1,i,i]), sigmaplot*1.e14, ls=lines[omega], color=palette[signs[i]])
    # plt.xlim(-100, np.min(np.array(emax))-np.min(np.array(emin))+300)
    plt.grid(visible=False)
    plt.xlabel('$E_{col}$, cm${}^{-1}$')
    plt.ylabel('$\sigma_{d}$, $10^{-14}$ cm${}^{2}$')
    if group == 'p':
        legend1 = plt.legend(handles=ppatchst, loc=1)
        plt.legend(handles=ppatchom, loc=9)
        plt.gca().add_artist(legend1)
    elif group == 's':
        legend1 = plt.legend(handles=spatchst, loc=1)
        plt.legend(handles=spatchom, loc=9)
        plt.gca().add_artist(legend1)
    plt.savefig(f'images/Diffuse_{group}.eps')
    plt.close()


def inelastic_plot(dirgroup):
    emax = []
    emin = []
    for dir in dirgroup:
        r, hls, ddrls = load_matrices(dir)
        n = len(hls[0])
        if n > 1:
            sig_mat = np.loadtxt(f'{dir}/sigmas_total_exp.txt', skiprows=1)
            e = sig_mat[:,0]
            emax.append(np.max(e))
            emin.append(np.min(e[np.argwhere(sig_mat[:,2] != 0)]))
            sigmas = np.zeros((len(e),n,n))
            for i in range(n):
                for j in range(n):
                    sigmas[:,i,j] = sig_mat[:,i*n+j+1]
            omega = dir[2:-2] if dir[2] != '0' else ('0+' if dir[2:5]=='0_p' else '0-')
            group = dir[-1]
            signs = labels[group][omega]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if (group == 'p' and int(signs[i][2:]) > 7 and int(signs[j][2:]) > 7) or (group == 's' and int(signs[i][2:]) > 3 and int(signs[j][2:]) > 3):
                            sign = f'${{{signs[i][:2]}_{{{signs[i][2:]}}} \\rightarrow {signs[j][:2]}_{{{signs[j][2:]}}}}}$'
                            eplot = np.linspace(np.min(e[np.argwhere(sigmas[:,i,j] != 0)]), np.max(e[np.argwhere(sigmas[:,i,j] != 0)]), 600)
                            spl = make_interp_spline(e[np.argwhere(sigmas[:,i,j] != 0)].transpose()[0], np.log10(sigmas[np.argwhere(sigmas[:,i,j] != 0),i,j].transpose()[0]))
                            sigmaplot = np.power(10, spl(eplot))
                            plt.plot(eplot-np.real(hls[-1,i,i]), sigmaplot, label=f'{sign}, $\Omega={omega}$', linestyle=lines[omega])
    plt.semilogy()
    plt.xlabel('$E_{col}$, cm${}^{-1}$')
    plt.ylabel('$\sigma$, cm${}^{2}$')
    # plt.xlim(np.min(np.array(emin))-200, np.min(np.array(emax))+200)
    if group == 'p':
        plt.legend(ncol=2,fontsize=13)
        # plt.legend(ncol=2)
    else:
        plt.legend()
    # plt.show()
    plt.savefig(f'images/Inelastic_{group}_EDWA.eps')
    plt.close()