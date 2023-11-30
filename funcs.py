import numpy as np
from numba import njit
from scipy.special import spherical_jn as sph_jn, spherical_yn as sph_yn, lpn

state_keys = ['n', 'lam', 's', 'sz']


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
    f[1] = -k2(x, u, e, j) * y[0]
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
    f = np.ones(2*len(ejpairs))*x*1e-3
    f[1::2] = ejpairs[:,0]*1e-3
    return f

    
def mulelequations(y, es, js, u, x):
    ejpairs = np.transpose([np.tile(js, len(es)), np.repeat(es, len(js))])
    f = np.zeros(len(y))
    f[::2] = y[1::2]
    f[1::2] = -k2(x, u, ejpairs[:,1], ejpairs[:,0]) * y[::2]
    return f


def mul_sigma_calc(psi, dpsi, r, u, es, js):
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
    part_sigma = (2 * ejpairs[:,0] + 1) * (B**2 / (B**2 + A**2)) / ejpairs[:,1]
    result = np.zeros(len(es))
    for i in range(len(ejpairs)):
        for j in range(len(es)):
            if ejpairs[i,1] == es[j]:
                result[j] += part_sigma[i]
    return result
        
        
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


def convert_c2v_cinfv(sym, lam):
    return [0, 1 if sym == 1 else -1] if lam == 0 else ([lam, None] if sym in [1,2] else [-lam, None])


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