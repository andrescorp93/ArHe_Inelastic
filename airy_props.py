import numpy as np
from scipy.special import airy
from scipy.linalg import inv, eigh

def airymp(x):
    """
    Compute moduli and phases of exponentially normalized Airy functions and their derivatives.

    Parameters:
    x (float): Argument of Airy functions.

    Returns:
    theta (float): Phase of Ai(x) and Bi(x) or chi for x > 0.
    phi (float): Phase of Ai'(x) and Bi'(x) or eta for x > 0.
    mmod (float): Modulus M(x) or bar{M}(x) for x > 0.
    nmod (float): Modulus N(x) or bar{N}(x) for x > 0.
    """
    ai, aip, bi, bip = airy(x)
    if x <= 0.0:
        mmod = np.sqrt(ai**2 + bi**2)
        nmod = np.sqrt(aip**2 + bip**2)
        theta = np.arctan2(bi, ai)
        phi = np.arctan2(bip, aip)
        if x < -5.0:
            zeta = (2/3) * (-x)**(3/2) + np.pi/4
            theta -= zeta
            phi -= zeta
    else:
        # Exponentially normalized Airy functions
        xi = (2/3) * x**(3/2)
        ai_scaled = np.exp(xi) * ai
        bi_scaled = np.exp(-xi) * bi
        aip_scaled = np.exp(xi) * aip
        bip_scaled = np.exp(-xi) * bip
        # Moduli
        mmod = np.sqrt(bi_scaled**2 - ai_scaled**2)
        nmod = np.sqrt(bip_scaled**2 - aip_scaled**2)
        # Phases
        ratio = ai_scaled / bi_scaled
        theta = np.arctanh(ratio) if abs(ratio) < 1 else 0.5 * np.log((1 + ratio)/(1 - ratio + 1e-10))
        ratio = aip_scaled / bip_scaled
        phi = np.arctanh(ratio) if abs(ratio) < 1 else 0.5 * np.log((1 + ratio)/(1 - ratio + 1e-10))
    return theta, phi, mmod, nmod

def stable_spropn(drnow, eigold, hp, nch, rlast):
    """
    Compute imbedding-type propagator matrices Y1, Y2, Y4 for stable propagation.

    Parameters:
    drnow (float): Step size (R_{n+1} - R_n).
    eigold (array): Diagonal wave vectors (k_n^2)_ii at current point.
    hp (array): Diagonal derivative of wave vector matrix (W_n')_ii.
    nch (int): Number of channels.
    rlast (float): Starting radius R_n.

    Returns:
    y1, y2, y4 (arrays): Propagator matrices (NCH x NCH).
    """
    pi = np.pi
    rnow = rlast + drnow * 0.5  # Midpoint R_{n+1/2}
    r1 = rnow - drnow * 0.5    # R_n
    r2 = rnow + drnow * 0.5    # R_{n+1}
    
    y1 = np.zeros((nch, nch))
    y2 = np.zeros((nch, nch))
    y4 = np.zeros((nch, nch))
    
    for i in range(nch):
        alpha = - (hp[i]) ** (1/3) if hp[i] > 0 else (hp[i]) ** (1/3)
        beta = eigold[i] / hp[i] if hp[i] != 0 else 0
        x1 = alpha * (r1 - rnow + beta)
        x2 = alpha * (r2 - rnow + beta)
        
        theta1, phi1, m1, n1 = airymp(x1)
        theta2, phi2, m2, n2 = airymp(x2)
        
        chi1, eta1, xi1 = theta1, phi1, 0
        chi2, eta2, xi2 = theta2, phi2, 0
        if x1 > 0:
            chi1 = theta1
            eta1 = phi1
            xi1 = (2/3) * x1 ** (3/2)
        if x2 > 0:
            chi2 = theta2
            eta2 = phi2
            xi2 = (2/3) * x2 ** (3/2)
        
        if x1 <= 0 and x2 <= 0:
            denom = np.sin(theta2 - theta1)
            y1[i,i] = alpha * (n1 / m1) * np.sin(phi1 - theta2) / denom if denom != 0 else 0
            y2[i,i] = alpha / (pi * m1 * m2 * denom) if denom != 0 else 0
            y4[i,i] = alpha * (n2 / m2) * np.sin(phi2 - theta1) / denom if denom != 0 else 0
        elif x1 > 0 and x2 > 0:
            dplus = np.sinh(chi1 - chi2) + np.tanh(xi2 - xi1) * np.sinh(chi1 + chi2)
            y1[i,i] = alpha * (n1 / m1) * (np.sinh(chi2 - eta1) - np.tanh(xi2 - xi1) * np.sinh(chi2 + eta1)) / dplus if dplus != 0 else 0
            y2[i,i] = alpha / (pi * m1 * m2 * np.cosh(xi2 - xi1) * dplus) if dplus != 0 else 0
            y4[i,i] = alpha * (n2 / m2) * (np.sinh(chi1 - eta2) + np.tanh(xi2 - xi1) * np.sinh(chi1 + eta2)) / dplus if dplus != 0 else 0
        # Mixed cases can be added if needed, following article expressions
    
    return y1, y2, y4
