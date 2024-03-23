import numpy as np
import copy
import scipy
from pycm.modem import ASK
from scipy import integrate, optimize


def bcequivocation(bits, softbits, minimize: bool = False):
    _softbits = copy.deepcopy(softbits)
    _softbits[softbits > 45] = 45
    _softbits[softbits < - 45] = - 45
    def fun(s):
        return np.mean(np.log2(1 + np.exp(-s*(1 - 2 * bits) * softbits)))
    if minimize:
        res = scipy.optimize.minimize_scalar(fun, bounds=(-11, 11))
        return fun(res.x), res.x
    else:
        return fun(1), 1
    
    
def centropy(p, q):
    _p = np.array(p)
    _q = np.array(q)
    return np.sum(-_p[_p > 0] * np.log2(_q[_p > 0]))


def entropy(p):
    return centropy(p, p)


def entropy2(p):
    return entropy(np.array([p, 1 - p]))
    

def _pZ(z, noise_power):
    return 1 / np.sqrt(2 * np.pi * noise_power) * np.exp(-z**2/2/noise_power)


def awgn_ber(cstll, noise_power):
        
    def bithat(y, b, j):
        if np.isscalar(y):
            _y = np.array([y])
        L = ASK.demapbits(_y, cstll, noise_power=noise_power)[:, j].reshape(-1)
        if np.isscalar(y):
            L = L[0]
        return (((1- 2*b) * L) < 0) + 0.5 * (L == 0)

    M, m = cstll.label.shape
    
    BER = np.zeros(m)
    # loop over bits
    for j in range(m):
            
        # loop over 0, 1
        for b in [0, 1]:
            X = cstll.alphabet[cstll.label[:, j] == b]
            pX = cstll.pX[cstll.label[:, j] == b]
            
            # loop over X0 (X1)
            for k in range(M//2):
                def fun(y):
                    return _pZ(y - X[k], noise_power) * bithat(y, b, j)
                ber_on_x, _ = integrate.quad(fun, -np.inf, np.inf)
                BER[j] = BER[j] + pX[k] * ber_on_x
    return np.mean(BER), BER


def decoding_metric(cstll, noise_power, whichmetric: str = 'bmd'):
    if whichmetric == 'bmd':
        M, m = cstll.label.shape
        def metric(idx, y):
            q = 1
            for j in range(m):
                b = cstll.label[idx, j]
                X = cstll.alphabet[cstll.label[:, j] == b]
                pX = cstll.pX[cstll.label[:, j] == b]
                qb = 0
                for k in range(len(X)):
                    qb = qb + pX[k] * _pZ(y - X[k], noise_power)
                q = q * qb
            return q
    elif whichmetric == 'smd':
        def metric(idx, y):
            return cstll.pX[idx] * _pZ(y - cstll.alphabet[idx], noise_power)
    else:
        assert False, f'supported metrics are bmd, however, whichmetric={whichmetric}'
    return metric


def awgn_equivocation(cstll, noise_power, whichmetric=None, metric=None):
    if whichmetric is not None:
        if metric is not None:
            assert False, 'either specify whichmetric or metric, not both'
        metric = decoding_metric(cstll, noise_power, whichmetric)
    elif metric is None:
        assert False, 'either whichmetric or metric must be specified'
    M = len(cstll.alphabet)
    
    # define equivocation
    def equ(idx, y):
        equj = metric(idx, y)
        if equj == 0:
            return 0
        equ_avg = 0
        for j in range(M):
            equ_avg = equ_avg + metric(j, y)
        return -np.log2(equj / equ_avg)
    
    # loop over alphabet
    equivocation = 0
    for j in range(M):
        # Gaussian expectation          
        def fun(y):
            return _pZ(y - cstll.alphabet[j], noise_power) * equ(j, y)
        equ_on_x, _ = integrate.quad(fun, -np.inf, np.inf)

        equivocation = equivocation + cstll.pX[j] * equ_on_x
    return equivocation


def getmb(w, entropy=None, cost=None):
    assert ((entropy is not None and cost is None)
            or (entropy is None and cost is not None)
            ), 'you must either specify entropy or cost, not both'
    if entropy is not None:
        return _getmb_from_entropy(w, entropy)
    elif cost is not None:
        return _getmb_from_cost(w, cost)
    else:
        assert False, 'We should never have gotten here'


def _getmb_from_entropy(w, H):
    w = np.array(w)
    f = lambda nu: entropy(np.exp(-nu * w) / sum(np.exp(-nu * w)));
    g = lambda nu: f(nu) + (f(0) - f(nu) - nu) * (nu < 0) - H
    res = optimize.root_scalar(g, bracket=(0., 10.))
    nu = res.root
    return np.exp(-nu * w) / sum(np.exp(-nu * w)), res


def _getmb_from_cost(w, W):
    w = np.array(w)
    f = lambda nu: np.sum(np.exp(-nu * w) * w / sum(np.exp(-nu * w)));
    g = lambda nu: f(nu) + (f(0) - f(nu) - nu) * (nu < 0) - W
    res = optimize.root_scalar(g, bracket=(0., 10.))
    nu = res.root
    return np.exp(-nu * w) / sum(np.exp(-nu * w)), res



def vdquant(p, n):
    c = np.floor(p * n).astype(int)
    d = c - p * n
    idx = np.argsort(d)
    nd = n - np.sum(c)
    c[idx[:nd]] = c[idx[:nd]] + 1
    return c / n, c