import numpy as np
from collections.abc import Iterable
import copy


def bi2de(bits):
    _, m = bits.shape
    k = np.arange(m)
    base = 2**k[::-1]
    return np.sum(base*bits, 1)
    

def de2bi(idx, m):
    _idx = copy.deepcopy(idx)
    if not isinstance(_idx, Iterable):
        _idx = np.array([_idx])
    n = len(_idx)
    k = np.arange(m)
    base = 2**k[::-1]
    
    bits = np.zeros((n, m), dtype=np.uint8)
    for j in range(m):
        t = _idx >= base[j]
        bits[t, j] = 1
        _idx[t] = _idx[t] - base[j]
    return bits


def nkron(A, m):
    _A = 1
    for _ in range(m):
        _A = np.kron(_A, A)
    return _A


def oh2r(OH):
    return 1 / (OH / 100 + 1)


def r2oh(R):
    return 100 * (1 / R - 1)


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def lognchoosek(n, k):
    i = np.arange(1, k + 1)
    return np.sum(np.log((n - k + i) / i))


def logmultinomial(ks):
    lmn = []
    ksum = np.cumsum(ks)
    for n, k in zip(ksum, ks):
        lmn.append(lognchoosek(n, k))
    return np.sum(lmn)


def multinomial(lst):
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0+1:]:
        for j in range(1,a+1):
            res *= i
            res //= j
            i -= 1
    return res


def version():
    print("alpha")
