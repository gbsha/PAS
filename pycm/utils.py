"""Utility functions for digital communications computations.

This module provides various utility functions for:
- Binary-decimal conversions
- Combinatorial calculations (binomial and multinomial coefficients)
- Coding rate conversions
- Matrix operations (Kronecker products, Cartesian products)
"""

import numpy as np
from collections.abc import Iterable
from typing import Union, List, Optional
from numpy.typing import NDArray, ArrayLike
import copy


def bi2de(bits: NDArray[np.integer]) -> NDArray[np.integer]:
    """Convert binary vectors to decimal integers.

    Converts each row of a binary matrix to its decimal representation,
    with the first column being the most significant bit.

    Parameters
    ----------
    bits : NDArray[np.integer]
        Binary matrix of shape (n, m) where n is the number of binary
        vectors and m is the number of bits per vector.

    Returns
    -------
    NDArray[np.integer]
        Array of shape (n,) containing decimal representations.

    Examples
    --------
    >>> bi2de(np.array([[0, 0, 1], [1, 0, 1]]))
    array([1, 5])

    Notes
    -----
    First column is the most significant bit (MSB).
    """
    _, m = bits.shape
    k = np.arange(m)
    base = 2 ** k[::-1]
    return np.sum(base * bits, 1)


def de2bi(idx: Union[int, ArrayLike], m: int) -> NDArray[np.uint8]:
    """Convert decimal integers to binary vectors.

    Converts decimal integer(s) to their binary representations with
    a specified number of bits.

    Parameters
    ----------
    idx : int or ArrayLike
        Decimal integer(s) to convert. Can be a single integer or
        an array-like of integers.
    m : int
        Number of bits in the output binary representation.

    Returns
    -------
    NDArray[np.uint8]
        Binary matrix of shape (n, m) where n is the number of input
        integers and m is the number of bits. Each row contains the
        binary representation with MSB in the first column.

    Examples
    --------
    >>> de2bi(5, 4)
    array([[0, 1, 0, 1]], dtype=uint8)
    >>> de2bi([1, 5, 7], 4)
    array([[0, 0, 0, 1],
           [0, 1, 0, 1],
           [0, 1, 1, 1]], dtype=uint8)

    Notes
    -----
    First column is the most significant bit (MSB).
    """
    _idx = copy.deepcopy(idx)
    if not isinstance(_idx, Iterable):
        _idx = np.array([_idx])
    n = len(_idx)
    k = np.arange(m)
    base = 2 ** k[::-1]

    bits = np.zeros((n, m), dtype=np.uint8)
    for j in range(m):
        t = _idx >= base[j]
        bits[t, j] = 1
        _idx[t] = _idx[t] - base[j]
    return bits


def nkron(A: NDArray, m: int) -> NDArray:
    """Compute m-fold Kronecker product of a matrix with itself.

    Calculates A ⊗ A ⊗ ... ⊗ A (m times).

    Parameters
    ----------
    A : NDArray
        Input matrix.
    m : int
        Number of times to apply Kronecker product.

    Returns
    -------
    NDArray
        Result of m-fold Kronecker product.

    Examples
    --------
    >>> A = np.array([[1, 0], [0, 1]])
    >>> nkron(A, 2)  # Returns A ⊗ A

    Notes
    -----
    For m=0, returns scalar 1. For m=1, returns A itself.
    """
    _A = 1
    for _ in range(m):
        _A = np.kron(_A, A)
    return _A


def oh2r(OH: float) -> float:
    """Convert overhead percentage to code rate.

    Parameters
    ----------
    OH : float
        Overhead percentage. For example, 25% overhead means OH=25.

    Returns
    -------
    float
        Code rate R = 1 / (1 + OH/100).

    Examples
    --------
    >>> oh2r(25)  # 25% overhead
    0.8  # corresponds to rate 4/5

    Notes
    -----
    Overhead is defined as OH = 100 * (1/R - 1).
    """
    return 1 / (OH / 100 + 1)


def r2oh(R: float) -> float:
    """Convert code rate to overhead percentage.

    Parameters
    ----------
    R : float
        Code rate (must be in (0, 1]).

    Returns
    -------
    float
        Overhead percentage OH = 100 * (1/R - 1).

    Examples
    --------
    >>> r2oh(0.8)  # rate 4/5
    25.0  # 25% overhead

    Notes
    -----
    A rate of 1 corresponds to 0% overhead.
    A rate of 0.5 corresponds to 100% overhead.
    """
    return 100 * (1 / R - 1)


def cartesian(arrays: List[ArrayLike], out: Optional[NDArray] = None) -> NDArray:
    """Generate a cartesian product of input arrays.

    Creates all possible combinations of elements from the input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray, optional
        Array to place the cartesian product in. If None, a new array
        is created.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays, where M is the product of the sizes of
        all input arrays.

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

    Notes
    -----
    Uses recursive implementation for efficient computation.
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    # m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            # for j in xrange(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out


def lognchoosek(n: int, k: int) -> float:
    """Compute natural logarithm of binomial coefficient.

    Calculates log(C(n, k)) = log(n! / (k! * (n-k)!)) in a numerically
    stable way without computing factorials.

    Parameters
    ----------
    n : int
        Total number of items.
    k : int
        Number of items to choose.

    Returns
    -------
    float
        Natural logarithm of the binomial coefficient C(n, k).

    Examples
    --------
    >>> lognchoosek(10, 3)  # log(C(10, 3)) = log(120)
    4.787491742782046

    Notes
    -----
    Uses the formula: log(C(n,k)) = sum(log((n-k+i)/i)) for i=1 to k.
    This avoids overflow for large n and k.
    """
    i = np.arange(1, k + 1)
    return np.sum(np.log((n - k + i) / i))


def logmultinomial(ks: ArrayLike) -> float:
    """Compute natural logarithm of multinomial coefficient.

    Calculates log of the multinomial coefficient for a sequence of counts.
    The multinomial coefficient is: (sum(ks))! / (k1! * k2! * ... * km!)

    Parameters
    ----------
    ks : ArrayLike
        Array of counts [k1, k2, ..., km] where each ki >= 0.

    Returns
    -------
    float
        Natural logarithm of the multinomial coefficient.

    Examples
    --------
    >>> logmultinomial([2, 3, 1])  # log(6!/(2!*3!*1!)) = log(60)
    4.094344562222100

    Notes
    -----
    Uses sequential application of binomial coefficients:
    log(C(k1+k2, k2)) + log(C(k1+k2+k3, k3)) + ...
    """
    lmn = []
    ksum = np.cumsum(ks)
    for n, k in zip(ksum, ks):
        lmn.append(lognchoosek(n, k))
    return np.sum(lmn)


def multinomial(lst: List[int]) -> int:
    """Compute multinomial coefficient.

    Calculates the multinomial coefficient (sum(lst))! / (lst[0]! * lst[1]! * ...)
    using an efficient algorithm that avoids computing large factorials.

    Parameters
    ----------
    lst : List[int]
        List of non-negative integers representing the counts.

    Returns
    -------
    int
        The multinomial coefficient.

    Examples
    --------
    >>> multinomial([2, 3, 1])  # 6!/(2!*3!*1!)
    60

    Notes
    -----
    Uses an efficient multiplication/division algorithm that keeps
    intermediate results as small as possible to avoid overflow.
    """
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0 + 1 :]:
        for j in range(1, a + 1):
            res *= i
            res //= j
            i -= 1
    return res


def version() -> None:
    """Print the version of the pycm package.

    Currently prints "alpha" as the package is in alpha stage.
    """
    print("alpha")
