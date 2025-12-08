"""Information theory tools for digital communications.

This module provides functions for calculating information-theoretic quantities
relevant to probabilistic amplitude shaping and coded modulation systems.

Functions include:
- Entropy and cross-entropy calculations
- AWGN channel bit error rate and equivocation analysis
- Maxwell-Boltzmann distribution computation
- Variable-length distribution quantization
"""

import numpy as np
import copy
import scipy
from typing import Union, Tuple, Callable, Optional, Any
from numpy.typing import NDArray, ArrayLike
from pycm.modem import ASK
from scipy import integrate, optimize


def bcequivocation(
    bits: NDArray[np.integer],
    softbits: NDArray[np.floating],
    minimize: bool = False
) -> Tuple[float, float]:
    """Calculate binary channel equivocation from bits and soft bits.

    Computes the equivocation (conditional entropy) for a binary channel
    given hard bits and their corresponding log-likelihood ratios (LLRs).
    Optionally optimizes the scaling factor for the soft bits.

    Parameters
    ----------
    bits : NDArray[np.integer]
        Binary input bits (0 or 1).
    softbits : NDArray[np.floating]
        Log-likelihood ratios (LLRs) corresponding to the bits.
    minimize : bool, optional
        If True, optimize the scaling factor for soft bits to minimize
        equivocation. If False, use scaling factor of 1. Default is False.

    Returns
    -------
    equivocation : float
        The calculated equivocation value in bits.
    scale : float
        The scaling factor used (optimized if minimize=True, otherwise 1).

    Notes
    -----
    The function clips soft bits to the range [-45, 45] to avoid numerical issues.
    The equivocation is computed as: E[log2(1 + exp(-s * (1-2b) * L))]
    where b are bits, L are LLRs, and s is the scaling factor.
    """
    _softbits = copy.deepcopy(softbits)
    _softbits[softbits > 45] = 45
    _softbits[softbits < -45] = -45
    bits = bits.astype(np.float64)

    def fun(s: float) -> float:
        return np.mean(np.log2(1 + np.exp(-s * (1 - 2 * bits) * _softbits)))

    if minimize:
        res = scipy.optimize.minimize_scalar(fun, bounds=(-11, 11))
        return fun(res.x), res.x
    else:
        return fun(1), 1


def centropy(p: ArrayLike, q: ArrayLike) -> float:
    """Calculate cross-entropy between two probability distributions.

    Computes H(p, q) = -sum(p * log2(q)) for probability distributions p and q.

    Parameters
    ----------
    p : ArrayLike
        First probability distribution.
    q : ArrayLike
        Second probability distribution.

    Returns
    -------
    float
        Cross-entropy in bits.

    Notes
    -----
    Only terms where p > 0 are included in the sum to avoid log(0).
    """
    _p = np.array(p)
    _q = np.array(q)
    return np.sum(-_p[_p > 0] * np.log2(_q[_p > 0]))


def entropy(p: ArrayLike) -> float:
    """Calculate Shannon entropy of a probability distribution.

    Computes H(p) = -sum(p * log2(p)) for probability distribution p.

    Parameters
    ----------
    p : ArrayLike
        Probability distribution.

    Returns
    -------
    float
        Entropy in bits.
    """
    return centropy(p, p)


def entropy2(p: float) -> float:
    """Calculate binary entropy function.

    Computes H(p) for a binary random variable with probabilities [p, 1-p].

    Parameters
    ----------
    p : float
        Probability of one outcome (must be in [0, 1]).

    Returns
    -------
    float
        Binary entropy in bits.
    """
    return entropy(np.array([p, 1 - p]))


def _pZ(z: Union[float, NDArray[np.floating]], noise_power: float) -> Union[float, NDArray[np.floating]]:
    """Gaussian noise probability density function.

    Computes the PDF of a zero-mean Gaussian with specified noise power.

    Parameters
    ----------
    z : float or NDArray[np.floating]
        Value(s) at which to evaluate the PDF.
    noise_power : float
        Variance of the Gaussian distribution.

    Returns
    -------
    float or NDArray[np.floating]
        PDF value(s).
    """
    return 1 / np.sqrt(2 * np.pi * noise_power) * np.exp(-(z**2) / 2 / noise_power)


def awgn_ber(cstll: ASK, noise_power: float) -> Tuple[float, NDArray[np.floating]]:
    """Calculate theoretical bit error rate on AWGN channel.

    Computes the BER for each bit position and the average BER over all positions
    for a given constellation and noise power using numerical integration.

    Parameters
    ----------
    cstll : ASK
        Constellation object with alphabet, labels, and probabilities.
    noise_power : float
        Noise variance (N0/2).

    Returns
    -------
    mean_ber : float
        Average bit error rate across all bit positions.
    ber_per_bit : NDArray[np.floating]
        Bit error rate for each bit position.

    Notes
    -----
    Uses bit-metric decoding (BMD) and numerical integration over the
    AWGN channel output distribution.
    """

    def bithat(y: Union[float, NDArray[np.floating]], b: int, j: int) -> Union[float, NDArray[np.floating]]:
        if np.isscalar(y):
            _y = np.array([y])
        L = ASK.demapbits(_y, cstll, noise_power=noise_power)[:, j].reshape(-1)
        if np.isscalar(y):
            L = L[0]
        return (((1 - 2 * b) * L) < 0) + 0.5 * (L == 0)

    M, m = cstll.label.shape

    BER = np.zeros(m)
    # loop over bits
    for j in range(m):

        # loop over 0, 1
        for b in [0, 1]:
            X = cstll.alphabet[cstll.label[:, j] == b]
            pX = cstll.pX[cstll.label[:, j] == b]

            # loop over X0 (X1)
            for k in range(M // 2):

                def fun(y: float) -> float:
                    return _pZ(y - X[k], noise_power) * bithat(y, b, j)

                ber_on_x, _ = integrate.quad(fun, -np.inf, np.inf)
                BER[j] = BER[j] + pX[k] * ber_on_x
    return np.mean(BER), BER


def decoding_metric(cstll: ASK, noise_power: float, whichmetric: str = "bmd") -> Callable[[int, float], float]:
    """Create a decoding metric function for AWGN channel.

    Returns a function that computes the decoding metric for a given
    constellation point index and received value.

    Parameters
    ----------
    cstll : ASK
        Constellation object with alphabet, labels, and probabilities.
    noise_power : float
        Noise variance (N0/2).
    whichmetric : str, optional
        Type of metric: "bmd" for bit-metric decoding or "smd" for
        symbol-metric decoding. Default is "bmd".

    Returns
    -------
    Callable[[int, float], float]
        Metric function that takes constellation index and received value,
        returns metric value.

    Raises
    ------
    AssertionError
        If whichmetric is not "bmd" or "smd".
    """
    if whichmetric == "bmd":
        M, m = cstll.label.shape

        def metric(idx: int, y: float) -> float:
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

    elif whichmetric == "smd":

        def metric(idx: int, y: float) -> float:
            return cstll.pX[idx] * _pZ(y - cstll.alphabet[idx], noise_power)

    else:
        assert False, f"supported metrics are bmd, however, whichmetric={whichmetric}"
    return metric


def awgn_equivocation(
    cstll: ASK,
    noise_power: float,
    whichmetric: Optional[str] = None,
    metric: Optional[Callable[[int, float], float]] = None
) -> float:
    """Calculate equivocation on AWGN channel.

    Computes the conditional entropy H(X|Y) for transmission over an AWGN channel,
    where X is the transmitted symbol and Y is the received value.

    Parameters
    ----------
    cstll : ASK
        Constellation object with alphabet, labels, and probabilities.
    noise_power : float
        Noise variance (N0/2).
    whichmetric : str, optional
        Type of metric: "bmd" or "smd". Used if metric is not provided.
    metric : Callable[[int, float], float], optional
        Custom metric function. If provided, whichmetric is ignored.

    Returns
    -------
    float
        Equivocation (conditional entropy) in bits.

    Raises
    ------
    AssertionError
        If both or neither of whichmetric and metric are specified.

    Notes
    -----
    Uses numerical integration over the AWGN channel output distribution
    to compute the expected conditional entropy.
    """
    if whichmetric is not None:
        if metric is not None:
            assert False, "either specify whichmetric or metric, not both"
        metric = decoding_metric(cstll, noise_power, whichmetric)
    elif metric is None:
        assert False, "either whichmetric or metric must be specified"
    M = len(cstll.alphabet)

    # define equivocation
    def equ(idx: int, y: float) -> float:
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
        def fun(y: float) -> float:
            return _pZ(y - cstll.alphabet[j], noise_power) * equ(j, y)

        equ_on_x, _ = integrate.quad(fun, -np.inf, np.inf)

        equivocation = equivocation + cstll.pX[j] * equ_on_x
    return equivocation


def getmb(
    w: ArrayLike,
    entropy: Optional[float] = None,
    cost: Optional[float] = None
) -> Tuple[NDArray[np.floating], Any]:
    """Get Maxwell-Boltzmann distribution from entropy or cost constraint.

    Computes the probability distribution that maximizes entropy subject to
    a cost constraint (or achieves a target entropy), following the
    Maxwell-Boltzmann form: p_i ∝ exp(-ν * w_i).

    Parameters
    ----------
    w : ArrayLike
        Cost/weight vector for each outcome.
    entropy : float, optional
        Target entropy in bits. Specify either entropy or cost, not both.
    cost : float, optional
        Target average cost. Specify either entropy or cost, not both.

    Returns
    -------
    distribution : NDArray[np.floating]
        Probability distribution following Maxwell-Boltzmann form.
    result : Any
        Optimization result object from scipy.optimize.root_scalar.

    Raises
    ------
    AssertionError
        If both or neither of entropy and cost are specified.

    Notes
    -----
    This is used in probabilistic amplitude shaping to design optimal
    input distributions for shaped modulation.
    """
    assert (entropy is not None and cost is None) or (
        entropy is None and cost is not None
    ), "you must either specify entropy or cost, not both"
    if entropy is not None:
        return _getmb_from_entropy(w, entropy)
    elif cost is not None:
        return _getmb_from_cost(w, cost)
    else:
        assert False, "We should never have gotten here"


def _getmb_from_entropy(w: ArrayLike, H: float) -> Tuple[NDArray[np.floating], Any]:
    """Compute Maxwell-Boltzmann distribution from target entropy.

    Parameters
    ----------
    w : ArrayLike
        Cost/weight vector.
    H : float
        Target entropy in bits.

    Returns
    -------
    distribution : NDArray[np.floating]
        Probability distribution.
    result : Any
        Optimization result.
    """
    w = np.array(w)
    f = lambda nu: entropy(np.exp(-nu * w) / sum(np.exp(-nu * w)))
    g = lambda nu: f(nu) + (f(0) - f(nu) - nu) * (nu < 0) - H
    res = optimize.root_scalar(g, bracket=(0.0, 10.0))
    nu = res.root
    return np.exp(-nu * w) / sum(np.exp(-nu * w)), res


def _getmb_from_cost(w: ArrayLike, W: float) -> Tuple[NDArray[np.floating], Any]:
    """Compute Maxwell-Boltzmann distribution from target cost.

    Parameters
    ----------
    w : ArrayLike
        Cost/weight vector.
    W : float
        Target average cost.

    Returns
    -------
    distribution : NDArray[np.floating]
        Probability distribution.
    result : Any
        Optimization result.
    """
    w = np.array(w)
    f = lambda nu: np.sum(np.exp(-nu * w) * w / sum(np.exp(-nu * w)))
    g = lambda nu: f(nu) + (f(0) - f(nu) - nu) * (nu < 0) - W
    res = optimize.root_scalar(g, bracket=(0.0, 10.0))
    nu = res.root
    return np.exp(-nu * w) / sum(np.exp(-nu * w)), res


def vdquant(p: ArrayLike, n: int) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Variable-length distribution quantization.

    Quantizes a probability distribution to have rational probabilities
    with denominator n, preserving the distribution shape as closely as possible.

    Parameters
    ----------
    p : ArrayLike
        Input probability distribution (should sum to 1).
    n : int
        Denominator for the quantized distribution (number of outcomes).

    Returns
    -------
    quantized_probs : NDArray[np.floating]
        Quantized probability distribution (sums to 1).
    counts : NDArray[np.integer]
        Integer counts for each probability (sum to n).

    Notes
    -----
    Uses a greedy algorithm to distribute the n outcomes among the
    probability bins while minimizing the approximation error.
    This is useful for distribution matching with finite blocklength.
    """
    c = np.floor(p * n).astype(int)
    d = c - p * n
    idx = np.argsort(d)
    nd = n - np.sum(c)
    c[idx[:nd]] = c[idx[:nd]] + 1
    return c / n, c
