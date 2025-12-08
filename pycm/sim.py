"""Simulation tools for digital communications performance evaluation.

This module provides Monte Carlo simulation tools and theoretical performance
bounds for coded communication systems, including:
- Monte Carlo bit/word error rate simulations
- Theoretical BER/WER bounds for various decoding schemes
- Net coding gain calculations
- Q-function and related utilities
"""

import time
import numpy as np
from typing import Tuple, Callable, Optional, List, Union
from numpy.typing import NDArray, ArrayLike
from pycm import source, modem, channel
from scipy.stats import norm
import math


def mc(
    encode: Optional[Callable[[NDArray], NDArray]] = None,
    decode: Optional[Callable[[NDArray], NDArray]] = None,
    k: int = 1,
    min_nwe: int = 20,
    max_time: int = 1,
    SNRdB: float = 0,
    decision: str = "HD",
) -> Tuple[int, int, int, int]:
    """Run Monte Carlo simulation for bit and word error rates.

    Simulates a coded communication system over an AWGN channel until
    either the minimum number of word errors or maximum time is reached.

    Parameters
    ----------
    encode : Callable[[NDArray], NDArray], optional
        Encoding function that takes k information bits and returns
        encoded bits. If None, no encoding is used.
    decode : Callable[[NDArray], NDArray], optional
        Decoding function that takes soft/hard bits and returns
        k decoded information bits. If None, no decoding is used.
    k : int, optional
        Number of information bits per codeword. Default is 1.
    min_nwe : int, optional
        Minimum number of word errors to collect. Default is 20.
    max_time : int, optional
        Maximum simulation time in seconds. Default is 1.
    SNRdB : float, optional
        Signal-to-noise ratio in decibels. Default is 0.
    decision : str, optional
        Decision type: "HD" for hard decision or "SD" for soft decision.
        Default is "HD".

    Returns
    -------
    nbe : int
        Total number of bit errors.
    nb : int
        Total number of transmitted bits.
    nwe : int
        Total number of word errors.
    nw : int
        Total number of transmitted words (codewords).

    Notes
    -----
    The simulation uses ASK(1) constellation (BPSK-like) and runs until
    either min_nwe word errors are observed or max_time seconds elapse.

    BER = nbe / nb
    WER = nwe / nw
    """
    nwe = 0
    nbe = 0
    nb = 0
    nw = 0
    start = time.time()
    cstll = modem.ASK(1)
    SNR = 10 ** (SNRdB / 10)
    noise_power = cstll.power / SNR
    while time.time() - start < max_time:
        bits = source.uniform(k)
        cbits = encode(bits)
        x = modem.ASK.mapbits(modem.demux(cbits, 1), cstll)
        y = channel.awgn(x, noise_power)
        softbits = modem.ASK.demapbits(y, cstll, decision=decision)
        bitshat = decode(modem.mux(softbits))
        nb += k
        nbe_j = np.sum(bits != bitshat)
        nbe += nbe_j
        nw += 1
        nwe += int(nbe_j > 0)
        if nwe >= min_nwe:
            break
    return nbe, nb, nwe, nw


def campaign(
    SNRdBs: ArrayLike,
    min_nwe: int,
    **kwargs
) -> List[NDArray]:
    """Run simulation campaign over a range of SNR values.

    Executes Monte Carlo simulations for multiple SNR points, stopping
    early if insufficient errors are observed at a given SNR.

    Parameters
    ----------
    SNRdBs : ArrayLike
        Array of SNR values in decibels to simulate.
    min_nwe : int
        Minimum number of word errors required at each SNR point.
    **kwargs
        Additional keyword arguments passed to mc() function
        (e.g., encode, decode, k, max_time, decision).

    Returns
    -------
    List[NDArray]
        List of four arrays [nbe, nb, nwe, nw] where each array
        contains results for each simulated SNR point.
        - nbe: bit errors at each SNR
        - nb: total bits at each SNR
        - nwe: word errors at each SNR
        - nw: total words at each SNR

    Notes
    -----
    The campaign stops early if at any SNR point, the number of word
    errors collected is less than min_nwe (indicating the error rate
    is too low to measure accurately in the given time).
    """
    results = []
    for SNRdB in SNRdBs:
        r = mc(SNRdB=SNRdB, min_nwe=min_nwe, **kwargs)
        if r[2] < min_nwe:
            break
        results.append(r)
    res = [np.array(res) for res in zip(*results)]
    return res


def qfun(x: Union[float, NDArray]) -> Union[float, NDArray]:
    """Compute the Q-function (complementary error function).

    Calculates Q(x) = P(X > x) where X ~ N(0,1), i.e., the tail
    probability of the standard normal distribution.

    Parameters
    ----------
    x : float or NDArray
        Input value(s).

    Returns
    -------
    float or NDArray
        Q-function value(s).

    Notes
    -----
    Q(x) = (1/2) * erfc(x/sqrt(2)) = 1 - Φ(x)
    where Φ is the standard normal CDF.
    """
    return 1 - norm.cdf(x)


def ber_uncoded(SNRdB: Union[float, NDArray]) -> Union[float, NDArray]:
    """Calculate theoretical BER for uncoded BPSK on AWGN channel.

    Parameters
    ----------
    SNRdB : float or NDArray
        Signal-to-noise ratio in decibels.

    Returns
    -------
    float or NDArray
        Bit error rate.

    Notes
    -----
    For BPSK: BER = Q(sqrt(2*SNR))
    """
    SNR = 10 ** (SNRdB / 10)
    return qfun(np.sqrt(SNR))


def db(x: Union[float, NDArray]) -> Union[float, NDArray]:
    """Convert linear value to decibels.

    Parameters
    ----------
    x : float or NDArray
        Linear value(s).

    Returns
    -------
    float or NDArray
        Value(s) in decibels: 10 * log10(x).
    """
    return 10 * np.log10(x)


def wer_bdd(n: int, d: int, SNRdB: float) -> float:
    """Calculate word error rate for bounded distance decoding.

    Computes the theoretical WER for a binary linear code with
    bounded distance decoding (corrects up to t = floor((d-1)/2) errors).

    Parameters
    ----------
    n : int
        Code length (number of bits in codeword).
    d : int
        Minimum Hamming distance of the code.
    SNRdB : float
        Signal-to-noise ratio in decibels.

    Returns
    -------
    float
        Word error rate.

    Notes
    -----
    For odd d: t = (d-1)/2, errors corrected deterministically
    For even d: t = (d-2)/2, plus 50% correction at d/2 errors

    WER = 1 - sum_{w=0}^{t} C(n,w) * be^w * (1-be)^(n-w)
    where be is the bit error probability.
    """
    SNR = 10 ** (SNRdB / 10)
    be = qfun(np.sqrt(SNR))
    pc = 0
    if d % 2 == 1:
        t = d // 2
    else:
        t = (d - 1) // 2
        w = d // 2
        pc = 0.5 * math.comb(n, w) * be**w * (1 - be) ** (n - w)
    for w in range(t + 1):
        pc += math.comb(n, w) * be**w * (1 - be) ** (n - w)
    return 1 - pc


def wer_bdd_t(n: int, t: int, SNRdB: float, m: int = 1) -> float:
    """Calculate word error rate for t-error-correcting code.

    Computes the WER for a code that can correct up to t errors.

    Parameters
    ----------
    n : int
        Code length (number of symbols).
    t : int
        Number of errors the code can correct.
    SNRdB : float
        Signal-to-noise ratio in decibels.
    m : int, optional
        Number of bits per symbol. Default is 1 (binary symbols).

    Returns
    -------
    float
        Word error rate.

    Notes
    -----
    For m > 1, computes symbol error probability from bit error probability.
    WER = 1 - sum_{w=0}^{t} C(n,w) * se^w * (1-se)^(n-w)
    where se is the symbol error probability.
    """
    SNR = 10 ** (SNRdB / 10)
    be = ber_uncoded(SNRdB)
    if m > 1:
        be = 1 - (1 - be) ** m
    pc = 0
    for w in range(t + 1):
        pc += math.comb(n, w) * be**w * (1 - be) ** (n - w)
    return 1 - pc


def wer_sd(dmin: int, Amin: int, SNRdB: float) -> float:
    """Calculate word error rate union bound for soft decision decoding.

    Computes the union bound on WER for maximum likelihood (soft decision)
    decoding using the minimum distance and weight spectrum.

    Parameters
    ----------
    dmin : int
        Minimum Hamming distance of the code.
    Amin : int
        Number of codewords at minimum distance (weight spectrum coefficient).
    SNRdB : float
        Signal-to-noise ratio in decibels.

    Returns
    -------
    float
        Upper bound on word error rate.

    Notes
    -----
    Union bound: WER <= Amin * Q(sqrt(dmin * SNR))
    This is the dominant term in the union bound expansion.
    """
    SNR = 10 ** (SNRdB / 10)
    return Amin * qfun(np.sqrt(dmin * SNR))


def wer_hd(dmin: int, Amin: int, SNRdB: float) -> float:
    """Calculate word error rate union bound for hard decision decoding.

    Computes the union bound on WER for hard decision decoding
    using the minimum distance and weight spectrum.

    Parameters
    ----------
    dmin : int
        Minimum Hamming distance of the code.
    Amin : int
        Number of codewords at minimum distance.
    SNRdB : float
        Signal-to-noise ratio in decibels.

    Returns
    -------
    float
        Upper bound on word error rate.

    Notes
    -----
    For even dmin: includes 50% correction probability at dmin/2 errors.
    For odd dmin: starts correction at (dmin+1)/2 errors.

    WER <= Amin * sum_{w=wmin}^{dmin} C(dmin,w) * epsilon^w * (1-epsilon)^(dmin-w)
    where epsilon is the channel bit error probability.
    """
    epsilon = ber_uncoded(SNRdB)
    if dmin % 2 == 0:
        pw = (
            0.5
            * math.comb(dmin, dmin // 2)
            * epsilon ** (dmin // 2)
            * (1 - epsilon) ** (dmin // 2)
        )
        wmin = dmin // 2 + 1
    else:
        pw = 0
        wmin = (dmin + 1) // 2
    for w in range(wmin, dmin + 1):
        pw += math.comb(dmin, w) * epsilon**w * (1 - epsilon) ** (dmin - w)
    return Amin * pw


def prepare_results(
    SNRdB: NDArray,
    results: List[NDArray],
    which: str = "BER"
) -> Tuple[NDArray, NDArray]:
    """Prepare simulation results for plotting.

    Extracts and formats error rate results from simulation campaign.

    Parameters
    ----------
    SNRdB : NDArray
        Array of SNR values in dB.
    results : List[NDArray]
        List of four arrays [nbe, nb, nwe, nw] from campaign().
    which : str, optional
        Error rate type: "BER" for bit error rate or "WER" for
        word error rate. Default is "BER".

    Returns
    -------
    snr : NDArray
        SNR values corresponding to the results (truncated to
        match the number of simulation points).
    er : NDArray
        Error rate values (BER or WER).

    Raises
    ------
    AssertionError
        If which is not "BER" or "WER".
    """
    if which == "BER":
        er = results[0] / results[1]
    elif which == "WER":
        er = results[2] / results[3]
    else:
        assert False, "only BER and WER supported"
    return (SNRdB[: len(er)], er)


def ncg(SNRdB: NDArray, BER: NDArray, BER0: float, R: float) -> float:
    """Calculate net coding gain at a target BER.

    Computes the net coding gain (NCG) of a coded system compared to
    uncoded transmission at a specified bit error rate.

    Parameters
    ----------
    SNRdB : NDArray
        Array of SNR values in dB for the coded system.
    BER : NDArray
        Array of BER values corresponding to SNRdB.
    BER0 : float
        Target bit error rate for comparison.
    R : float
        Code rate (information bits / total bits).

    Returns
    -------
    float
        Net coding gain in dB.

    Notes
    -----
    NCG = SNR_uncoded(BER0) - SNR_coded(BER0) - 10*log10(R)

    This accounts for both the coding gain and the rate loss.
    Positive NCG means the coded system requires less SNR than
    the uncoded system at the same BER.
    """
    _BER = BER[BER > 0]
    _SNRdB = SNRdB[BER > 0]
    xc = np.log(_BER[::-1])
    xfc = _SNRdB[::-1]
    xu = np.log(ber_uncoded(SNRdB[::-1]))
    SNRdBu = np.interp(np.log(BER0), xu, SNRdB[::-1])
    SNRdBc = np.interp(np.log(BER0), xc, xfc - db(R))
    return SNRdBu - SNRdBc


def errorestimate(
    fun: Callable[[], int],
    n: int,
    min_ne: int,
    max_time: float
) -> Tuple[float, int, int]:
    """Generic error rate estimation framework.

    Runs a custom error-generating function repeatedly to estimate
    an error rate, stopping when minimum errors or maximum time is reached.

    Parameters
    ----------
    fun : Callable[[], int]
        Function that returns the number of errors in one trial.
        Should take no arguments.
    n : int
        Number of events per trial (e.g., bits or symbols transmitted).
    min_ne : int
        Minimum number of total errors to collect.
    max_time : float
        Maximum simulation time in seconds.

    Returns
    -------
    error_rate : float
        Estimated error rate (total errors / total events).
        Returns at least 1/ntx to avoid zero estimates.
    ne : int
        Total number of errors observed.
    ntx : int
        Total number of events transmitted.

    Notes
    -----
    This is a flexible framework that can be used for custom
    error rate simulations beyond the standard mc() function.
    """
    start = time.time()
    ne = 0
    ntx = 0
    while ne < min_ne and time.time() - start < max_time:
        ne += fun()
        ntx += n
    return np.maximum(np.array(ne), 1).astype(float) / float(ntx), ne, ntx
