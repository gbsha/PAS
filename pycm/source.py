import numpy as np


def uniform(num_symbols: int, bits_per_symbol: int = None, seed: int = None):
    """
    Uniform bit source.

    Parameters
    ----------
    num_symbols : int
        Number of symbols.
    bits_per_symbol : int, optional.
        Number of bits_per_symbol, default is None.
        See Returns for resulting signal formats.
    seed : int, optional
        For the same integer seed, the same signal is sampled.
        For None (default) system entropy is used and each call
        returns signal independent from previous calls.

    Returns
    -------
    bits : ndarray
        bits_per_symbol=i, i>0: (n, i) binary signal is returned.
        bits_per_symbol=None:   (n,) binary signal is returned.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    if bits_per_symbol is None:
        shape = (num_symbols,)
    else:
        shape = (num_symbols, bits_per_symbol)
    return rng.choice([0, 1], size=shape)


def constant_amplitude_composition(amplitude_composition, amplitude_label):
    """
    Sample binary signal with uniform sign bits and constant composition amplitude bits.

    Parameters
    ----------
    amplitude_composition : (n,) int ndarray
        Sequence of n ints with values in range(2**(m-1)). The order of the entries
        has no effect, as the sequence is randomly permuted. The number of occurences
        of each value determines the distribution of the amplitude bits.
    amplitude_label : (2**(m-1), m-1) ndarray
        Binary amplitude label.
    
    Returns
    -------
    bits : (n, m) ndarray
        bits[:, 0] are uniformly distributed sign bits. bits[:, 1:] are amplitude bits
        according to amplitude_composition (randomly permuted) and amplitude_label.
    """
    _composition = np.random.permutation(amplitude_composition)
    amplitude_bits = amplitude_label[_composition, :]
    sign_bits = np.random.choice([0, 1], (len(amplitude_composition), 1))
    bits = np.concatenate((sign_bits, amplitude_bits), 1)
    return bits.reshape((-1,))


def ntype(typ):
    """
    Sample ntype sequence.

    Parameters
    ----------
    typ : (M,) ndarray
        typ[i] is the number of occurence of symbol i in a sequence of length
        n = sum(typ).

    Returns
    -------
    symbols : (n,) ndarray
        Random ntype sequence with i in range(len(typ)) occuring typ[i] times.
    """
    n = np.sum(typ)
    composition = np.zeros(n, dtype=np.uint8)
    cumtyp = np.concatenate(([0], np.cumsum(typ)))
    for j in range(len(typ)):
        composition[cumtyp[j]:cumtyp[j+1]] = j
    return np.random.permutation(composition)
    