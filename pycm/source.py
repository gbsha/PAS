import numpy as np


def uniform(num_symbols, bits_per_symbol=None, seed=None):
    rng = np.random.Generator(np.random.PCG64(seed))
    if bits_per_symbol is None:
        shape = (num_symbols,)
    else:
        shape = (num_symbols, bits_per_symbol)
    return rng.choice([0, 1], size=shape)


def constant_amplitude_composition(amplitude_composition, amplitude_label):
    _composition = np.random.permutation(amplitude_composition)
    amplitude_bits = amplitude_label[_composition, :]
    sign_bits = np.random.choice([0, 1], (len(amplitude_composition), 1))
    bits = np.concatenate((sign_bits, amplitude_bits), 1)
    return bits.reshape((-1,))


def ntype(typ):
    n = np.sum(typ)
    composition = np.zeros(n, dtype=np.uint8)
    cumtyp = np.concatenate(([0], np.cumsum(typ)))
    for j in range(len(typ)):
        composition[cumtyp[j]:cumtyp[j+1]] = j
    return np.random.permutation(composition)
    