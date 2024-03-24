import numpy as np

def awgn(x, noise_power: float, seed: int = None):
    """
    Additive White Gaussian Noise (AWGN) channel.

    Parameters
    ----------
    x : (n,) ndarray
        Input signal.
    noise_power : float
        Gaussian noise variance.
    seed: int, optional
        For same integer, each call samples the same noise sequence.
        If set to None, each call samples noise sequence independent
        from previous calls. Default is None.

    Returns
    -------
    y : (n,) ndarray
        Noisy signal x + z where z is Gaussian with variance noise_power.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = np.sqrt(noise_power) * rng.normal(size=x.shape)
    return x + noise


def bsc(x, crossover_prob: float, seed: int = None):
    """
    Binary symmetric channel.

    Parameters
    ----------
    x : (n,) ndarray
        Input signal with values in [-a, a]
    crossover_prob: float
        Crossover probability of BSC channel. For crossover_prob=0.,
        the channel is noiseless and returns input signal. For
        crossover_prob=0.5, the channel is maximally noisy.
    seed: int, optional
        For same integer, each call samples the same noise sequence.
        If set to None, each call samples noise sequence independent
        from previous calls. Default is None.

    Returns
    -------
    y : (n,) ndarray
        Noisy signal with same alphabet as input signal x.
    """
    alphabet = np.unique(x)
    if alphabet[0] != -alphabet[1] or len(alphabet) != 2:
        raise ValueError("x must be signal with values in [-a, a]")
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = rng.choice([-1, 1], size=x.shape, p=[crossover_prob, 1 - crossover_prob])
    return x * noise
