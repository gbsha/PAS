import numpy as np

def awgn(x, noise_power, seed=None):
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = np.sqrt(noise_power) * rng.normal(size=x.shape)
    return x + noise


def bsc(x, crossover_prob, seed=None):
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = rng.choice([-1, 1], size=x.shape, p=[crossover_prob, 1 - crossover_prob])
    return x * noise
