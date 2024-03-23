import time
import numpy as np
from pycm import source, modem, channel
from scipy.stats import norm
import math


def mc(encode=None, 
       decode=None, 
       k: int = 1, 
       min_nwe: int = 20,
       max_time: int = 1, 
       SNRdB: float = 0, 
       decision: str = 'HD'):
    nwe = 0
    nbe = 0
    nb = 0
    nw = 0
    start = time.time()
    cstll = modem.ASK(1)
    SNR = 10**(SNRdB/10)
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


def campaign(SNRdBs, min_nwe, **kwargs):
    results = []
    for SNRdB in SNRdBs:
        r = mc(SNRdB=SNRdB, min_nwe=min_nwe, **kwargs)
        if r[2] < min_nwe:
            break
        results.append(r)
    res = [np.array(res) for res in zip(*results)]
    return res


def qfun(x):
    return 1 - norm.cdf(x)


def ber_uncoded(SNRdB):
    SNR = 10**(SNRdB / 10)
    return qfun(np.sqrt(SNR))


def db(x):
    return 10*np.log10(x)


def wer_bdd(n, d, SNRdB):
    SNR = 10**(SNRdB/10)
    be = qfun(np.sqrt(SNR))
    pc = 0
    if d % 2 == 1:
        t = d // 2
    else:
        t = (d - 1)//2
        w = d // 2
        pc = 0.5 * math.comb(n, w) * be**w * (1 - be)**(n - w)
    for w in range(t + 1):
        pc += math.comb(n, w) * be**w * (1 - be)**(n - w)
    return 1 - pc


def wer_bdd_t(n, t, SNRdB, m=1):
    SNR = 10**(SNRdB/10)
    be = ber_uncoded(SNRdB)
    if m > 1:
        be = 1 - (1 - be)**m
    pc = 0
    for w in range(t + 1):
        pc += math.comb(n, w) * be**w * (1 - be)**(n - w)
    return 1 - pc



def wer_sd(dmin, Amin, SNRdB):
    SNR = 10**(SNRdB / 10)
    return Amin * qfun(np.sqrt(dmin * SNR))


def wer_hd(dmin, Amin, SNRdB):
    epsilon = ber_uncoded(SNRdB)
    if dmin % 2 == 0:
        pw = 0.5 * math.comb(dmin, dmin//2)*epsilon**(dmin//2)*(1-epsilon)**(dmin//2)
        wmin = dmin//2 + 1
    else:
        pw = 0
        wmin = (dmin + 1)//2
    for w in range(wmin, dmin + 1):
        pw += math.comb(dmin, w)*epsilon**w * (1-epsilon)**(dmin - w)
    return Amin * pw


def prepare_results(SNRdB, results, which='BER'):
    if which == 'BER':
        er = results[0] / results[1]
    elif which == 'WER':
        er = results[2] / results[3]
    else:
        assert False, 'only BER and WER supported'
    return (SNRdB[:len(er)], er)


def ncg(SNRdB, BER, BER0, R):
    _BER = BER[BER > 0]
    _SNRdB = SNRdB[BER > 0]
    xc = np.log(_BER[::-1])
    xfc = _SNRdB[::-1]
    xu = np.log(ber_uncoded(SNRdB[::-1]))
    SNRdBu = np.interp(np.log(BER0), xu, SNRdB[::-1])
    SNRdBc = np.interp(np.log(BER0), xc, xfc - db(R))
    return SNRdBu - SNRdBc


def errorestimate(fun, n, min_ne, max_time):
    start = time.time()
    ne = 0
    ntx = 0
    while ne < min_ne and time.time() - start < max_time:
        ne += fun()
        ntx += n
    return np.maximum(np.array(ne),1).astype(float) / float(ntx), ne, ntx
