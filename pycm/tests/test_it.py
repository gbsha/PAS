from pycm import channel, it, source, modem
from pycm.fec import Uncoded
from pycm.modem import ASK
import unittest
import numpy as np
import matplotlib.pyplot as plt


class TestIT(unittest.TestCase):

    def test_bcequivocation(self):
        n = 100000
        m = 1
        bits = source.uniform(n, m)
        cstll = modem.ASK(m)
        x = ASK.mapbits(bits, cstll)
        y = channel.bsc(x, 0.)
        self.assertTrue(np.all(x == y))
        
        crossover_prob = 0.11
        self.assertAlmostEqual(it.entropy2(crossover_prob), 0.5, delta=1e-4)
        
        x = ASK.mapbits(bits, cstll)
        y = channel.bsc(x, crossover_prob)
        softbits = ASK.demapbits(y, cstll)
        
        xe, s = it.bcequivocation(bits, softbits, minimize=True)
        self.assertAlmostEqual(xe, 0.5, delta=1e-2)


    def test_awgn_ber(self):
        m = 2
        n = 100000
        SNRdB = np.arange(0, 5)
        SNR = 10**(SNRdB/10)
        cstll = ASK(bits_per_symbol=m)
        BER = []
        BER_sim = []
        for snr in SNR:
            # by integration
            b, _ = it.awgn_ber(cstll, cstll.power/snr)
            BER.append(b)
            # by simulation
            bits = source.uniform(n, m)
            x = ASK.mapbits(bits, cstll)
            y = channel.awgn(x, cstll.power/snr)
            softbits = ASK.demapbits(y, cstll, noise_power=cstll.power/snr)
            bitshat = Uncoded.decode(softbits)
            BER_sim.append(np.mean(bits != bitshat))
        plt.semilogy(SNRdB, BER, label='by integration')
        plt.semilogy(SNRdB, BER_sim, linestyle='dashed', label='by simulation')
        plt.ylabel('BER')
        plt.xlabel('SNR [dB]')
        plt.show()
        
        
    def test_awgn_equivocation(self):
        m = 3
        n = 100000
        cstll = ASK(m)
        SNRdB = np.arange(0, 5)
        SNR = 10**(SNRdB/10)
        XE = []
        XE_sim = []
        for snr in SNR:
            # by integration
            noise_power = cstll.power / snr
            XE.append(it.awgn_equivocation(cstll,
                                             noise_power,
                                             metric=it.decoding_metric(cstll,
                                                                noise_power)))
            # by simulation
            bits = source.uniform(n, m)
            x = ASK.mapbits(bits, cstll)
            y = channel.awgn(x, noise_power)
            softbits = ASK.demapbits(y, cstll, noise_power)
            xe, _ = it.bcequivocation(bits, softbits)
            XE_sim.append(xe)
            
        plt.plot(SNRdB, XE, label='by integration')
        plt.plot(SNRdB, m * np.array(XE_sim), linestyle='dashed', label='by simulation')
        plt.legend()
        plt.xlabel('SNR [dB]')
        plt.ylabel('equivocation')
        plt.show()


class TestMB(unittest.TestCase):
    
    def test_getmb(self):
        w = np.abs([1, 3, 5, 7])**2
        P_H, _ = it.getmb(w, 1.)
        cost = np.sum(P_H * w)
        P_W, _ = it.getmb(w, cost=cost)
        self.assertAlmostEqual(np.sum(np.abs(P_H - P_W)), 0)
        


if __name__ == '__main__':
    unittest.main()
