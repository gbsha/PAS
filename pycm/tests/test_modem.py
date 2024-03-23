import unittest
from pycm import modem, source, channel
from pycm.modem import ASK
import numpy as np


class TestModem(unittest.TestCase):
    
    def test_ASK(self):
        pass
    
    def test_mapbits_demapbits(self):
        n = 10
        ms = [1, 2, 3, 4]
        for m in ms:
            cstll = modem.ASK(bits_per_symbol=m)
            bits = source.uniform(n, m)
            x = ASK.mapbits(bits, cstll)
            softbits = ASK.demapbits(x, cstll)
            bits_hat = (softbits < 0).astype(np.uint8)
            self.assertTrue(np.all(bits == bits_hat))
    
    def test_demapbits(self):
        n = 10
        m = 1
        noise_power = 1.234
        cstll = modem.ASK(bits_per_symbol=m)
        bits = source.uniform(n, m)
        x = ASK.mapbits(bits, cstll)
        x = channel.awgn(x, noise_power)
        softbits = ASK.demapbits(x, cstll, noise_power=noise_power, decision='SD')
        self.assertAlmostEqual(np.max(np.abs(softbits.reshape(-1) + 2 / noise_power * x)), 0, delta=1e-15)
    
if __name__ == '__main__':
    unittest.main()