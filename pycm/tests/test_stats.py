from pycm import source, channel, modem, stats, fec
import numpy as np

import unittest

class Test_stats(unittest.TestCase):
    
    def test_credibility_interval(self):
        m = 1
        n = 100
        cross = 0.1
        credibility = 0.95
        bits = source.uniform(n, m)
        cstll = modem.ASK(m)
        x = modem.ASK.mapbits(bits, cstll)
        y = channel.bsc(x, cross)
        bitshat = fec.Uncoded.decode(modem.ASK.demapbits(y, cstll, decision='HD'))
        nbe = np.sum(bits != bitshat)
        q, l, u = stats.credibility_interval(nbe, n, credibility)
        self.assertTrue(l < q)
        self.assertTrue(q < u)
        self.assertTrue(q == nbe/n)
        self.assertAlmostEqual(stats.pecdf(u, nbe, n) - stats.pecdf(l, nbe, n), credibility)
        
        
if __name__ == '__main__':
    unittest.main()