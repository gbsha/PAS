from pycm import channel
import unittest
import numpy as np
from pycm import channel


class TestChannel(unittest.TestCase):

    def test_bsc(self):
        self.assertRaises(ValueError, channel.bsc, x=np.array([-1., 2.]), crossover_prob=0.1)
        self.assertRaises(ValueError, channel.bsc, x=np.array([-1., 1.]), crossover_prob=-0.1)
        x = np.array([-1, 1.])
        y = channel.bsc(x, crossover_prob=0.)
        self.assertTrue(np.all(y == x))

if __name__ == "__main__":
    unittest.main()
