#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 19:13:07 2021

@author: georg
"""

import unittest
import numpy as np
from scipy import special
from pycm import utils


class TestUtils(unittest.TestCase):
    
    def test_bi2de(self):
        bits = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])
        x_expected = np.array([0, 1, 2, 3])
        x_actual = utils.bi2de(bits)
        self.assertTrue(np.all(x_actual == x_expected))
        
    def test_de2bi(self):
        m = 2
        idx = np.arange(2**m)
        bits_expected = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])
        bits_actual = utils.de2bi(idx, m)
        self.assertTrue(np.all(bits_actual == bits_expected))

    def test_nkron(self):
        A = np.array([[1, 0], [1, 1]])
        m = 2
        B = utils.nkron(A, m)
        self.assertTrue(np.all(B.shape == (4, 4)))
        
    def test_logmultinomial(self):
        for a in range(1, 10):
            for b in range(1, 10):
                for c in range(1, 10):
                    expected = utils.multinomial([a, b, c])
                    actual = np.round(np.exp(utils.logmultinomial([a, b, c])))
                    self.assertTrue(expected == actual)
                    
    def test_lognchoosek(self):
        for n in range(1, 10):
            for k in range(1, n + 1):
                expected = special.comb(n, k)
                actual = np.round(np.exp(utils.lognchoosek(n, k)))
                self.assertTrue(expected == actual)

if __name__ == '__main__':
    unittest.main()