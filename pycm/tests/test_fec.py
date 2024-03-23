#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:53:20 2021

@author: georg
"""

from pycm.fec import RM, Hamming, FEC, SPC, Repetition
import unittest
import numpy as np
import sympy
from pycm.modem import ASK
from pycm import modem, utils

class TestFEC(unittest.TestCase):

    def test_RM(self):
        m = 4
        r = 3
        G = RM.generator(m, r)
        H = RM.paritychecker(m, r)
        self.assertTrue(np.all(G.dot(H.T) % 2 == 0))
        
    def test_Hamming(self):
        for m in range(2, 8):
            H = Hamming.paritychecker(m)
            self.assertTrue(np.all((m, 2**m - 1) == H.shape))
            
    def test_Repetition(self):
        for n in range(2,  8):
            hfec = FEC(Repetition.paritychecker(n))
            u = np.random.choice([0, 1], size=(1,))
            c = hfec.encode(u)
            self.assertEqual(n, len(c))
            self.assertTrue(np.all(c == u))

    def test_leftsystematic(self):
        m = 4
        r = 1
        nk = 2**(m - r -1)
        H = RM.paritychecker(4, 1)
        Hsys = FEC.leftsystematic(H)
        self.assertTrue(sympy.Matrix(Hsys[:, -nk:]).rank() == nk)
        
    def test_QR(self):
        H = np.array([[1, 1, 0, 1, 1, 0],
                      [1, 0, 1, 0, 1, 1],
                      [0, 1, 1, 0, 1, 0]]);
        nk, n = H.shape
        k = n - nk
        Ntx = 10000
        hfec = FEC(H)
        for i in range(Ntx):
            u = np.random.choice([0, 1], size=k)
            c = hfec.encode(u)
            self.assertTrue(np.all(c.dot(H.T) % 2 == 0))

    def test_decode_sd(self):
        n = 4
        m = 1
        hfec = FEC(H=SPC.paritychecker(n))
        cstll = ASK(m)
        for w in range(2**hfec.k):
            u = utils.de2bi(w, hfec.k).reshape(-1)
            c = hfec.encode(u)
            x = ASK.mapbits(modem.demux(c, m), cstll)
            uhat = hfec.decode_sd(x)
            self.assertTrue(np.all(u == uhat))
            
    def test_SPC_decode_sd(self):
        u = [1,0,0]
        c = SPC.encode(u)
        self.assertTrue(np.sum(c) % 2 == 0)
        self.assertTrue(len(c) == len(u) + 1)
        uhat = SPC.decode_sd(1 - 2*c)
        self.assertTrue(np.all(u == uhat))
        softbits = 1. - 2.*c
        softbits[0] = -0.5 * softbits[0]
        uhat = SPC.decode_sd(softbits)
        self.assertTrue(np.all(u == uhat))
        

if __name__ == '__main__':
    unittest.main()
