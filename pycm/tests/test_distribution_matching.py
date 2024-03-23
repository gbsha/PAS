#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:27:08 2022

@author: georg
"""

from pycm import distribution_matching as dm
from pycm import utils, it
import numpy as np
import unittest
from scipy import special


class TestDM(unittest.TestCase):
    
    def test_MCDM(self):
        n = 10
        w = np.array([1, 3, 5, 7])**2
        M = len(w)
        dmdm = dm.DM(w, n)
        # Use all types
        dmdm.set_mcdm_rate(2 * n)
        P_expected = np.ones(4) / 4
        n_types_expected = np.exp(utils.lognchoosek(n + M - 1, M - 1))
        self.assertAlmostEqual(n_types_expected, dmdm.mcdm_types.shape[0])
        self.assertAlmostEqual(np.sum(np.abs(P_expected - dmdm.mcdm_P)), 0)
      
    def test_MCDM2(self):
        n = 200
        w = np.abs([1, 3, 5, 7])**2
        dmdm = dm.DM(w, n)
        # Source
        R = 0.2
        P, _ = it.getmb(w, R)
        source_weight = np.sum(P * w)
        # MCDM
        dmdm.set_mcdm_rate(int(R * n))
        mcdm_weight = np.sum(dmdm.mcdm_P * w)
        dmdm.set_mcdm_rate(int(R * n))
        self.assertTrue(mcdm_weight > source_weight)
        
    def test_MCDM3(self):
        n = 256
        M = 4
        w = np.abs([1, 3, 5, 7])**2
        dmdm = dm.DM(w, n)
        self.assertTrue(len(dmdm.all_log2cumsizes) == special.comb(n + M - 1, M - 1))

    def test_MCDM4(self):
        n = 10
        w = np.array([1, 3, 5, 7])**2
        dmdm = dm.DM(w, n)
        # Use all types
        expected = np.copy(dmdm.all_logsizes)
        dmdm.set_mcdm_rate(n)
        self.assertTrue(np.all(dmdm.all_logsizes == expected))

        
    def test_CCDM(self):
        n = 20
        # calculate maximum supported rate
        k_bits = np.floor(utils.logmultinomial(np.array([1/4, 1/4, 1/4, 1/4]) * n) / np.log(2)).astype(int)
        w = np.array([1, 3, 5, 7])**2
        dmdm = dm.DM(w, n)

        dmdm.set_ccdm_rate(k_bits)
        expected_weight = np.mean(w)
        self.assertTrue(dmdm.ccdm_weight <= expected_weight)
        self.assertTrue(dmdm.ccdm_logsize / np.log(2) >= k_bits)
        
    
if __name__ == '__main__':
    unittest.main()