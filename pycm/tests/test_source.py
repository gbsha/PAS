#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:19:28 2022

@author: georg
"""

from lntcm import source
import unittest

class TestSource(unittest.TestCase):
    
    def test_ntype(self):
        types = [[0, 0, 1],
                 [0, 1, 1],
                 [1, 1, 1],
                 [1, 1, 2],
                 [1, 2, 1],
                 [2, 1, 1],
                 [34, 57, 32, 23, 87]]
        for typ in types:
            idx = source.ntype(typ)
            for j in range(len(typ)):
                self.assertEqual(typ[j], sum(idx==j))
        
    
    
if __name__ == '__main__':
    unittest.main()