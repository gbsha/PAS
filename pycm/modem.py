#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 18:56:49 2021

@author: georg
"""
import numpy as np
import scipy
import scipy.stats
import collections
from pycm import utils



class ASK():
    def __init__(self, bits_per_symbol: int = 1, whichlabel: str = 'gray', pX=None):
        '''
        Amplitude shift keying (ASK) constellation for mapping and demapping.

        Parameters
        ----------
        bits_per_symbol : int, optional
            Alphabetsize in bits, i.e., the alphabet contains 2**bits_per_symbol
            signal points. The default is 1.
        whichlabel : str, optional
            Identifier for binary label of signal points. Supported are 'natural'
            and 'gray', which is the default.

        Returns
        -------
        Initialized ASK constellation object.

        '''
        self.M = 2**bits_per_symbol
        self.bits_per_symbol = bits_per_symbol
        self.alphabet = np.arange(-(self.M - 1.), self.M, 2.)
        if pX is None:
            self.pX = np.ones(self.M)
            self.pX = self.pX / np.sum(self.pX)
        else:
            self.pX = pX
        self.label = ASK.get_label(bits_per_symbol, whichlabel)
        self.label2idx = np.zeros(self.M, dtype=int)
        self.label2idx[utils.bi2de(self.label)] = np.arange(self.M)
        
    @property
    def power(self):
        return np.sum(self.pX * self.alphabet**2)
        
    @staticmethod
    def get_label(m, whichlabel):
        if whichlabel == 'gray':
            if m == 1:
                label = np.array([[0],[1]], dtype='uint8')
                return label
            else: 
                label_n = ASK.get_label(m-1, whichlabel)
                tmp = 2**(m-1)
                first_half = np.hstack((np.zeros((tmp,1), dtype='uint8'), label_n)) #tuple
                second_half = np.hstack((np.ones((tmp,1), dtype='uint8'), np.flipud(label_n)))
                label = np.vstack((first_half, second_half))
                return label
            
        elif whichlabel == 'natural':
            d = np.array(range(2**m))
            power = 2**np.arange(m);
            label = np.floor((d[:,None]%(2*power))/power)
            label = np.array(label[:,::-1], dtype='uint8')
            return label
    
    @staticmethod
    def demapbits(y, cstll, noise_power=1, decision='SD'):
        sigma = np.sqrt(noise_power)
        M, m = cstll.label.shape
        n = 1
        if isinstance(y, collections.Iterable):
            n = len(y)
        llrs = np.zeros((n, m))
    
        for level in range(0,m):
            idx_0 = np.nonzero(cstll.label[:,level]==0)[0]
            tmp = y.reshape((y.size, 1))-cstll.alphabet[idx_0]
            p0 = (scipy.stats.norm.pdf(tmp, 0, sigma)*cstll.pX[idx_0]).sum(1)
            idx_1 = np.nonzero(cstll.label[:,level]==1)[0]
            tmp = y.reshape((n, 1))-cstll.alphabet[idx_1]
            p1 = (scipy.stats.norm.pdf(tmp, 0, sigma) * cstll.pX[idx_1]).sum(1)
            llrs[:,level] = np.log(p0) - np.log(p1)
    
        if decision == 'SD':
            return llrs
        elif decision == 'HD':
            return 1 - 2 * (llrs < 0)

    @staticmethod
    def mapbits(bits, cstll):
        return cstll.alphabet[cstll.label2idx[utils.bi2de(bits)]]
    

def demux(bits, bits_per_symbol):
    return bits.reshape(-1, bits_per_symbol)

def mux(bits):
    return bits.reshape(-1)
