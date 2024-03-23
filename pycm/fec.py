import numpy as np
from pycm import utils
import sympy


def decide(softbits):
    return (softbits < 0).astype(np.uint8)
    

class Uncoded():
    
    @staticmethod
    def decode(softbits):
        return (softbits < 0).astype(np.uint8)


def polarmatrix(m):
    return utils.nkron(np.array([[1, 0],[1, 1]]), m)


class RM():

    @staticmethod
    def generator(m, r):
        G = polarmatrix(m)
        weights = np.sum(G, 1)
        idx = weights >= 2**(m - r)
        return G[idx, :]
    
    @staticmethod
    def paritychecker(m, r):
        return RM.generator(m, m - r - 1)

    @staticmethod
    def Amin(r, m):
        tmp = r
        for i in range(m - r):
            tmp = tmp + np.log2((2**(m - i) - 1)
                                /(2**(m - r - i) - 1))
        return 2**tmp

class Hamming():
        
    @staticmethod
    def paritychecker(m, extended=False):        
        H = RM.paritychecker(m, m - 2)
        nk, n = H.shape
        return H[:nk - int(not extended), :n - int(not extended)]


class SPC():
    @staticmethod
    def encode(u):
        return np.append(u, np.sum(u) % 2)
    
    @staticmethod
    def decode_sd(softbits):
        chat = decide(softbits)
        if np.sum(chat) % 2 == 1:
            ierror = np.argmin(np.abs(softbits))
            chat[ierror] = (chat[ierror] + 1) % 2
            
        return chat[:-1]

    @staticmethod
    def paritychecker(n):
        return np.ones(n).reshape((1, n))
    

class Repetition():
    def __init__(self, n):
        self.n = n
    
    @staticmethod
    def paritychecker(n):
        return np.concatenate((np.ones((n-1,  1), dtype=np.uint8), np.identity(n - 1, dtype=np.uint8)), 1)

    def encode(self, u):
        return np.tile(u, self.n)

    @staticmethod
    def decode_hd(softbits):
        chat = decide(softbits)
        n = len(chat)
        if np.sum(chat) <= n / 2.:
            u = 0
        else:
            u = 1
        return np.array([u], dtype=np.uint8)


class FEC():
    def __init__(self, H):
        self.H = FEC.leftsystematic(H)
        self.QT, self.RTinv = FEC.get_QR(self.H)
        self.nk, self.n = self.H.shape
        self.k = self.n - self.nk
        

    def encode(self, u):
        return np.append(u, self.parity(np.array(u)))


    def parity(self, u):
        return (u.dot(self.QT).dot(self.RTinv) % 2).astype(np.uint8)
        
    
    def decode_sd(self, softbits):
        best_w = -1
        best_score = -np.inf
        for w in range(2**self.k):
            c = self.encode(utils.de2bi(w, self.k))
            c_score = np.sum((1. - 2.*c)*softbits)
            if c_score > best_score:
                best_score = c_score
                best_w = w
        return utils.de2bi(best_w, self.k).reshape(-1)
        

    @staticmethod
    def get_QR(H):
        nk,n = H.shape
        k = n-nk
        Q = H[:, :k]
        R = H[:, k:]
        RTinv = np.linalg.inv(R.T)%2
        RTinv = RTinv.astype(int)
        return Q.T, RTinv

    @staticmethod
    def leftsystematic(H):
        nk, n = H.shape
        if sympy.Matrix(H[:, -nk:]).rank() == nk:
            return H
        _, inds = sympy.Matrix(H).rref()
        idx = np.zeros(H.shape[1], dtype=bool)
        idx[list(inds)] = True
        return np.concatenate((H[:, ~idx], H[:, idx]), 1)
