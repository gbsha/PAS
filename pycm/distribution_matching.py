import numpy as np
from pycm import utils


class DM():
    def __init__(self, w, n):
        self.w = w
        self.n = n
        (self.all_types, 
         self.all_weights, 
         self.all_logsizes, 
         self.all_log2cumsizes) = self.gettypesizes(w, n)
    
        self.mcdm_types = None
        self.mcdm_weights = None
        self.mcdm_logsizes = None
        self.mcdm_logsize_maxtype = None
        self.mcdm_log2cumsizes = None
        self.mcdm_P = None

        self.ccdm_types = None
        self.ccdm_weight = None
        self.ccdm_logsize = None
        self.ccdm_P = None
    
    @staticmethod
    def gettypesizes(w, n):
        M = len(w)
        w = np.array(w)
        coeffs = []
        for _ in range(M-1):
            coeffs.append(np.arange(n + 1, dtype=np.uint16))
        types = utils.cartesian(coeffs)
        valid = np.sum(types, 1) <= n
        types = types[valid]
        types = np.concatenate((types, n - np.sum(types, 1, keepdims=True, dtype=np.uint16)), 1)
        ntypes = types.shape[0]
        weights = np.zeros(ntypes)
        logsizes = np.zeros(ntypes)
        for k in range(ntypes):
            typ = types[k]
            logsizes[k] = utils.logmultinomial(typ)
            weights[k] = np.sum(typ * w) / n
            if k % 100_000 == 0 and k > 0:
                print(f'{k//1000}k...')
        idx = np.argsort(weights)
        types = types[idx, :]
        logsizes = logsizes[idx]
        weights = weights[idx]
        log2cumsizes = np.log2(np.cumsum(np.exp(logsizes)))
        return types, weights, logsizes, log2cumsizes

    def set_mcdm_rate(self, k_bits):
        indices = np.arange(len(self.all_log2cumsizes))
        idx = self.all_log2cumsizes >= k_bits
        if not np.all(idx) and k_bits == np.log2(len(self.w))*self.n:
            indices = indices[-1:None]
        else:
            indices = indices[idx]
        max_idx = indices[0]
    
        d_size = np.exp(self.all_log2cumsizes[max_idx] * np.log(2)) - np.exp(k_bits * np.log(2))
        self.mcdm_logsizes = np.copy(self.all_logsizes[:max_idx + 1])
        self.mcdm_logsize_maxtype = self.mcdm_logsizes[-1]
        self.mcdm_logsizes[-1] = np.log(np.exp(self.mcdm_logsizes[-1]) - d_size)

        self.mcdm_weights = self.all_weights[:max_idx + 1]
        self.mcdm_types = self.all_types[:max_idx + 1]
        self.mcdm_log2cumsizes = np.log2(np.cumsum(np.exp(self.mcdm_logsizes)))
#        if not self.mcdm_log2cumsizes[-1] == k_bits:
#            print(f'codebook size differs by {self.mcdm_log2cumsizes[-1] == k_bits :.1e}bits')
        P_typ = np.exp(self.mcdm_logsizes) / np.sum(np.exp(self.mcdm_logsizes))
        self.mcdm_P = np.sum(self.mcdm_types / self.n * P_typ.reshape(-1, 1), 0)
        self.mcdm_P = self.mcdm_P / np.sum(self.mcdm_P)

    def set_ccdm_rate(self, k_bits):
        indices = np.arange(len(self.all_log2cumsizes))
        idx = self.all_logsizes / np.log(2) >= k_bits
        indices = indices[idx]
        if len(indices) == 0:
            assert False, f'k_bits={k_bits} not supported, largest value is {np.max(self.all_logsizes) / np.log(2)}'
        max_idx = indices[0]
    
        self.ccdm_logsize = self.all_logsizes[max_idx]
        self.ccdm_weight = self.all_weights[max_idx]
        self.ccdm_typ = self.all_types[max_idx]
        self.ccdm_P = self.ccdm_typ / self.n
