#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:17:53 2021

@author: georg
"""
import numpy as np
import scipy
import scipy.optimize


def pecdf(q, ne, n):
    '''
    cumulative distribution function of random event probability Q, assuming
    uniform prior.

    Parameters
    ----------
    q : float
        The event probability at which the CDF should be evaluated.
    ne : positive integer
        The number of times the event was observed.
    n : positive integer
        The number of trials.

    Returns
    -------
    float
        CDF(q; ne, n).

    '''
    assert ne <= n, f'The event cannot occur more often than the number of trials however, ne={ne} > n={n}'
    _q = max(q, 0)
    _q = min(q, 1)
    return scipy.special.betainc(ne + 1, n- ne + 1, _q)


def credibility_interval(ne, n, credibility):
    '''
    Calculate log-centered interval with desired credibility.

    Parameters
    ----------
    ne : positive integer
        The number of times the event was observed.
    n : positive integer
        The number of trials.
    credibility : float
        Desired credibility between 0 and 1, e.g., for 95%, choose credibility=0.95.
        credibility is the probability by which, given ne and n, the unknown
        event probability of interest lies in the credibility interval, as returned
        by this function.
    Returns
    -------
    center, lower, upper
        center is the estimate ne/n. (lower, upper) is the interval log-centerd
        around ne/n with the desired credibility.

    '''
    _ne = max(1, ne)
    qhat = _ne / n
    def fun(r):
        return pecdf(qhat*np.sqrt(r), _ne, n) - pecdf(qhat / np.sqrt(r), _ne, n) - credibility
    lower = 1
    assert fun(lower) < 0, 'something impossible happened ??;"++*%$'
    upper = 2
    while fun(upper) <= 0:
        upper *= 2
    
    res = scipy.optimize.root_scalar(fun, bracket=(1, upper))
    r = res.root
    return qhat,  qhat / np.sqrt(r), qhat*np.sqrt(r)
