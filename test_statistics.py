import numpy as np
from scipy.stats import hypergeom
from scipy.stats import binom

def T5(R, n_basepairs, q_size):
    return n_basepairs * R[:,0] / (R[:,2] * q_size)

def T6(R, n_basepairs, q_size):
    o = R[:,0]
    n = q_size
    p = R[:,2] / n_basepairs
    S = n * p * (1 - p)

    return (o - n * p) / \
            np.sqrt(S)


def pval(R, n_basepairs, q_size):
    t = R[:,0]
    pvals = np.zeros(len(t))
    for i in range(0, len(t)):
        pvals[i] = hypergeom.cdf(t[i], n_basepairs, q_size, R[i,2])

    return  1 - pvals


def binomial_variance(n, p):
    #TODO: should not be here
    global alt_var_cnt

    n1 = np.floor(n*p)
    diff = binom.pmf(np.arange(0, n1), n, p)
    s = np.sum(diff * np.power(np.arange(0, n1) - n*p, 2))
    old_var = n*p*(1-p)
    ret = (old_var - s) / (1- np.sum(diff))
    alt_var_cnt += 1
    return ret
