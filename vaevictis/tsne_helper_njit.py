import sys
import numpy as np
import numba
from numba import jit
MAX_VAL = np.log(sys.float_info.max) / 2.0

np.random.seed(0)


@jit(nopython=True)
def compute_entropy(dist=np.array([]), beta=1.0) -> np.float64:
    p = -dist * beta
    shift = MAX_VAL - max(p)
    p = np.exp(p + shift)
    sum_p = np.sum(p)

    h = (np.log(sum_p) - shift + beta * np.sum(np.multiply(dist, p)) / sum_p)
   
    return h, p / sum_p


    
@jit(nopython=True)
def compute_transition_probability(x, dist, perplexity=5.0,
                                   tol=1e-4, max_iter=50, verbose=False) -> np.float64:
    # x should be properly scaled so the distances are not either too small or too large

    (n, d) = x.shape
    
    p = np.zeros((n, n),dtype=np.float64)

    entropy = np.log(perplexity) / np.log(2.0)

    # Binary search for sigma_i
    idx = range(n)
    dd=np.zeros(n-1,dtype=np.float64)
    
    for i in range(n):
        
        beta_min = -np.inf
        beta_max = np.inf
        beta=1.0
       
        iii=0
        for ii in range(n):
            if (ii!=i):
                dd[iii]=dist[i,ii]
                iii+=1
                
        h_i, p_i = compute_entropy(dd, beta)
        h_diff = h_i - entropy

        iter_i = 0
        
        while np.abs(h_diff) > tol and iter_i < max_iter:
            if h_diff > 0:
                beta_min = beta
                if np.isfinite(beta_max):
                    beta = (beta + beta_max) / 2.0
                else:
                    beta *= 2.0
            else:
                beta_max = beta
                if np.isfinite(beta_min):
                    beta = (beta + beta_min) / 2.0
                else:
                    beta /= 2.0
            
            h_i, p_i = compute_entropy(dd, beta)
            h_diff = h_i - entropy

            iter_i += 1

       
        iii=0
        for ii in range(n):
            if (ii!=i):
                p[i, ii] =p_i[iii]
                iii+=1
        


    return p

