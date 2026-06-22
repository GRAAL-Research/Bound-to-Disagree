import math
from scipy.special import betainc

def tighter_p2l_bound(k, n, delta):
    """
    From the python code of [2]
    
    :param k: Compression set size
    :param n: Train set size
    :param delta: Confidence parameter
    """
    if k == n:
        return 1.0

    t1 = 0.0
    t2 = 1.0
    while t2 - t1 > 1e-10:
        t = (t1 + t2) / 2
        left = delta*betainc(k+1,n-k,t)
        right = t*n*(betainc(k,n-k+1,t) -betainc(k+1,n-k,t))
        if left > right:
            t2 = t
        else:
            t1 = t
    return t2

def two_sided_p2l_bound(k, n, delta):
    """
    Implemented from the matlab code provided by [1]
    
    :param k: Compression set size
    :param n: Train set size
    :param delta: Confidence parameter
    """
    if k == n:
        return 1
    
    t1 = k/n
    t2 = 1
    while t2 - t1 > 1e-10:
        t = (t1 + t2) / 2

        left = (delta / 2 - delta / 6) * betainc(k+1, n-k, t)
        left += (delta / 6 ) * betainc(k+1, 4*n+1-k, t)
        right = (1+delta / 6 /n) * t * n * (betainc(k, n-k+1, t) - betainc(k+1, n-k, t))
        if left > right:
            t2 = t
        else:
            t1 = t

    return t2

def p2l_upper_bound(k, n, delta):
    bound = k/n + 2 * math.sqrt(k+1) * (math.sqrt(math.log(k+1))+4)/n
    bound += 2 * math.sqrt(k+1) * math.sqrt(math.log(1/delta))/n + math.log(1/delta)/n
    return bound

def compute_all_p2l_bounds(k:int, n:int, error :int, delta:float, information_dict : dict, verbose : bool=True):
    """
    k = Size of the compression set (int)
    n = Size of the whole dataset (int)
    error = Number of error on the complement set (int)
    delta = Confidence parameter (float)
    information_dict = Dictionnary of results
    """
    # If zero error, we converged and we can compute the original P2L bound
    # Otherwise, we use the bound of P2L-ES with a union bound over all possible M
    if error < 1:
        new_delta = delta
    else:
        new_delta = delta/n

    information_dict['p2l_bound'] = two_sided_p2l_bound(error + k, n, new_delta)
    information_dict['tighter_p2l_bound'] = tighter_p2l_bound(error + k, n, new_delta)
    information_dict['p2l_upper_bound'] = p2l_upper_bound(error + k, n, new_delta)

    if verbose:
        print("P2L bound with numerical evaluation :", information_dict['p2l_bound'])
        print("Tighter P2L bound with numerical evaluation :", information_dict['tighter_p2l_bound'])
        print("P2L upper bound :", information_dict['p2l_upper_bound'])



#################
# [1] Campi, M. C., & Garatti, S. (2023). Compression, generalization and learning. Journal of Machine Learning Research, 24(339), 1-74.
# [2] Paccagnan, D., Marks, D., Campi, M. C., & Garatti, S. (2025). Pick-to-Learn for Systems and Control: Data-driven Synthesis with State-of-the-art Safety Guarantees. arXiv preprint arXiv:2512.04781.