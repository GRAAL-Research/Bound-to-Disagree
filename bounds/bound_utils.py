import math
import numpy as np

from scipy.stats import norm

def log_stirlings_upper_bound(n):
    """
    Robbins's upper bound for the logarithm of the factorial [1]


    """
    if n == 0:
        return 0
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n) + 1/(12 * n)

def log_stirlings_lower_bound(n):
    """
    Robbins's lower bound for the logarithm of the factorial [1]

    """
    if n == 0:
        return 0
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n) + 1/(12 * n + 1)


def log_binomial_coefficient(n, k):
    """
    Logarithm of the binomial coefficient using Stirling's approximation

    """
    try:
        bin_coeff = np.log(float(math.comb(n, k)))
        return bin_coeff
    except OverflowError:
        return (log_stirlings_upper_bound(n) -
                log_stirlings_lower_bound(k) -
                log_stirlings_lower_bound(n - k))

def zeta(x):
    return (6 / (np.pi * (x + 1) )**2)



##############
# [1] Robbins, H. (1955). A remark on Stirling's formula. The American mathematical monthly, 62(1), 26-29.