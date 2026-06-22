from scipy.stats import binom, beta
from scipy.special import comb

import numpy as np

from bounds.bound_utils import log_binomial_coefficient, zeta

def bisection_binomial_tail_inversion(k, n, delta):
  # sanity condition
  if n == 0:
    return 1
  
  low = 0
  high = 1
  while (high - low) >  1e-10:
    t = (high + low) / 2
    cdf = binom.logcdf(k, n, t)
    if cdf < delta:
      high = t
    else:
      low = t

  if np.abs(binom.logcdf(k, n, high) - delta) < 1e-5:
    return high
  else:
    return 1

def brute_force_binomial_tail_inversion(k, n, delta, precision=1e-06):
  # sanity condition
  if n == 0:
    return 1
  
  linspace = np.arange(0,1+precision, precision)
  cdf = binom.logcdf(k, n, linspace)
  diff = cdf - delta
  diff[diff < 0] = 1
  p = linspace[np.argmin(diff)]

  if np.abs(binom.logcdf(k, n, p) - delta) < 1e-2:
    return p
  else:
    return 1

def beta_ppf(k, n, delta):
  # sanity condition
  if n == 0:
    return 1
  
  ppf = beta.ppf(1-delta, k+1, n-k)
  return ppf

def beta_isf(k, n, delta):
  """
  Selon la documentation, (voir https://docs.scipy.org/doc/scipy/tutorial/stats/discrete.html#inverse-survival-function)
  isf(delta) = ppf(1-delta), et isf est parfois plus précis que ppf.
  """
  # sanity condition
  if n == 0:
    return 1
  
  isf = beta.isf(delta, k+1, n-k)
  return isf

def binomial_approximation(k, n, delta):
  # sanity condition
  if n == 0:
    return 1
  
  first = log_binomial_coefficient(n, k)
  second = -delta
  exponential = first + second
  return 1-np.exp(-exponential / (n-k))

def compute_classical_compression_bounds(m, n_sigma, n, k, delta, information_dict, log_divergence=None):
  prior_message = 1/n_sigma
  log_delta = np.log(zeta(m)*prior_message*delta) - log_binomial_coefficient(n,m)
  if log_divergence is not None:
    log_delta -= log_divergence 
  delta_prime = np.exp(log_delta)

  information_dict['brute_force_binomial_tail_inversion'] = brute_force_binomial_tail_inversion(k, n-m, log_delta)
  print("Bound computed using a brute force binomial tail inversion :", information_dict['brute_force_binomial_tail_inversion'])

  information_dict['beta_ppf'] = beta_ppf(k, n-m, delta_prime)
  print("Bound computed using the beta ppf :", information_dict['beta_ppf'])

  information_dict['beta_isf'] = beta_isf(k, n-m, delta_prime)
  print("Bound computed using the beta isf :", information_dict['beta_isf'])
  information_dict["binomial_approximation_shah"] = binomial_approximation(k, n-m, log_delta)
  print("Bound computed using the binomial approximation (Laviolette, Marchand et Shah):", information_dict["binomial_approximation_shah"])
  log_delta += np.log(zeta(k))
  information_dict['binomial_approximation_sokolova'] = binomial_approximation(k, n-m, log_delta)
  print("Bound computed using the binomial approximation (Marchand et Sokolova):", information_dict['binomial_approximation_sokolova'])