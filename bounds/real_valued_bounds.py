import math
import numpy as np

from scipy.stats import norm
from bounds.bound_utils import log_binomial_coefficient, zeta
from bounds.kl_inv import kl_inv

def compute_epsilon(m, n_sigma, n, delta, log_divergence=None):
  if n == m:
     return np.inf
  prior_message = 1/n_sigma
  log_delta = log_binomial_coefficient(n,m) - np.log(zeta(m)*prior_message*delta)
  if log_divergence is not None:
     log_delta += log_divergence
  return log_delta / (n-m)


def kl_bound(m, n, val_error, epsilon):
    # sanity condition
    if n == m:
       return 1
    epsilon += np.log(2*np.sqrt(n-m))/ (n-m)
    return kl_inv(val_error, epsilon, "MAX")

def linear_bound(m,n, val_error, epsilon, t, min_val, max_val):
    # sanity condition
    if n == m:
       return 1
    
    epsilon = epsilon / t + (t * (max_val - min_val) ** 2)/8
    return val_error + epsilon

def catoni_bound(m, n, val_error, epsilon, C):
    # sanity condition
    if n == m:
       return 1
    if np.isinf(C) and val_error < 1e-10:
         return 1- np.exp(-epsilon)
    bound = -C * val_error - epsilon
    bound = 1- np.exp(bound)
    return bound / (1-np.exp(-C))

def compute_real_valued_bounds(m, n_sigma, n, val_error, delta, nbr_parameter_bounds,  information_dict, min_val=0, max_val=1, prefix="", log_divergence=None):
   if len(prefix) != 0:
      prefix = prefix + "_"

   bounded_val_error = (val_error - min_val) / (max_val - min_val)

   # some precision errors happened, where the normalization led to errors up to 1 + 1e-15
   if 1.0 < bounded_val_error <= 1+1e-10:
      bounded_val_error = 1.0
   elif -1e-5 < bounded_val_error < 0.0:
      bounded_val_error = 0.0

   epsilon = compute_epsilon(m, n_sigma, n, delta, log_divergence)
   information_dict[prefix + 'kl_bound'] = (max_val - min_val) * kl_bound(m, n, bounded_val_error, epsilon) + min_val
   print("Real valued bound with kl", information_dict[prefix + 'kl_bound'])

   information_dict[prefix + 'nbr_parameter_bounds'] = nbr_parameter_bounds
   epsilon = compute_epsilon(m, n_sigma, n, delta / nbr_parameter_bounds, log_divergence)

   linear_bound_parameters = np.logspace(-5, 1, nbr_parameter_bounds)
   linear_bounds = np.array([linear_bound(m, n, val_error, epsilon, t, min_val, max_val) for t in linear_bound_parameters])

   information_dict[prefix + 'min_val_linear_bound'] = np.min(linear_bounds)
   information_dict[prefix + 'min_param_linear_bound'] = linear_bound_parameters[np.argmin(linear_bounds)]

   print(f"Real valued bound with linear Delta (lambda={information_dict[prefix + 'min_param_linear_bound']}) :", information_dict[prefix + 'min_val_linear_bound'])

   catoni_bound_parameters = np.logspace(-5, 5, nbr_parameter_bounds-1)
   catoni_bounds = [catoni_bound(m, n, bounded_val_error, epsilon, C) for C in catoni_bound_parameters]
   catoni_bounds.append(catoni_bound(m, n, bounded_val_error, epsilon, np.inf))
   catoni_bounds = np.array(catoni_bounds)

   information_dict[prefix + 'min_val_catoni_bound'] = (max_val - min_val) * np.min(catoni_bounds) + min_val
   information_dict[prefix + 'min_param_catoni_bound'] = catoni_bound_parameters[np.argmin(catoni_bounds)]

   print(f"Real valued bound with Catoni Delta (C={information_dict[prefix + 'min_param_catoni_bound']}) :", information_dict[prefix + 'min_val_catoni_bound'])
  