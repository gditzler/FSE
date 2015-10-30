#!/usr/bin/env python

import feast 
import numpy as np

from scipy.stats import binom
from sklearn.feature_selection import chi2
from ..utils import bin_data


def bootstrap_selection(counts, N,  normalizer="poly", poly=2.):
  """
  Parameters
  ----------
  counts : array-like
      Numpy vector containing the sums of the rows of the NPFS Bernoulli 
      matrix. 

  N : int
      Integer specifying the number of bootstraps to perform with the 
      sampling proceedure 

  normalizer : string
      Type of scaling to perform with the counts ['minmax', 'poly']

  poly : double
      If 'poly' was specified with the normalizer option, this input specifies
      the order of the polynomial. 
  
  Returns
  -------
  imp_set_sz : double 
      Estimate of the important feature subset size
  
  features : list
      List of features that were detected as relevant 
  """
  n_feat = len(counts)
  all_features = np.array(range(n_feat))
  v = np.zeros((N,))
  if isinstance(counts, np.ndarray) is False:
    counts = np.array(counts)

  z = counts.copy()
  if normalizer == "minmax":
    z = (z+0.0000001)/z.sum()
    x1 = 1./(z.min()*(z.max()/z.min()-1))
    x2 = 1./(1-z.max()/z.min())
    probs = np.abs(np.array([x1*z[n]+x2 for n in range(n_feat)]))
  elif normalizer == "poly":
    z = z/z.sum()
    probs = z**poly
  else:
    raise("Unknown bootstrap selection methods.")
  probs = probs/probs.sum()

  for n in range(N):
    v[n] = len(np.unique(np.random.choice(range(n_feat), n_feat, p=probs)))
  
  idx = np.argsort(probs)[-int(np.floor(v.mean())):]
  features = all_features[idx]

  return v.mean(), features

def npfs(X, y, n_select, base="mim", alpha=.01, n_bootstraps=100):
  """
  Parameters
  ----------
  X : array-like, shape = (n_samples, n_features_in)
      Sample vectors.
  
  y : array-like, shape = (n_samples,)
      Target vector (class labels).
  
  base : string
      PyFeast feature selection method. ['mim', 'mrmr', 'jmi']
  
  alpha : double
      Size of the hypothesis test for NPFS 
  
  n_bootstraps : double 
      Number of boostraps 

  Returns
  -------
  selections : array 
      Vector of selected features. Length is variable.  
  """
 
  try: 
    fs_method = getattr(feast, base)
  except ImportError: 
    raise("Method does not exist in FEAST")
  
  n_samp, n_feat = X.shape
  
  X = bin_data(X, n_bins=np.sqrt(n_samp))

  if n_samp != len(y):
    ValueError('len(y) and X.shape[0] must be the equal.')

  bern_matrix = np.zeros((n_feat,n_bootstraps))

  for n in range(n_bootstraps):
    # generate a random sample
    idx = np.random.randint(0, n_samp, n_samp)
    sels = fs_method(1.0*X[idx], y[idx], n_select)
    b_sels = np.zeros((n_feat,))
    b_sels[sels] = 1.
    bern_matrix[:, n] = b_sels

  delta = binom.ppf(1-alpha, n_bootstraps, 1.*n_select/n_feat)
  z = np.sum(bern_matrix, axis=1)

  selections = []
  for k in range(n_feat):
    if z[k] > delta:
      selections.append(k)
  
  return selections, bern_matrix, delta


def npfs_chi2(X, y, fpr=0.05, alpha=.01, n_bootstraps=100):
  """
  Parameters
  ----------
  X : array-like, shape = (n_samples, n_features_in)
      Sample vectors.
  
  y : array-like, shape = (n_samples,)
      Target vector (class labels).
  
  fpr : double
      False positive rate for the Chi2-test feature selection approach
  
  alpha : double
      Size of the hypothesis test for NPFS 
  
  n_bootstraps : double 
      Number of boostraps 

  Returns
  -------
  selections : array 
      Vector of selected features. Length is variable.  
  """
  n_samp, n_feat = X.shape
  
  X = bin_data(X, n_bins=np.sqrt(n_samp))

  if n_samp != len(y):
    ValueError('len(y) and X.shape[0] must be the equal.')

  bern_matrix = np.zeros((n_feat,n_bootstraps))

  for n in range(n_bootstraps):
    # generate a random sample
    idx = np.random.randint(0, n_samp, n_samp)
    chi, pval = chi2(1.0*X[idx], y[idx])
    sels = np.where(pval <= fpr)
    b_sels = np.zeros((n_feat,))
    b_sels[sels] = 1.
    bern_matrix[:, n] = b_sels

  delta = binom.ppf(1-alpha, n_bootstraps, fpr)
  z = np.sum(bern_matrix, axis=1)

  selections = []
  for k in range(n_feat):
    if z[k] > delta:
      selections.append(k)

  return selections, bern_matrix, delta  


