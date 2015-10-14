#!/usr/bin/env pyth 
import numpy as np

from sklearn.feature_selection import chi2 
from sklearn.feature_selection.univariate_selection import SelectFpr 



def chi2_selection(X, y, n_sel=None):
  """
  Parameters
  ----------
  X : array-like, shape = (n_samples, n_features_in)
      Sample vectors.
  
  y : array-like, shape = (n_samples,)
      Target vector (class labels).
  
  """
  if n_sel == None: 
    # simply choose 1/4 of the features if the users have not specified how many 
    n_sel = np.floor(X.shape[1]*0.25)
  chi, pval = chi2(1.0*X, y) 
  
  sels = []
  k = np.argsort(1.-pval)[::-1] # sort from best to worst 
  sels = k[:n_sel]

  return (pval, sels) 
