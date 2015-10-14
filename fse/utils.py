#!/usr/bin/env python

import numpy as np


def similarities(A, B, n):
  """
  Parameters
  ----------
  A : list
      List of features for set 1

  B : list
      List of features for set 2 

  n : int
      Total number of features 

  Returns
  -------
  jaccard : double 
      Similarity between A & B
  
  kuncheva : double 
      Similarity between A & B

  nogueira : double 
      Similarity between A & B
  """
  jaccard = sim_jaccard(A, B)
  if len(A) != len(B):
    kuncheva = None
  else:
    kuncheva = sim_kuncheva(A, B, n)
  nogueira = sim_nogeuira(A, B, n)
  return (jaccard, kuncheva, nogueira)

def sim_kuncheva(A, B, n):
  """
  Parameters
  ----------
  A : list
      List of features for set 1

  B : list
      List of features for set 2 

  n : int
      Total number of features 

  Returns
  -------
  sim : double 
      Similarity between A & B
  """
  A = set(A)
  B = set(B)
  if len(A) != len(B):
    raise("Set cardnalities must be the same.")

  k = 1.*len(A)
  Er = 1.*k1*k2/n
  r = 1.*len(A.intersection(B))
  sim = (r-k**2/n)/(k-k**2/n)
  return None

def sim_nogeuira(A, B, n):
  """
  Parameters
  ----------
  A : list
      List of features for set 1

  B : list
      List of features for set 2 

  n : int
      Total number of features 

  Returns
  -------
  sim : double 
      Similarity between A & B
  """
  A = set(A)
  B = set(B)
  k1 = len(A)
  k2 = len(B)
  Er = k1*k2/n
  r = len(A.intersection(B))
  sim = (r-k1*k2/n)/np.abs(np.min([k1,k2])-k1*k2/n)
  return sim

def sim_jaccard(A, B):
  """
  Parameters
  ----------
  A : list
      List of features for set 1

  B : list
      List of features for set 2 

  Returns
  -------
  sim : double 
      Similarity between A & B
  """
  A = set(A)
  B = set(B)
  num = A.intersection(B)
  den = A.union(B)
  sim = len(num)/len(den)
  return sim

def syn_data(n_features=25, n_observations=100, n_relevant=5):
  """
  Parameters
  ----------

  Returns
  -------
  data : {array-like}, shape = (n_observations, n_features)
      Sample vectors.

  labels : {array-like}, shape = (n_observations, 1)
      Class labels
  """
  xmax = 10
  xmin = 0

  data = 1.0*np.random.randint(xmax + 1, size = (n_features, n_observations))
  delta = n_relevant * (xmax - xmin) / 2.0
  labels = np.zeros((n_observations,))

  for m in range(n_observations):
    zz = 0.0
    for k in range(n_relevant):
      zz += data[k, m]
    if zz > delta:
      labels[m] = 1.
    else:
      labels[m] = 2.
  data = 1.0*data.transpose()

  return (data, labels)


def bin_data(X, n_bins=100):
  """
  Parameters
  ----------
  X : {array-like}, shape = (n_samples, n_features)
      Sample vectors.

  n_bins : integer
      Number of bins to discretize the data into.

  Returns
  -------
  Xo : {array-like}, shape = (n_samples, n_features)
      Sample vectors now discretized.
  """
  Xo = X.copy()
  n_feat = len(Xo[1])
  for n in range(n_feat):
    bins = np.linspace(np.min(X[:,n]), np.max(X[:,n]), n_bins)
    Xo[:,n]= np.digitize(X[:,n], bins)
  return Xo
