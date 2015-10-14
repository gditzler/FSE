#!/usr/bin/env python 

import pickle
import numpy as np

from .utils import similarities
from .utils import syn_data
from .feature_selection.ensemble import npfs_chi2
from .feature_selection.ensemble import npfs
from .feature_selection.ensemble import bootstrap_selection


def exp_syn_stability(fname="out.pkl", n_avg=25, n_feat=100, n_obs=250, n_rel=15, n_boots=100, fpr=0.05, alpha=0.1):
  """
  Parameters
  ----------
  fname : string
      Pickle file output 

  n_avg : int
      Number of averages

  n_feat : int
      Number of features

  n_obs : int
      Number of observations

  n_rel : int
      Number of relevant features 

  n_boots : int
      Number of bootstraps

  fpr : double
      False positive rate for Chi2

  alpha : double
      Hypothesis test size for NPFS 


  Returns
  -------
  None
  """
  polies = [1.*x/10 for x in range(60)]
  rels = np.array(range(n_rel))
  samplers = 100

  npfs_ja = 0
  boot_ja = np.zeros((len(polies),))
  nosc_ja = 0

  npfs_ss = 0
  boot_ss = np.zeros((len(polies),))
  nosc_ss = 0

  npfs_no = 0
  boot_no = np.zeros((len(polies),))
  nosc_no = 0

  for p in range(len(polies)):
    poly = polies[p]
    for na in range(n_avg):
      data, labels = syn_data(n_features=n_feat, n_observations=n_obs, n_relevant=n_rel)
      osel, binm, delta = npfs_chi2(data, labels, fpr=fpr, alpha=alpha, n_bootstraps=n_boots)
      
      ss_p, sset_p = bootstrap_selection(binm.sum(axis=1), samplers, normalizer="poly", poly=poly)
      ss_mm, sset_mm = bootstrap_selection(binm.sum(axis=1), samplers, normalizer="minmax")

      npfs_ss += 1.*len(osel)
      boot_ss[p] += 1.*ss_p
      nosc_ss += 1.*ss_mm

      np_ja, np_ku, np_no = similarities(A=rels, B=osel, n=n_feat)
      no_ja, no_ku, no_no = similarities(A=rels, B=sset_mm, n=n_feat)
      bo_ja, bo_ku, bo_no = similarities(A=rels, B=sset_p, n=n_feat)
      
      npfs_ja += np_ja
      boot_ja[p] += bo_ja
      nosc_ja += no_ja

      npfs_no += np_no
      boot_no[p] += bo_no
      nosc_no += no_no

  nosc_ss /= (n_avg*len(polies))
  npfs_ss /= (n_avg*len(polies))
  boot_ss /= n_avg

  nosc_ja /= (n_avg*len(polies))
  npfs_ja /= (n_avg*len(polies))
  boot_ja /= n_avg

  nosc_no /= (n_avg*len(polies))
  npfs_no /= (n_avg*len(polies))
  boot_no /= n_avg

  statistics = {"nosc_ss":nosc_ss,
                "npfs_ss":npfs_ss, 
                "boot_ss":boot_ss,
                "nosc_ja":nosc_ja,
                "npfs_ja":npfs_ja,
                "boot_ja":boot_ja,
                "nosc_no":nosc_no,
                "npfs_no":npfs_no,
                "boot_no":boot_no
                }
  pickle.dump(statistics, open(fname, "wb"))

  return None 

