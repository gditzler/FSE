#!/usr/bin/env python 

import pickle
import numpy as np

from .utils import similarities
from .utils import syn_data
from .feature_selection.ensemble import npfs_chi2
from .feature_selection.ensemble import npfs
from .feature_selection.ensemble import bootstrap_selection
from .feature_selection.single import chi2_selection

def exp_syn_stability(fname="out.pkl", n_avg=25, n_feat=100, n_obs=250, n_rel=15, n_boots=100, fpr=0.05, alpha=0.1, n_select=25):
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
  polies = [1.*x/10 for x in range(120)]
  rels = np.array(range(n_rel))
  samplers = 100

  npfs_ja = 0
  npfs2_ja = 0
  boot_ja = np.zeros((len(polies),))
  nosc_ja = 0
  chi2_ja = 0

  npfs_ss = 0
  npfs2_ss = 0
  boot_ss = np.zeros((len(polies),))
  nosc_ss = 0
  chi2_ss = 0 

  npfs_no = 0
  npfs2_no = 0
  boot_no = np.zeros((len(polies),))
  nosc_no = 0
  chi2_no = 0

  for p in range(len(polies)):
    poly = polies[p]
    for na in range(n_avg):
      data, labels = syn_data(n_features=n_feat, n_observations=n_obs, n_relevant=n_rel)
      osel, binm, delta = npfs_chi2(data, labels, fpr=fpr, alpha=alpha, n_bootstraps=n_boots)
      nsel, binm2, delta2 = npfs(data, labels, n_select=n_select, base="MIM", alpha=alpha, n_bootstraps=n_boots)
      pval, sels = chi2_selection(data, labels)
      sels = np.where(pval <= fpr)[0]
      
      ss_p, sset_p = bootstrap_selection(binm.sum(axis=1), samplers, normalizer="poly", poly=poly)
      ss_mm, sset_mm = bootstrap_selection(binm.sum(axis=1), samplers, normalizer="minmax")

      npfs_ss += 1.*len(osel)
      npfs2_ss += 1.*len(nsel)
      boot_ss[p] += 1.*ss_p
      nosc_ss += 1.*ss_mm
      chi2_ss += 1.*len(sels)

      np_ja, np_ku, np_no = similarities(A=rels, B=osel, n=n_feat)
      np2_ja, np2_ku, np2_no = similarities(A=rels, B=nsel, n=n_feat)
      no_ja, no_ku, no_no = similarities(A=rels, B=sset_mm, n=n_feat)
      bo_ja, bo_ku, bo_no = similarities(A=rels, B=sset_p, n=n_feat)
      ch_ja, ch_ku, ch_no = similarities(A=rels, B=sels, n=n_feat)

      npfs_ja += np_ja
      npfs2_ja += np2_ja
      boot_ja[p] += bo_ja
      nosc_ja += no_ja
      chi2_ja += ch_ja

      npfs_no += np_no
      npfs2_no += np2_no
      boot_no[p] += bo_no
      nosc_no += no_no
      chi2_no += ch_no
  
  nosc_ss /= (n_avg*len(polies))
  npfs_ss /= (n_avg*len(polies))
  npfs2_ss /= (n_avg*len(polies))
  boot_ss /= n_avg
  chi2_ss /= (n_avg*len(polies))

  nosc_ja /= (n_avg*len(polies))
  npfs_ja /= (n_avg*len(polies))
  npfs2_ja /= (n_avg*len(polies))
  boot_ja /= n_avg
  chi2_ja /= (n_avg*len(polies))

  nosc_no /= (n_avg*len(polies))
  npfs_no /= (n_avg*len(polies))
  npfs2_no /= (n_avg*len(polies))
  boot_no /= n_avg
  chi2_no /= (n_avg*len(polies))

  statistics = {"nosc_ss":nosc_ss,
                "npfs_ss":npfs_ss, 
                "boot_ss":boot_ss,
                "chi2_ss":chi2_ss,
                "npfs2_ss":npfs2_ss,
                "nosc_ja":nosc_ja,
                "npfs_ja":npfs_ja,
                "boot_ja":boot_ja,
                "chi2_ja":chi2_ja,
                "npfs2_ja":npfs2_ja,
                "nosc_no":nosc_no,
                "npfs_no":npfs_no,
                "boot_no":boot_no,
                "chi2_no":chi2_no,
                "npfs2_no":npfs2_no,
                "polies":polies
                }
  pickle.dump(statistics, open(fname, "wb"))

  return None 

