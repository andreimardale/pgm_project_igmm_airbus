from numpy.linalg import cholesky, det, inv, slogdet
from scipy.special import logsumexp
from scipy.special import gammaln
import logging
import math
import numpy as np
import time
import random


from gaussian_components_diag import GaussianComponentsDiag


class IGMM(object):
    def __init__(
            self, X, prior, alpha, K=1, K_max=None):

        self.alpha = alpha
        N, _ = X.shape
        assignments = np.arange(N)
        
        self.components = GaussianComponentsDiag(X, prior, assignments, K_max)


    def draw(self, p_k):
      k_uni = random.random()
      for i in range(len(p_k)):
          k_uni = k_uni - p_k[i]
          if k_uni < 0:
              return i
      return len(p_k) - 1
    
    
    '''
    Implementation of the Gibbs Sampling algorithm for Dirichlet Process Mixture Models. 
    See artifacts/Mardale, Doust, Couge, Ekpo_PGM_Deliverable_2.pdf for the mathematical derivations.
    '''
    def gibbs_sample(self, n_iter):
        record_dict = {}
        record_dict["components"] = []

        for i_iter in range(n_iter):
            for i in range(self.components.N):
                k_old = self.components.assignments[i]
                K_old = self.components.K
                stats_old = self.components.cache_component_stats(k_old)

                self.components.del_item(i)

                log_prob_z = np.zeros(self.components.K + 1, np.float)
                log_prob_z[:self.components.K] = np.log((self.components.counts[:self.components.K]) / (self.components.N + self.alpha -1))+  self.components.log_post_pred(i)
                log_prob_z[-1] = math.log(self.alpha / (self.components.N + self.alpha -1)) + self.components.cached_log_prior[i] 
                # andrei: log-sim-exp trick / like normalization
                prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))
               
                k = self.draw(prob_z)

                if k == k_old and self.components.K == K_old:
                    self.components.restore_component_from_stats(k_old, *stats_old)
                    self.components.assignments[i] = k_old
                else:
                    self.components.add_item(i, k)
              

            record_dict["components"].append(self.components.K - 1)
        return record_dict
