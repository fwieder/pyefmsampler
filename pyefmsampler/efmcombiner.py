#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:26:43 2025

@author: frederik
"""

from pyefmsampler import sample_efms

from pyefmsampler_functions import *


import numpy as np
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix
import random

def generate_sparse_int_matrix(n, m, density=0.1, low=-3, high=3, random_state=None):
    rng = np.random.default_rng(random_state)

    def random_integers(size):
        # Generate integers excluding zero
        choices = np.concatenate((np.arange(low, 0), np.arange(1, high + 1)))
        return rng.choice(choices, size=size)

    # Generate sparse matrix with nonzero entries replaced by random integers
    sparse = sparse_random(n, m, density=density, format='csr', random_state=random_state, data_rvs=random_integers)
    return sparse.astype(int)





def combine_efms(efm1,efm2,target,model):
    rev_supp1 = np.intersect1d(supp(model.rev), supp(efm1))
    rev_supp2 = np.intersect1d(supp(model.rev), supp(efm2))
    
    pos1 = np.intersect1d(np.where(efm1 > 0)[0],rev_supp1)
    pos2 = np.intersect1d(np.where(efm2 > 0)[0],rev_supp1)
    
    neg2 = np.intersect1d(np.where(efm1 < 0)[0],rev_supp1)
    neg2 = np.intersect1d(np.where(efm2 < 0)[0],rev_supp2)
    possible_cancels = np.union1d(np.intersect1d(pos1, neg2),np.intersect1d(pos1,neg2))
    combined_supp = np.union1d(supp(efm1),supp(efm2))
    new_efms = []
    new_supps = []
    for cancel_index in possible_cancels:
        blockset = np.union1d(np.setdiff1d(np.arange(model.num_reacs),combined_supp),cancel_index)
        composed_efm =  unsplit_vector(find_efm(model.split_stoich, target, blocked=blockset),model)
        if len(supp(composed_efm>0)) and tuple(supp(composed_efm)) not in new_supps:
            new_efms.append(composed_efm)
            new_supps.append(tuple(supp(composed_efm)))
                
    return new_efms

if __name__ == "__main__":
    S = np.array([[ 1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
              [ 0.,  1.,  1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
              [ 0.,  1.,  0., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0., -1.,  0.],
              [ 0.,  0.,  0.,  0.,  0.,  1., -1., -1.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.]])
    rev = np.array([1,0,1,1,1,0,0,0,1,1,1,1])

    model = FluxCone(S,rev)
    
    efms = np.array([[ 0. ,  0. , -1. ,  0. , -1. ,  0. ,  0. ,  0. , -1. , -1. ,  0. ,
             0. ],
           [ 0.5,  0.5, -1. ,  0. , -0.5,  0.5,  0. ,  0.5,  0. ,  0. ,  0. ,
             0. ],
           [ 0. ,  0. , -1. , -1. , -1. ,  1. ,  0. ,  1. ,  0. ,  0. ,  0. ,
             0. ],
           [ 0. ,  0. ,  1. ,  0. ,  1. ,  0. ,  0. ,  0. ,  1. ,  1. ,  0. ,
             0. ],
           [ 1. ,  1. ,  0. ,  1. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,
             1. ],
           [ 1. ,  1. ,  0. ,  1. ,  1. ,  0. ,  0. ,  0. ,  1. ,  1. ,  0. ,
             0. ],
           [ 1. ,  1. ,  0. ,  0. ,  1. ,  1. ,  1. ,  0. ,  1. ,  1. ,  0. ,
             0. ],
           [ 0.5,  0.5,  0. ,  0. ,  0.5,  0.5,  0. ,  0.5,  1. ,  1. ,  0. ,
             0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ,  1. ,  1. ,  0. ,  0. ,  0. ,  0. ,
             0. ],
           [ 0.5,  0.5,  0. ,  0. ,  0.5,  0.5,  0. ,  0.5,  0. ,  0. ,  1. ,
             1. ],
           [ 1. ,  1. , -1. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
             0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  1. , -1. ,
            -1. ],
           [ 0. ,  0. , -1. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,
            -1. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ,  1. ,  0. ,  1. ,  1. ,  1. ,  0. ,
             0. ],
           [ 1. ,  1. , -1. ,  0. ,  0. ,  1. ,  0. ,  1. ,  1. ,  1. ,  0. ,
             0. ],
           [ 1. ,  1. , -1. ,  0. ,  0. ,  1. ,  1. ,  0. ,  0. ,  0. ,  0. ,
             0. ],
           [ 1. ,  1. , -1. ,  0. ,  0. ,  1. ,  0. ,  1. ,  0. ,  0. ,  1. ,
             1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. , -1. ,  1. ,
             1. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ,  1. ,  0. ,  1. ,  0. ,  0. ,  1. ,
             1. ],
           [ 1. ,  1. ,  0. ,  0. ,  1. ,  1. ,  1. ,  0. ,  0. ,  0. ,  1. ,
             1. ],
           [ 0. ,  0. ,  1. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,
             1. ]])
    
    efm1 = efms[4]
    efm2 = efms[2]
    print(len(combine_efms(efm1, efm2,23, model)))
    
    