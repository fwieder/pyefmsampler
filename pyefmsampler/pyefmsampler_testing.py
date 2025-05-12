#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:56:33 2025

@author: frederik
"""

import numpy as np
import cobra
from pyefmsampler_functions import *
from pyefmsampler import sample_efms
from pyefmsampler_plots import *
import random
from tqdm import tqdm

if __name__ == "__main__":
    #model_id = "/Users/frederik/Documents/pyefmsampler/metamodel_20230510.xml"
    
    #cobra_model = cobra.io.read_sbml_model(model_id)
    #essential_indices =[20352,20372,20408,20590,21153,21845,21861,22346,23939,24307,24311,24319,24784,27563,27564,27565,27566,27567,27568,27569,48728,48849]
    #efm_sample = np.load("/Users/frederik/pyefmsampler_metamodel_10k.npy")
    #model = FluxCone.from_sbml(model_id)
    
    model_id = "iAF1260"
    cobra_model = cobra.io.load_model(model_id)
    model = FluxCone.from_bigg_id(model_id)
    objective_index = find_objective_index(cobra_model)
    essential_indices = find_essential_reactions(model.split_stoich, objective_index)
    
    attempts = 10000
    max_efms= 1000
    efm_sample = sample_efms(model,objective_index,"df",attempts,max_efms,essential_indices,True)
    efm_sample = np.array([unsplit_vector(efm,model) for efm in efm_sample])
    efm_supps = [supp(efm) for efm in efm_sample]
    for k in [2,10,20,30,40,50,60,70,80,90,100]:
        umap_supps(supports_to_binary_matrix(efm_supps, len(efm_sample[0])),neighbors=k)
    
    import sys
    sys.exit()
    #%%

    
    combined_efms = []
    combined_supps = []
    for i in tqdm(range(attempts)):
        pair = random.sample(range(max_efms),2)
        new_efms = combine_efms(efm_sample[pair[0]],efm_sample[pair[1]],objective_index,model)
        for efm in new_efms:
            if supp(efm) not in combined_supps:
                combined_efms.append(efm)
                combined_supps.append(supp(efm))
    print(len(combined_efms))