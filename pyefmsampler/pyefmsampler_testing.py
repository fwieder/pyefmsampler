#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:56:33 2025

@author: frederik
"""

import numpy as np
import cobra
from pyefmsampler_functions import *
from pyefmsampler import sample_efms,efm_combiner
from pyefmsampler_plots import *
import random
from tqdm import tqdm

if __name__ == "__main__":
    model_id = "/Users/frederik/Documents/pyefmsampler/metamodel_20230510.xml"
    
    cobra_model = cobra.io.read_sbml_model(model_id)
    essential_indices =[20352,20372,20408,20590,21153,21845,21861,22346,23939,24307,24311,24319,24784,27563,27564,27565,27566,27567,27568,27569,48728,48849]
    efm_sample = np.load("/Users/frederik/pyefmsampler_metamodel_10k.npy")
    #efm_supps = [supp(efm) for efm in efm_sample]
    
    model = FluxCone.from_sbml(model_id)
    
    #model_id = "iAF1260"
    #cobra_model = cobra.io.load_model(model_id)
    #model = FluxCone.from_bigg_id(model_id)
    objective_index = find_objective_index(cobra_model)
    #essential_indices = find_essential_reactions(model.split_stoich, objective_index)
    
    
    #efm_sample = sample_efms(model,objective_index,"rf",attempts,max_efms,essential_indices,True)
    #efm_sample = np.array([unsplit_vector(efm,model) for efm in efm_sample])
    
    
    #efms_combined = efm_combiner(model,objective_index,efm_sample,1000,recombine =True)
    for efm in efm_sample:
        print(model.degree(efm))
    #%%

