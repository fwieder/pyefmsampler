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
    #model_id = "/Users/frederik/Documents/pyefmsampler/metamodel_20230510.xml"
    #model = FluxCone.from_sbml(model_id)
    
    
    #cobra_model = cobra.io.read_sbml_model(model_id)
    #essential_indices =[20352,20372,20408,20590,21153,21845,21861,22346,23939,24307,24311,24319,24784,27563,27564,27565,27566,27567,27568,27569,48728,48849]
    #efm_sample = np.load("/Users/frederik/pyefmsampler_metamodel_10k.npy")
    #efm_supps = [supp(efm) for efm in efm_sample]
    #objective_index = find_objective_index(cobra_model)
    #combined_efms = efm_combiner(model,objective_index,efm_sample,1000,recombine =True)
    #rf_sample = sample_efms(model,objective_index, "rf" , blockset_percent = 10 , max_attempts = attempts, max_efms = len(combined_efms), essential_indices = essential_indices, random_search_direction = True)

    
    model_id = "e_coli_core"
    cobra_model = cobra.io.load_model(model_id)
    model = FluxCone.from_bigg_id(model_id)
    all_efms = model.get_efms_efmtool()
    

    objective_index = find_objective_index(cobra_model)
    
    biomass_efms = all_efms[np.abs(all_efms[:,objective_index])> 1e-9]
    
    
    essential_indices = find_essential_reactions(model.split_stoich, objective_index)
    blockset_percent = 100
    random_search_direction = True
    
    attempts = 1000000
    max_efms = 100
    
    
    
    #wf_sample = sample_efms(model,objective_index, "wf" , blockset_percent , attempts,max_efms,essential_indices,random_search_direction = False)
    #wf_sample = np.array([unsplit_vector(efm,model) for efm in wf_sample])
    
    #rf_sample = sample_efms(model,objective_index, "rf" , blockset_percent = 100 , max_attempts = attempts, max_efms = max_efms, essential_indices = essential_indices, random_search_direction = True)
    #rf_sample = np.array([unsplit_vector(efm,model) for efm in rf_sample])
    df_sample = sample_efms(model,objective_index, "df" , blockset_percent = 100 , max_attempts = attempts, max_efms = max_efms, essential_indices = essential_indices, random_search_direction = False)
    df_sample = np.array([unsplit_vector(efm,model) for efm in df_sample])
    
    
    
    combined_efms = efm_combiner(model,objective_index,rf_sample,10000,recombine =True)
    
    #df_sample = sample_efms(model,objective_index, "df" , blockset_percent = 100 , max_attempts = attempts, max_efms = len(combined_efms), essential_indices = essential_indices, random_search_direction = False)
    #df_sample = np.array([unsplit_vector(efm,model) for efm in df_sample])
    rf_sample = sample_efms(model,objective_index, "rf" , blockset_percent = 100 , max_attempts = attempts, max_efms = len(combined_efms), essential_indices = essential_indices, random_search_direction = True)
    rf_sample = np.array([unsplit_vector(efm,model) for efm in rf_sample])
    
    
    embedding_full, sample_embeddings = umap_supps_multiple(full_set = biomass_efms,samples = [combined_efms,rf_sample,df_sample],
    names =["combined_efms","random_first","depth_first"], neighbors = 200, title_name="Support Comparison", min_dist = 0.1 )
