#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:13:50 2025

@author: frederik
"""

from pyefmsampler_functions import find_objective_index,find_efm,FluxCone,find_essential_reactions,supp,unsplit_vector,supports_to_binary_matrix, combine_efms
import numpy as np
from scipy.optimize import linprog
import random
import cobra
from tqdm import tqdm
###############################################################################
# HELPER FUNCTIONS ARE IN "pyefmsampler_functions"
###############################################################################



def sample_efms(model, target,search_strategy:str,max_attempts=1000,max_efms = 1000,essential_indices = [],random_search_direction = False):
    
    """
    Repeatedly calls find_efm with randomly chosen blocked sets until
    the desired number of EFMs is found, avoiding duplicate blocked sets
    and supersets of failed sets.
    
    Parameters:
    model: Metabolic model object with stoichiometry and reversibility data.
    target (int): The index of the reaction that must have at least flux 1.
    max_attempts (int): Maximum number of tries before stopping.
    
    Returns:
    list: A list of unique EFMs found.
    """
    
    if search_strategy == "df":
        popindex = -1
    elif search_strategy == "wf":
        popindex = 0
        
    
    efms = []  # Store unique EFMs
    supports = []
    # Expand stoichiometric matrix for reversible reactions
    blocksets = [[]]
    S = model.split_stoich
    opti_dir = np.ones(S.shape[1]) if random_search_direction == False else np.random.rand(S.shape[1])
    if essential_indices == []:
        print("Determining essential reactions...")
        essential_indices = find_essential_reactions(S, target)
        print(len(essential_indices), "essential reactions found.")
    with tqdm(total=max_attempts, desc="Searching EFMs") as pbar:
        for attempts in range(max_attempts):
            #blocked = blocksets.pop(0)
            if len(blocksets) == 0:
                return efms
            
            if search_strategy == "rf":
                popindex = random.randint(0,len(blocksets)-1)
            
            blocked = blocksets.pop(popindex)
            
            pbar.update(1)  # Update progress bar
            
            
            try:
                efm = find_efm(S, target,blocked,costs = opti_dir)

                if tuple(supp(efm)) not in supports: # and len(supp(efm))>2:
                    efms.append(efm)
                    if len(efms) == max_efms:
                        return efms
                    supports.append(tuple(supp(efm)))
                    for i in supp(efm):
                        if i not in essential_indices and sorted(blocked+[np.int64(i)]) not in blocksets:
                            blocksets.append(sorted(blocked + [np.int64(i)]))
            except ValueError:
                pass
            pbar.set_postfix({"EFMs Found": len(efms),"Remaining blocksets":len(blocksets),"Last EFM Length":len(supp(efm))}) # Update progress bar info
            
    return efms

if __name__ == "__main__":
    model_id = "iAF1260"
    cobra_model = cobra.io.load_model(model_id)
    model_id = "/Users/frederik/Documents/pyefmsampler/metamodel_20230510.xml"
    
    cobra_model = cobra.io.read_sbml_model(model_id)
    essential_indices =[20352,20372,20408,20590,21153,21845,21861,22346,23939,24307,24311,24319,24784,27563,27564,27565,27566,27567,27568,27569,48728,48849]
    
    objective_index = find_objective_index(cobra_model)
    
    model = FluxCone.from_sbml(model_id)
    #all_coli_efms = model.get_efms_efmtool()
    #all_coli_supps = [supp(efm) for efm in all_coli_efms if objective_index in supp(efm)]

    num_efms = 100
    efm_sample = sample_efms(model,objective_index,"rf",max_attempts=10000,max_efms = num_efms,random_search_direction=True,essential_indices=essential_indices)
    efm_sample = np.array([unsplit_vector(efm,model) for efm in efm_sample])
    sample_supps = [tuple(supp(efm)) for efm in efm_sample]
    combined_efms = []
    combined_supps = []
    for i in tqdm(range(1000)):
        pair = random.sample(range(num_efms),2)
        new_efms = compose_efm(efm_sample[pair[0]],efm_sample[pair[1]],objective_index,model)
        for efm in new_efms:
            if tuple(supp(efm)) not in combined_supps:
                combined_efms.append(efm)
                combined_supps.append(tuple(supp(efm)))
    import sys
    sys.exit()
    for sample_size in [100,500,2500,5000]:
        umap_supps_sample(supports_to_binary_matrix(all_coli_supps, len(all_coli_efms[0])),supports_to_binary_matrix(sample_supps[:sample_size],len(efm_sample[0])),neighbors=100)
    
    #model_id = "/Users/frederik/Documents/pyefmsampler/metamodel_20230510.xml"
    #cobra_model = cobra.io.read_sbml_model(model_id)
    #essential_indices =[20352,20372,20408,20590,21153,21845,21861,22346,23939,24307,24311,24319,24784,27563,27564,27565,27566,27567,27568,27569,48728,48849]
    
    #model = FluxCone.from_bigg_id(model_id)
    #cobra_model = cobra.io.load_model(model_id)
    
    
    import sys
    
    sys.exit()
#%%    

    f = open("/Users/frederik/EFMSample-27563.list","r")
    lines = [line.rstrip('\n') for line in f]
    supports = [[int(x) for x in line.split(",")] for line in lines]
    f.close()
    max_val = (max([max(sup) for sup in supports]))
    umap_supps(supports_to_binary_matrix(supports, max_val+1),neighbors=100)
    
    
    #%%
    
    sample_efms = np.load("/Users/frederik/pyefmsampler_metamodel_10k.npy")
    sample_supps = [supp(efm) for efm in sample_efms]
    umap_supps(supports_to_binary_matrix(sample_supps,len(sample_efms[0])),neighbors=100)
    