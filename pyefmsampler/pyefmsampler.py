#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:13:50 2025

@author: frederik
"""

from pyefmsampler_functions import find_objective_index,find_efm,FluxCone,find_essential_reactions,supp,unsplit_vector,umap_supps,supports_to_binary_matrix
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
    #model_id = "/Users/frederik/Documents/pyefmsampler/metamodel_20230510.xml"
    #cobra_model = cobra.io.read_sbml_model(model_id)
    #essential_indices =[20352,20372,20408,20590,21153,21845,21861,22346,23939,24307,24311,24319,24784,27563,27564,27565,27566,27567,27568,27569,48728,48849]
    
    model_id = "iAB_RBC_283"
    cobra_model = cobra.io.load_model(model_id)

    stoich = cobra.util.array.create_stoichiometric_matrix(cobra_model)
    rev = np.array([rea.reversibility for rea in cobra_model.reactions]).astype(int)
    objective_index = find_objective_index(cobra_model)
    
    model = FluxCone(stoich, rev)
    essential_indices = find_essential_reactions(model.split_stoich, objective_index)
    for efm_number in [100,200,500,1000,2000]:
        
        sample_efms1 = sample_efms(model,objective_index,"rf",max_attempts = 10000, max_efms = efm_number,essential_indices=essential_indices,random_search_direction=False)
        sample_efms1 = np.array([unsplit_vector(efm,model) for efm in sample_efms1])
        #sample_efms2 = sample_efms(model,objective_index,"rf",max_attempts = 10000, max_efms = efm_number,essential_indices=essential_indices,random_search_direction=True)
        #sample_efms2 = np.array([unsplit_vector(efm,model) for efm in sample_efms2])
        sample_supports1 = [supp(efm) for efm in sample_efms1]
        #sample_supports2 = [supp(efm) for efm in sample_efms2]
        umap_supps(supports_to_binary_matrix(sample_supports1, len(sample_efms1[0])))
        #umap_supps(supports_to_binary_matrix(sample_supports2, len(sample_efms2[0])))
        #pca_efms(sample_efms1)
        #pca_efms(sample_efms2)
        #all_efms = model.get_efms_efmtool()
        #all_supps = [supp(efm) for efm in all_efms]
        #umap_supps(supports_to_binary_matrix(all_supps, len(all_efms[0])))
    