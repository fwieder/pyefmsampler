#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:13:50 2025

@author: frederik
"""

from pyefmsampler_functions import find_efm,supp,find_essential_reactions,combine_efms
import numpy as np
import random
from tqdm import tqdm
###############################################################################
# HELPER FUNCTIONS ARE IN "pyefmsampler_functions"
###############################################################################



def sample_efms(model, target,search_strategy:str = "wf", blockset_percent:int = 100 ,max_attempts=1000,max_efms = 1000,essential_indices = [],random_search_direction = False):
    
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
        
    # Initialisation:
    efms = []
    supports = []
    blocksets = [[]]
    
    
    # Expand stoichiometric matrix for reversible reactions
    S = model.split_stoich
    
    opti_dir = np.ones(S.shape[1]) if random_search_direction == False else np.random.rand(S.shape[1])
    
    if essential_indices == []:
        print("Determining essential reactions...")
        essential_indices = find_essential_reactions(S, target)
        print(len(essential_indices), "essential reactions found.")
    
    with tqdm(total=max_attempts, desc="Searching EFMs") as pbar:
        for attempts in range(max_attempts):
            if len(blocksets) == 0:
                return efms
            
            if search_strategy == "rf":
                popindex = random.randint(0,len(blocksets)-1)
            
            blocked = blocksets.pop(popindex)
            pbar.update(1)  # Update progress bar
            
            
            try:
                efm = find_efm(S, target,blocked,costs = opti_dir)

                if supp(efm) not in supports: # and len(supp(efm))>2:
                    efms.append(efm)
                    if len(efms) == max_efms:
                        return efms
                    supports.append(supp(efm))
                    for i in random.sample(supp(efm), round(len(supp(efm))*blockset_percent/100)):
                        if i not in essential_indices and sorted(blocked+[np.int64(i)]) not in blocksets:
                            blocksets.append(sorted(blocked + [np.int64(i)]))
                            
            except ValueError:
                pass
            pbar.set_postfix({"EFMs Found": len(efms),"Remaining blocksets":len(blocksets),"Last EFM Length":len(supp(efm))}) # Update progress bar info
            
    return efms

def efm_combiner(model,objective_index,start_efms,attempts,recombine = False):
    combined_efms = [efm for efm in start_efms]
    combined_supps = [supp(efm) for efm in combined_efms]
    for i in tqdm(range(attempts)):
        if recombine:
            pair = random.sample(range(len(combined_efms)),2)
            
        else:
            pair = random.sample(range(len(start_efms)),2)
        
        new_efms = combine_efms(combined_efms[pair[0]],combined_efms[pair[1]],objective_index,model)

        for efm in new_efms:
            if supp(efm) not in combined_supps:
                combined_efms.append(efm)
                combined_supps.append(supp(efm))
                
    print(len(combined_efms)-len(start_efms), " new EFMs found.")
    return combined_efms
