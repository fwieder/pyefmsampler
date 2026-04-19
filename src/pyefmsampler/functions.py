#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:13:50 2025

@author: frederik
"""

from pyefmsampler.helpers import find_efm,supp,find_essential_reactions,combine_efms,unsplit_vector
import numpy as np
import random
from tqdm import tqdm
###############################################################################
# HELPER FUNCTIONS ARE IN "pyefmsampler_functions"
###############################################################################



def sample_efms(model,target, max_efms = 1000, essential_indices = []):
    
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
        
    # Initialisation:
    efms = []
    supports = []
    blocksets = {}

    
    # Expand stoichiometric matrix for reversible reactions
    S = model.split_stoich
    
    
    if essential_indices == []:
        print("Determining essential reactions...")
        essential_indices = find_essential_reactions(S, target)
        print(len(essential_indices), "essential reactions found.")

    with tqdm(total=max_efms, desc="Searching EFMs") as pbar:
        attempts = 0
        stagnation_counter = 0
        max_stagnation = 500  # safety threshold
        
        while len(efms) < max_efms:
            if attempts == 0:
                blocked = []
            else:
                if not blocksets:
                    print("No more blocksets to explore.")
                    break
            
                block_num = random.choice(list(blocksets.keys()))
                blocked = blocksets[block_num].pop(random.randrange(len(blocksets[block_num])))
                if not blocksets[block_num]:
                    del blocksets[block_num]
            attempts += 1
            
            
            try:
                efm = find_efm(S, target,blocked,costs = np.random.rand(S.shape[1]))

                if supp(efm) not in supports:
                    efms.append(efm)
                    supports.append(supp(efm))

                    pbar.update(1) # Update progress bar
                    
                    stagnation_counter = 0
                    
                    if len(efms) == max_efms:
                        return np.array([unsplit_vector(efm,model) for efm in efms])
                    
                    
                    for i in supp(efm):
                        key = len(blocked) + 1
                        if key not in blocksets:
                            blocksets[key] = []
                        if i not in essential_indices and sorted(blocked+[np.int64(i)]) not in blocksets[key]:
                           
                            blocksets[key].append(sorted(blocked + [np.int64(i)]))
            
            except ValueError:
                pass
            stagnation_counter += 1
            
            if stagnation_counter > max_stagnation:
                print("Stopping early due to stagnation.")
                break
            pbar.set_postfix({"EFMs Found": len(efms),"Largest Blockset":max(blocksets.keys())}) #,"Dimension of sample": curr_dim}) # Update progress bar info
    return np.array([unsplit_vector(efm,model) for efm in efms])

def efm_combiner(model,objective_index,start_efms,max_attempts,max_efms,recombine = False):
    combined_efms = [efm for efm in start_efms]
    combined_supps = [supp(efm) for efm in combined_efms]
    with tqdm(total=max_attempts, desc="Searching EFMs") as pbar:
        for i in range(max_attempts):
            if recombine:
                pair = random.sample(range(len(combined_efms)),2)
                
            else:
                pair = random.sample(range(len(start_efms)),2)
            
            pbar.update(1)  # Update progress bar
            new_efms = combine_efms(combined_efms[pair[0]],combined_efms[pair[1]],objective_index,model)
            
            for efm in new_efms:
                if supp(efm) not in combined_supps:
                    combined_efms.append(efm)
                    combined_supps.append(supp(efm))
                    if len(combined_efms)-len(start_efms) >=max_efms:
                        return combined_efms
            pbar.set_postfix({"EFMs Found": len(combined_efms)}) #,"Dimension of sample": curr_dim}) # Update progress bar info
        print(len(combined_efms)-len(start_efms), " new EFMs found.")
        return combined_efms
