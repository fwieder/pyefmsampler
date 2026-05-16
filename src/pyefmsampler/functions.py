#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:13:50 2025

@author: frederik
"""

from pyefmsampler.helpers import find_efm,supp,find_essential_reactions,combine_efms
import numpy as np
import random
from tqdm import tqdm
###############################################################################
# HELPER FUNCTIONS ARE IN "pyefmsampler_functions"
###############################################################################



def sample_efms(model, target, max_efms=1000,
                essential_indices=None,
                solves_per_blockset=1,
                progress=True):

    efms = []
    supports = []
    blocksets = {}

    S = model.split_stoich

    if essential_indices is None:
        print("Determining essential reactions...")
        essential_indices = find_essential_reactions(S, target)
        print(len(essential_indices), "essential reactions found.")

    # ---------------------------
    # THIS MUST ALWAYS RUN
    # ---------------------------
    pbar = tqdm(total=max_efms, desc="Searching EFMs") if progress else None

    attempts = 0
    stagnation_counter = 0
    max_stagnation = 500

    while len(efms) < max_efms:

        if attempts == 0:
            blocked = []
        else:
            if not blocksets:
                print("No more blocksets to explore.")
                break

            block_num = random.choice(list(blocksets.keys()))
            blocked = blocksets[block_num].pop(
                random.randrange(len(blocksets[block_num]))
            )

            if not blocksets[block_num]:
                del blocksets[block_num]

        attempts += 1

        try:
            for _ in range(solves_per_blockset):

                efm = find_efm(
                    S,
                    target,
                    blocked,
                    costs=np.random.rand(S.shape[1])
                )

                s = tuple(supp(efm))

                if s not in supports:
                    efms.append(efm)
                    supports.append(s)

                    if progress:
                        pbar.update(1)

                    for i in s:
                        key = len(blocked) + 1
                        if key not in blocksets:
                            blocksets[key] = set()

                        new_blockset = sorted(blocked + [np.int64(i)])

                        if (i not in essential_indices and
                                new_blockset not in blocksets[key]):
                            blocksets[key].add(new_blockset)

                stagnation_counter = 0

                if len(efms) == max_efms:
                    if progress:
                        pbar.close()
                    return efms

        except ValueError:
            pass

        stagnation_counter += 1

        if stagnation_counter > max_stagnation:
            print("Stopping early due to stagnation.")
            break

        if progress:
            pbar.set_postfix({
                "EFMs Found": len(efms),
                "Largest Blockset": max(blocksets.keys()) if blocksets else 0
            })

    if progress:
        pbar.close()

    return efms

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
