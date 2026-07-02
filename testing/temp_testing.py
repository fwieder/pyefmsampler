
from pyefmsampler.helpers import FluxCone,find_objective_index,find_efm,supp
from pyefmsampler.parallel import sample_efms_parallel
from pyefmsampler.functions import sample_efms
import numpy as np
from tqdm import tqdm
import random
from multiprocessing import freeze_support



def efmcomber(start_efms,model,num_of_test = 1000):
    from collections import defaultdict
    efms = []
    supports = set()
    
    positive_at = defaultdict(set)
    negative_at = defaultdict(set)
    def add_vector(vec):
        vec_id = len(efms)
        efms.append(vec)
        for idx in supp(model.rev):
            idx = int(idx)
            val = vec[idx]
            if val > 0:
                positive_at[idx].add(vec_id)

            elif val < 0:
                negative_at[idx].add(vec_id)
    # Initialize the positive_at and negative_at dictionaries with the starting EFMs
    for efm in start_efms:
        add_vector(efm)
        supports.add(tuple(supp(efm)))
    
    for _ in tqdm(range(num_of_test)):
        rev_index = random.choice(list(supp(model.rev)))
        
        if len(positive_at[rev_index]) == 0 or len(negative_at[rev_index]) == 0:
            continue
        
        pos_index = random.choice(list(positive_at[rev_index]))
        neg_index = random.choice(list(negative_at[rev_index]))
        pos_val_at_rev = efms[pos_index][rev_index]
        neg_val_at_rev = efms[neg_index][rev_index]
        
        res = (-neg_val_at_rev) * efms[pos_index] + pos_val_at_rev * efms[neg_index]        
        if np.allclose(res, 0):
            #print("Zero vector encountered, skipping...")
            continue
        if supp(res) in supports:
            #print("Duplicate support encountered, skipping...")
            continue
        if model.rank_test(res) == True:
            add_vector(res)
            supports.add(tuple(supp(res)))
    return efms

if __name__ == "__main__":
    freeze_support()                             # Safer when using windows multiprocessing

    
    model = FluxCone.from_bigg_id("iAF1260")
    objective_index = find_objective_index(model)
    from pyefmsampler.helpers import find_essential_reactions
    #essential_indices = find_essential_reactions(model.split_stoich, objective_index)
    essential_indices = np.load("iAF_essential_indices.npy")
    
    # Sample EFMs using multiple workers
    efm_sample = sample_efms(
        model=model,
        target=objective_index,
        essential_indices=essential_indices,
        max_efms=1000
    )
    efm_sample = [model.unsplit(efms) for efms in efm_sample]
    combs = efmcomber(efm_sample,model,num_of_test = 100000)
    print(len(combs))