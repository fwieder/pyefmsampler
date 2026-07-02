from pyefmsampler.helpers import (
    FluxCone,
    find_objective_index,
    find_efm,
    find_essential_reactions,
    supp
)
from pyefmsampler.functions import sample_efms

import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, RLock


# -------------------------------------------------
# IMPORTANT: makes tqdm multiprocessing-safe
# -------------------------------------------------
tqdm.set_lock(RLock())


# -------------------------------------------------
# GLOBAL MODEL (loaded once per worker process)
# -------------------------------------------------
MODEL = None


def init_worker(model_id):
    global MODEL
    MODEL = FluxCone.from_bigg_id(model_id)


# -------------------------------------------------
# WORKER (each has its OWN tqdm bar)
# -------------------------------------------------
def _worker(args):
    objective_index, essential_indices, worker_id = args
    seed = 1234 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    global MODEL

    efms = []

    pbar = tqdm(
        total=2000,
        desc=f"worker {worker_id}",
        position=worker_id,
        leave=True,
        dynamic_ncols=True
    )

    for _ in range(2000):

        batch = sample_efms(
            MODEL,
            objective_index,
            1,
            essential_indices,
            solves_per_blockset=1,
            progress=False
        )

        efms.extend(batch)

        pbar.update(1)

    pbar.close()
    return np.array(efms)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    model_id = "iAF1260"

    print("Loading model (main process)...")
    model = FluxCone.from_bigg_id(model_id)

    objective_index = find_objective_index(model)
    
    essential_indices = np.load("iAF_essential_indices.npy")
    essential_indices = find_essential_reactions(model.split_stoich, objective_index)
    n_tasks = 10
    n_workers = min(cpu_count(), n_tasks)

    args_list = [
        (objective_index, essential_indices, i)
        for i in range(n_tasks)
    ]

    print(f"Running {n_tasks} workers on {n_workers} cores...")

    with Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(model_id,)
    ) as pool:

        samples = pool.map(_worker, args_list)

    from pyefmsampler.plots import umap_efm_sets

    umap_efm_sets(
        samples,
        neighbors=100,
        sample_names=[f"worker {i}" for i in range(len(samples))]
    )

    print("Done.")