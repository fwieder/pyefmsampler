import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, RLock


# ----------------------------
# Worker initialization for tqdm lock
# ----------------------------
def _init_worker(lock):
    tqdm.set_lock(lock)


# ----------------------------
# Patched sample_efms with positioned progress bar
# ----------------------------
def sample_efms_positioned(model, target, max_efms, essential_indices, solves_per_blockset=1):
    """Wrapper around sample_efms that shows progress with proper positioning."""
    from pyefmsampler.helpers import find_efm, supp, find_essential_reactions
    import numpy as np
    import random
    from multiprocessing import current_process
    
    efms = []
    supports = []
    blocksets = {}

    S = model.split_stoich

    if essential_indices is None:
        print("Determining essential reactions...")
        essential_indices = find_essential_reactions(S, target)
        print(len(essential_indices), "essential reactions found.")

    # Get position from process identity
    pos = current_process()._identity[0] - 1
    
    # Create progress bar with dynamic position
    pbar = tqdm(total=max_efms, desc=f"worker {pos}", position=pos,
                leave=False, dynamic_ncols=True)

    attempts = 0
    stagnation_counter = 0
    max_stagnation = 500

    while len(efms) < max_efms:
        if attempts == 0:
            blocked = []
        else:
            if not blocksets:
                break

            block_num = random.choice(list(blocksets.keys()))
            blocked_entry = random.choice(tuple(blocksets[block_num]))
            blocksets[block_num].remove(blocked_entry)
            blocked = list(blocked_entry)

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
                    pbar.update(1)

                    for i in s:
                        if i in supp(model.rev):
                            continue
                        key = len(blocked) + 1
                        if key not in blocksets:
                            blocksets[key] = set()

                        new_blockset = tuple(sorted(blocked + [np.int64(i)]))

                        if (i not in essential_indices and
                                new_blockset not in blocksets[key]):
                            blocksets[key].add(new_blockset)

                stagnation_counter = 0

                if len(efms) == max_efms:
                    pbar.close()
                    return efms

        except ValueError:
            stagnation_counter += 1
            if stagnation_counter >= max_stagnation:
                break

    pbar.close()
    return efms


# ----------------------------
# Worker: samples EFMs with continuous blockset building
# ----------------------------
def _mult_efm_worker(args):
    model, objective_index, essential_indices, max_efms = args

    # Use positioned version
    efms = sample_efms_positioned(
        model,
        objective_index,
        max_efms=max_efms,
        essential_indices=essential_indices,
        solves_per_blockset=1
    )

    return np.array(efms)


# ----------------------------
# Main sampling function
# ----------------------------
def sample_efms_parallel(model, objective_index, essential_indices, 
                     n_tasks=10, n_workers=None, max_efms=2000):
    """
    Sample EFMs using multiple parallel workers with real-time progress tracking.
    
    Parameters
    ----------
    model : FluxCone
        Flux cone model object
    objective_index : int
        Index of the objective reaction
    essential_indices : array-like
        Indices of essential reactions
    n_tasks : int
        Number of workers to spawn (default: 10)
    n_workers : int or None
        Number of parallel processes. If None, uses min(cpu_count(), n_tasks) (default: None)
    max_efms : int
        Total EFMs to sample per worker (default: 2000)
    
    Returns
    -------
    list of np.array
        List where results[i] contains all EFMs sampled by worker i
    """
    
    if n_workers is None:
        n_workers = min(cpu_count()-1, n_tasks)
    
    print(f"Starting {n_tasks} workers with {n_workers} processes...\n")
    
    # Create an RLock for tqdm multiprocessing safety
    lock = RLock()
    
    args_list = [
        (model, objective_index, essential_indices, max_efms)
        for i in range(n_tasks)
    ]

    # Run workers with tqdm lock for multiprocessing safety
    with Pool(n_workers, initializer=_init_worker, initargs=(lock,)) as pool:
        results = pool.map(_mult_efm_worker, args_list)

    return results
