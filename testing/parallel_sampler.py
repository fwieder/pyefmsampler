from pyefmsampler_functions import (
    find_efm,
    supp,
    find_essential_reactions
)

import numpy as np
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os


# -------------------------------------------------
# worker: one LP evaluation
# -------------------------------------------------
def _efm_worker(args):
    S, target, blocked, cost_seed = args
    try:
        costs = np.random.default_rng(cost_seed).random(S.shape[1])
        efm = find_efm(S, target, blocked, costs=costs)
        return efm
    except Exception:
        return None


# -------------------------------------------------
# main sampler
# -------------------------------------------------
def sample_efms(model, target, max_efms=1000, essential_indices=None):

    efms = []
    supports = []
    blocksets = {}

    S = model.split_stoich

    if essential_indices is None or len(essential_indices) == 0:
        print("Determining essential reactions...")
        essential_indices = find_essential_reactions(S, target)
        print(len(essential_indices), "essential reactions found.")

    cpu_count = os.cpu_count() or 4
    parallel_costs = 10  # <-- your new idea

    rng = np.random.default_rng()

    with tqdm(total=max_efms, desc="Searching EFMs") as pbar:
        with ProcessPoolExecutor(max_workers=parallel_costs) as pool:

            attempts = 0
            stagnation_counter = 0
            max_stagnation = 500

            while len(efms) < max_efms:

                # ----------------------------------------
                # 1. pick one blockset (same as before)
                # ----------------------------------------
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

                # ----------------------------------------
                # 2. 10 parallel cost evaluations
                # ----------------------------------------
                tasks = [
                    (S, target, blocked, rng.integers(1e9))
                    for _ in range(parallel_costs)
                ]

                results = list(pool.map(_efm_worker, tasks))

                # ----------------------------------------
                # 3. process results
                # ----------------------------------------
                found_any = False

                for efm in results:

                    if efm is None:
                        continue

                    try:
                        s = supp(efm)
                    except Exception:
                        continue

                    if s not in supports:
                        efms.append(efm)
                        supports.append(s)
                        pbar.update(1)
                        found_any = True
                        stagnation_counter = 0

                        if len(efms) >= max_efms:
                            return efms

                        # expand blocksets (same logic)
                        for i in s:
                            key = len(blocked) + 1

                            if key not in blocksets:
                                blocksets[key] = []

                            candidate = sorted(blocked + [np.int64(i)])

                            if (
                                i not in essential_indices
                                and candidate not in blocksets[key]
                            ):
                                blocksets[key].append(candidate)

                # ----------------------------------------
                # 4. stagnation handling
                # ----------------------------------------
                if not found_any:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0

                if stagnation_counter > max_stagnation:
                    print("Stopping due to stagnation.")
                    break

                # ----------------------------------------
                # 5. status
                # ----------------------------------------
                pbar.set_postfix({
                    "EFMs": len(efms),
                    "Blocksets": len(blocksets)
                })

    return efms
if __name__ == "__main__":
    import cobra
    from pyefmsampler_functions import FluxCone,find_objective_index
    
    model_id = "Recon3D"
    cobra_model = cobra.io.load_model(model_id)
    model = FluxCone.from_bigg_id(model_id)
    objective_index = find_objective_index(cobra_model)
    
    essential_indices = [np.int64(9), np.int64(11), np.int64(12), np.int64(13), np.int64(16), np.int64(17), np.int64(20), np.int64(26), np.int64(29), np.int64(30), np.int64(61), np.int64(84), np.int64(85), np.int64(88), np.int64(90), np.int64(119), np.int64(134), np.int64(135), np.int64(136), np.int64(137), np.int64(139), np.int64(154), np.int64(155), np.int64(179), np.int64(180), np.int64(182), np.int64(184), np.int64(193), np.int64(194), np.int64(200), np.int64(211), np.int64(219), np.int64(255), np.int64(274), np.int64(275), np.int64(284), np.int64(286), np.int64(291), np.int64(292), np.int64(321), np.int64(322), np.int64(330), np.int64(344), np.int64(363), np.int64(369), np.int64(387), np.int64(388), np.int64(396), np.int64(401), np.int64(414), np.int64(433), np.int64(434), np.int64(435), np.int64(442), np.int64(448), np.int64(452), np.int64(453), np.int64(456), np.int64(458), np.int64(469), np.int64(488), np.int64(490), np.int64(500), np.int64(504), np.int64(505), np.int64(509), np.int64(516), np.int64(517), np.int64(529), np.int64(532), np.int64(557), np.int64(568), np.int64(574), np.int64(575), np.int64(584), np.int64(591), np.int64(595), np.int64(597), np.int64(598), np.int64(599), np.int64(600), np.int64(602), np.int64(604), np.int64(609), np.int64(611), np.int64(612), np.int64(613), np.int64(614), np.int64(623), np.int64(625), np.int64(626), np.int64(630), np.int64(638), np.int64(644), np.int64(645), np.int64(646), np.int64(654), np.int64(658), np.int64(691), np.int64(925), np.int64(1010), np.int64(1081), np.int64(1089), np.int64(1091), np.int64(1099), np.int64(1100), np.int64(1138), np.int64(1164), np.int64(1177), np.int64(1182), np.int64(1190), np.int64(1203), np.int64(1206), np.int64(1211), np.int64(1245), np.int64(1282), np.int64(1305), np.int64(1307), np.int64(1310), np.int64(1318), np.int64(1362), np.int64(1364), np.int64(1369), np.int64(1378), np.int64(1386), np.int64(1387), np.int64(1390), np.int64(1391), np.int64(1402), np.int64(1412), np.int64(1413), np.int64(1432), np.int64(1440), np.int64(1446), np.int64(1453), np.int64(1456), np.int64(1468), np.int64(1475), np.int64(1483), np.int64(1484), np.int64(1492), np.int64(1493), np.int64(1595), np.int64(1627), np.int64(1628), np.int64(1631), np.int64(1635), np.int64(1636), np.int64(1645), np.int64(1646), np.int64(1652), np.int64(1657), np.int64(1658), np.int64(1670), np.int64(1690), np.int64(1695), np.int64(1772), np.int64(1779), np.int64(1783), np.int64(1793), np.int64(1805), np.int64(1813), np.int64(1821), np.int64(1825), np.int64(1830), np.int64(1831), np.int64(1834), np.int64(1838), np.int64(1842), np.int64(1848), np.int64(1849), np.int64(1853), np.int64(1861), np.int64(1862), np.int64(1863), np.int64(1876), np.int64(1935), np.int64(1952), np.int64(1956), np.int64(1961), np.int64(1962), np.int64(1989), np.int64(1991), np.int64(2006), np.int64(2007), np.int64(2008), np.int64(2026), np.int64(2031), np.int64(2032), np.int64(2033), np.int64(2034), np.int64(2036), np.int64(2037), np.int64(2038), np.int64(2039), np.int64(2041), np.int64(2042), np.int64(2057), np.int64(2060), np.int64(2063), np.int64(2074), np.int64(2112), np.int64(2123), np.int64(2126), np.int64(2132), np.int64(2139), np.int64(2140), np.int64(2141), np.int64(2145), np.int64(2152), np.int64(2158), np.int64(2159), np.int64(2161), np.int64(2162), np.int64(2164), np.int64(2165), np.int64(2179), np.int64(2184), np.int64(2186), np.int64(2204), np.int64(2211), np.int64(2229), np.int64(2231), np.int64(2232), np.int64(2236), np.int64(2237), np.int64(2238), np.int64(2269), np.int64(2303), np.int64(2306), np.int64(2307), np.int64(2320), np.int64(2321), np.int64(2344), np.int64(2345), np.int64(2346), np.int64(2347), np.int64(2348), np.int64(2349), np.int64(2350), np.int64(2351), np.int64(2368), np.int64(2369), np.int64(2375), np.int64(2380), np.int64(2381), np.int64(2432), np.int64(2437), np.int64(2444), np.int64(2453), np.int64(2471), np.int64(2481), np.int64(2491), np.int64(2554), np.int64(2585), np.int64(2587), np.int64(2589), np.int64(2590), np.int64(2591), np.int64(2592), np.int64(2593), np.int64(2595), np.int64(2600), np.int64(2601), np.int64(2603), np.int64(2605), np.int64(2631), np.int64(2665), np.int64(2717), np.int64(2721), np.int64(2724), np.int64(2735), np.int64(2736), np.int64(2738), np.int64(2741), np.int64(2801), np.int64(2818), np.int64(2821), np.int64(2874), np.int64(2927)]

    

    max_efms = 1000
    
    rf_sample = sample_efms(model,objective_index, max_efms = max_efms,essential_indices=essential_indices)
    rf_sample = np.array([unsplit_vector(efm,model) for efm in rf_sample])
    embedding = umap_supps(rf_sample,neighbors=100)
    from sklearn.cluster import DBSCAN

    labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(embedding)
    clusters = {
    k: embedding[labels == k]
    for k in set(labels) if k != -1
}