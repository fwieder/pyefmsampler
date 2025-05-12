#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:54:01 2025

@author: frederik
"""


import numpy as np
import cobra
import tqdm
import matplotlib.pyplot as plt

from collections import Counter
from matplotlib.ticker import MaxNLocator
from scipy.optimize import linprog
from pyefmsampler_functions import *
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

def plot_reaction_frequencies(efms,model_id):
    """

    Parameters
    ----------
    efms : np.array
        Set of EFMs
    model_id : str
        Name of the model for title

    Returns
    -------
    Plots the frequency of each reaction in the input-efms as a bar diagram.

    """
    
    rea_freqs = sum([list(supp(efm)) for efm in efms],[])
    
    data = sorted(Counter(rea_freqs).items())
    print(f"{len(data)} different reactions used.")
    
    ax = plt.figure().gca()      # Get the current axes
    ax.set_title(model_id + " - pyefmsampler found EFMs: " + str(len(efms)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Reaction Index")
    ax.set_ylabel("Number of EFMs")
    plt.bar(list(zip(*data))[0],list(zip(*data))[1])
    plt.savefig(model_id + "pyefmsampler_" + model_id +".pdf",dpi=300)
    plt.show()
    

def plot_efm_lengths(efms,model_id):
    """

    Parameters
    ----------
    efms : np.array
        Set of EFMs
    model_id : str
        Name of the model for title

    Returns
    -------
    Plots the lengths of the supports of the input-efms as a bar diagram.

    """
    
    efm_lens = [len(supp(efm)) for efm in efms]
    
    data = sorted(Counter(efm_lens).items())
   
    
    ax = plt.figure().gca()      # Get the current axes
    ax.set_title(model_id + " - pyefmsampler found EFMs: " + str(len(efms)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Efm_length")
    ax.set_ylabel("Number of EFMs")
    plt.bar(list(zip(*data))[0],list(zip(*data))[1])
    plt.savefig(model_id + "pyefmsampler_" + model_id +".pdf",dpi=300)
    plt.show()

def umap_two_samples(efms1,efms2, neighbors = 10):
    
    all_efms = np.r_[efms1,efms2]
    
    # Fit UMAP
    umap_model = umap.UMAP(n_components=2, metric='hamming',n_neighbors=neighbors,min_dist=0.1,random_state=42)

    embedding_full = umap_model.fit_transform(all_efms)
    embedding_sample1 = embedding_full[:len(efms1)]
    embedding_sample2 = embedding_full[len(efms1):]
    
    # Compute trustworthiness score
    trust = trustworthiness(all_efms, embedding_full, n_neighbors=5)
    print(f"Trustworthiness (0-1): {trust:.3f}")

    # Pairwise distance variance (spread)
    pairwise_dist = pairwise_distances(embedding_full)
    spread = np.std(pairwise_dist)
    print(f"Pairwise Distance Std Dev (Spread): {spread:.3f}")
    
    plt.figure(figsize=(10, 7))
    plt.scatter(embedding_sample1[:, 0], embedding_sample1[:, 1], c='blue', alpha=0.5, s=5, label='Sample1')
    plt.scatter(embedding_sample2[:, 0], embedding_sample2[:, 1], c='red' , alpha=0.4, s=5, label='Sample2')

    plt.title("UMAP Projection (Hamming)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(loc='upper right')
    plt.figtext(0.5, -0.05, f"Trustworthiness: {trust:.3f}    Spread: {spread:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    

def umap_supps(efms, labels=None,neighbors = 10):
    
    # Fit UMAP
    umap_model = umap.UMAP(n_components=2, metric='hamming',n_neighbors=neighbors,min_dist=0.1,random_state=42)
    embedding = umap_model.fit_transform(efms)

    # Compute trustworthiness score
    trust = trustworthiness(efms, embedding, n_neighbors=neighbors)


    # Optional: Silhouette score (requires labels)
    if labels is not None:
        sil_score = silhouette_score(embedding, labels)
        print(f"Silhouette Score (0-1): {sil_score:.3f}")

    # Pairwise distance variance (spread)
    pairwise_dist = pairwise_distances(embedding)
    spread = np.std(pairwise_dist)
    print(f"Pairwise Distance Std Dev (Spread): {spread:.3f}")

    # Create a color gradient based on entry order
    n = efms.shape[0]
    colors = np.linspace(0, 1, n)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        cmap='Reds',
        alpha=0.8,
        s=5
    )

    plt.title("UMAP Projection (Hamming Distance) - " + str(len(efms)) + "EFMs")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("EFM Index (Early → Late)")
    
    plt.figtext(0.5, -0.05, f"Trustworthiness (0-1): {trust:.3f}" + "    " + f"Pairwise Distance Std Dev (Spread): {spread:.3f}" + "    " + f"Nearest neighbors: {neighbors}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    return embedding

def umap_supps_sample(full_set, sample, labels=None, neighbors=10):


    # Define a helper to convert rows to consistent tuples of ints
    def row_key(row):
        return tuple(int(x) for x in row)

    # Fit UMAP on full set
    umap_model = umap.UMAP(n_components=2, metric='hamming', n_neighbors=neighbors, min_dist=0.1, random_state=42)
    embedding_full = umap_model.fit_transform(full_set)

    # Map sample indices (requires exact row match)
    full_rows_str = {row_key(row): idx for idx, row in enumerate(full_set)}
    try:
        sample_indices = [full_rows_str[row_key(row)] for row in sample]
    except KeyError as e:
        raise ValueError("Some sample elements are not present in the full set. Please verify subset relationship.") from e

    embedding_sample = embedding_full[sample_indices]

    # Trustworthiness and spread for the sample
    trust = trustworthiness(full_set, embedding_full, n_neighbors=neighbors)
    spread = np.std(pairwise_distances(embedding_full))
    
    # Optional: Silhouette score if labels are provided
    if labels is not None:
        sil_score = silhouette_score(embedding_sample, labels)
        print(f"Silhouette Score (0-1): {sil_score:.3f}")

    # Color gradient for sample points based on appearance
    colors = np.linspace(0, 1, len(sample))

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(embedding_full[:, 0], embedding_full[:, 1], c='lightgray', alpha=0.5, s=5, label='Full set')
    scatter = plt.scatter(embedding_sample[:, 0], embedding_sample[:, 1], c=colors, cmap='Reds', s=5, alpha=0.9, label='Sample subset')

    plt.title("UMAP Projection (Hamming) — Sample vs Full Set")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Sample Index (Early → Late)")
    plt.legend(loc='upper right')
    plt.figtext(0.5, -0.05, f"Trustworthiness: {trust:.3f}    Spread: {spread:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    return embedding_full, embedding_sample